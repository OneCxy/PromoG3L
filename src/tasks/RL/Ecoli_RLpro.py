#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, re, sys, json, shutil, tempfile, subprocess, warnings, types, random, inspect
from datetime import datetime
from typing import List

import torch
import pandas as pd
import numpy as np
import scipy.stats as spstats
import transformers
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model

warnings.filterwarnings("ignore")
transformers.utils.logging.set_verbosity_error()

RL_DIR = os.path.dirname(os.path.abspath(__file__))
ECOLI_PREDICTOR_DIR = os.path.join(RL_DIR, "expression_reward", "Ecoli_predictor")
TFBS_REWARD_DIR = os.path.join(RL_DIR, "tfbs_reward")

if ECOLI_PREDICTOR_DIR not in sys.path:
    sys.path.insert(0, ECOLI_PREDICTOR_DIR)
if TFBS_REWARD_DIR not in sys.path:
    sys.path.insert(0, TFBS_REWARD_DIR)

from predictor_infer import PREDICT as ActivityPredictor
from motifs_fimo import read_meme, scan


# Paths
POLICY_DIR       = "outputs/low20/base"
PREDICTOR_CKPT   = os.path.join(ECOLI_PREDICTOR_DIR, "LSTMModel.pth")

MEME_PATH        = "datasets/Eclio_Regulon_all.txt"
GLOBAL_VEC_CSV   = "tfbs/re_nature_tfbs.csv"

RL_OUT_ROOT   = "outputs/low20"
BEST_DIR_PATH = os.path.join(RL_OUT_ROOT, "ppo_best")
BEST_META     = os.path.join(RL_OUT_ROOT, "ppo_best_meta.json")

GEN_SCRIPT = os.path.join(os.path.dirname(RL_DIR), "Generation", "generate_seed.py")
os.makedirs(RL_OUT_ROOT, exist_ok=True)


# Reproducibility
RNG = {
    "SEED": 42,
    "DETERMINISTIC": False,
}
GLOBAL_SEED = int(RNG["SEED"])


# Sequence length
BP_PER_TOKEN   = 6
TOKENS_165_RAW = 28               
RAW_BP_LEN     = TOKENS_165_RAW * BP_PER_TOKEN
TARGET_LEN     = 165
TOKENS_GEN = TOKENS_165_RAW


# Training parameters
EVAL_EVERY    = 50
NUM_UPDATES   = 1200
GEN_TEMP      = 1.15
GEN_TOP_P     = 0.9
NUM_EVAL_SEQS = 519

# Evaluation generation parameters
EVAL_GEN_BATCH_SIZE = 16
EVAL_REP_PENALTY    = 1.05
EVAL_RESAMPLE_TRIES = 3

# best 判定
IMPROVE_EPS     = 0.01
PATIENCE_EVALS  = 6

# Activity-anchored reference-point scalarization

# TFBS acceptable level
TAU_B = 0.0

# diversity acceptable level
TAU_D = 0.5

# penalty strength
LAMBDA_CONSTR = 0.6

# penalty sharpness
ALPHA_B = 5.0
ALPHA_D = 3.0

# 熵奖励
ENTROPY_BETA = 0.01

# TFBS
FIMO_THRESHOLD = 1e-3
INVERT_TFBS = False


# TACO replay buffer
class ElitePrioritizedReplay:

    def __init__(self, max_size=256, priority=True, alpha=1.0):
        self.max_size = int(max_size)
        self.priority = bool(priority)
        self.alpha = float(alpha)
        self.memory = []  # [{"dna":str,"q":Tensor[1],"r":Tensor[T],"score":float,"step":int}, ...]

    def add_many(self, dnas, q_ids, r_ids, scores, step: int):
        if self.max_size <= 0:
            return

        for dna, q, r, sc in zip(dnas, q_ids, r_ids, scores):
            self.memory.append({
                "dna": dna,
                "q": q.detach().cpu(),
                "r": r.detach().cpu(),
                "score": float(sc),
                "step": int(step),
            })

        # 去重：同 dna 只保留最高 score
        best = {}
        for it in self.memory:
            dna = it["dna"]
            if (dna not in best) or (it["score"] > best[dna]["score"]):
                best[dna] = it
        self.memory = list(best.values())

        # Top-K 截断
        self.memory.sort(key=lambda x: x["score"], reverse=True)
        if len(self.memory) > self.max_size:
            self.memory = self.memory[:self.max_size]

    def sample(self, n, recent_window=None):
        """
        recent_window: 只从最近 N step 的样本里抽（降低 off-policy）
        """
        pool = self.memory
        if recent_window is not None and len(pool) > 0:
            max_step = max(x["step"] for x in pool)
            lo = max_step - int(recent_window)
            pool = [x for x in pool if x["step"] >= lo]

        if len(pool) < n:
            return None

        if not self.priority:
            idx = np.random.choice(len(pool), size=n, replace=False)
            return [pool[i] for i in idx]

        scores = np.array([x["score"] for x in pool], dtype=np.float64)

        # Keep sampling probabilities nonnegative and normalized.
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.maximum(scores, 0.0)
        scores = scores + 1e-12                   # 避免全 0
        w = np.power(scores, self.alpha)
        s = w.sum()
        if (not np.isfinite(s)) or s <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
            w = w / w.sum()

        idx = np.random.choice(len(pool), size=n, replace=False, p=w)
        return [pool[i] for i in idx]

    def __len__(self):
        return len(self.memory)


# Replay parameters
REPLAY_MAX      = 256          
REPLAY_ALPHA    = 1.0         
REPLAY_ADD_FRAC = 0.5          
REPLAY_FIXED_K  = 2            
REPLAY_MIN_FILL = 32           
REPLAY_RECENT_WINDOW = 200     

replay = ElitePrioritizedReplay(max_size=REPLAY_MAX, priority=True, alpha=REPLAY_ALPHA)
INIT_BEST_SCORE = None


# Adaptive KL parameters
TARGET_KL_PER_TOKEN = 0.12
INIT_KL_COEF        = 0.05
KL_ADAPT_SPEED      = 0.05
KL_COEF_MIN         = 1e-5
KL_COEF_MAX         = 2.0


# Utilities
def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _c(text: str, color: str = "green", bold: bool = True) -> str:
    colors = {
        "black": 30, "red": 31, "green": 32, "yellow": 33,
        "blue": 34, "magenta": 35, "cyan": 36, "white": 37
    }
    c = colors.get(color, 37)
    b = "1" if bold else "0"
    return f"\033[{b};{c}m{text}\033[0m"

def _clean_acgt(s: str) -> str:
    return re.sub(r"[^ACGTacgt]", "", s or "").upper()

def _to_168_then_165(s: str) -> str:
    s = _clean_acgt(s)
    if len(s) < RAW_BP_LEN:
        s = s + ("A" * (RAW_BP_LEN - len(s)))
    return s[:TARGET_LEN]

def _batch_base_type_stats(seqs: List[str]) -> dict:
    uniqs = []
    gc_list = []

    for s in seqs:
        s = _to_168_then_165(s)
        if not s:
            uniqs.append(0)
            gc_list.append(0.0)
            continue

        uniqs.append(len(set(s)))
        gc = (s.count("G") + s.count("C")) / max(len(s), 1)
        gc_list.append(gc)

    uniqs = np.array(uniqs, dtype=np.int64)
    gc_list = np.array(gc_list, dtype=np.float64)

    return {
        "uniq1_ratio": float((uniqs == 1).mean()) if len(uniqs) else 0.0,
        "uniq2_ratio": float((uniqs == 2).mean()) if len(uniqs) else 0.0,
        "uniq3_ratio": float((uniqs == 3).mean()) if len(uniqs) else 0.0,
        "uniq4_ratio": float((uniqs >= 4).mean()) if len(uniqs) else 0.0,
        "uniq_mean": float(uniqs.mean()) if len(uniqs) else 0.0,
        "gc_mean": float(gc_list.mean()) if len(gc_list) else 0.0,
        "gc_std": float(gc_list.std()) if len(gc_list) else 0.0,
    }

def _load_best_score() -> float:
    if not os.path.isfile(BEST_META):
        return float("-inf")
    try:
        with open(BEST_META, "r") as f:
            meta = json.load(f)
        return float(meta.get("best_activity_mean", float("-inf")))
    except Exception:
        return float("-inf")

def _write_best_meta(best_dir: str, score: float):
    with open(BEST_META, "w") as f:
        json.dump(
            {"best_activity_mean": float(score),
             "saved_at": datetime.now().isoformat(),
             "model_dir": best_dir},
            f, indent=2
        )

def save_best_run_config(path: str, ppo_cfg: PPOConfig):
    cfg = {
        "saved_at": datetime.now().isoformat(),
        "seed": {"GLOBAL_SEED": GLOBAL_SEED},
        "ppo": {
            "learning_rate": ppo_cfg.learning_rate,
            "batch_size": ppo_cfg.batch_size,
            "mini_batch_size": ppo_cfg.mini_batch_size,
            "gradient_accumulation_steps": ppo_cfg.gradient_accumulation_steps,
            "kl_penalty": getattr(ppo_cfg, "kl_penalty", None),
            "target_kl": getattr(ppo_cfg, "target_kl", None),
            "init_kl_coef": getattr(ppo_cfg, "init_kl_coef", None),
            "whiten_rewards": getattr(ppo_cfg, "whiten_rewards", None),
            "use_score_norm": getattr(ppo_cfg, "use_score_norm", None),
            "use_score_scaling": getattr(ppo_cfg, "use_score_scaling", None),
        },
        "reward_weights": {"TAU_B": TAU_B, "TAU_D": TAU_D, "LAMBDA_CONSTR": LAMBDA_CONSTR, "ALPHA_B": ALPHA_B, "ALPHA_D": ALPHA_D, "ENTROPY_BETA": ENTROPY_BETA},
        "tfbs": {
            "FIMO_THRESHOLD": FIMO_THRESHOLD,
            "INVERT_TFBS": INVERT_TFBS,
            "MEME_PATH": MEME_PATH,
            "GLOBAL_VEC_CSV": GLOBAL_VEC_CSV,
        },
        "generation": {
            "GEN_TEMP": GEN_TEMP,
            "GEN_TOP_P": GEN_TOP_P,
            "TOKENS_165_RAW": TOKENS_165_RAW,
            "TARGET_LEN": TARGET_LEN,
        },
        "training": {
            "NUM_UPDATES": NUM_UPDATES,
            "EVAL_EVERY": EVAL_EVERY,
            "NUM_EVAL_SEQS": NUM_EVAL_SEQS,
            "IMPROVE_EPS": IMPROVE_EPS,
            "PATIENCE_EVALS": PATIENCE_EVALS,
        },
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": __import__("trl").__version__,
        },
        "replay": {
            "REPLAY_MAX": REPLAY_MAX,
            "REPLAY_ALPHA": REPLAY_ALPHA,
            "REPLAY_ADD_FRAC": REPLAY_ADD_FRAC,
            "REPLAY_FIXED_K": REPLAY_FIXED_K,
            "REPLAY_MIN_FILL": REPLAY_MIN_FILL,
            "REPLAY_RECENT_WINDOW": REPLAY_RECENT_WINDOW,
        },
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[CFG] best run config saved -> {path}")

def _resolve_lm_module(container):
    cands = [container]
    for attr in ("pretrained_model", "model", "base_model", "transformer"):
        if hasattr(container, attr):
            cands.append(getattr(container, attr))
    for c in list(cands):
        for attr in ("pretrained_model", "model", "base_model", "transformer"):
            if hasattr(c, attr):
                cands.append(getattr(c, attr))
    def has_head(x):
        return hasattr(x, "lm_head") or hasattr(x, "get_output_embeddings")
    for m in cands:
        if has_head(m):
            return m
    return container

def export_policy_hf(model_with_value_head, tok, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    base_lm = _resolve_lm_module(model_with_value_head)
    base_lm.save_pretrained(dst_dir, safe_serialization=True)
    tok.save_pretrained(dst_dir)

def bootstrap_best_if_needed() -> str:
    global INIT_BEST_SCORE
    if os.path.isdir(BEST_DIR_PATH):
        INIT_BEST_SCORE = _load_best_score()
        print(f"[BOOT] Found best: {BEST_DIR_PATH}  score={INIT_BEST_SCORE}")
        return BEST_DIR_PATH

    print("[BOOT] Init best from POLICY_DIR")
    if os.path.isdir(BEST_DIR_PATH):
        shutil.rmtree(BEST_DIR_PATH)
    shutil.copytree(POLICY_DIR, BEST_DIR_PATH)

    try:
        INIT_BEST_SCORE = eval_model_dir_activity(POLICY_DIR, num_seqs=NUM_EVAL_SEQS)
        if not np.isfinite(INIT_BEST_SCORE):
            INIT_BEST_SCORE = float("-inf")
        _write_best_meta(BEST_DIR_PATH, INIT_BEST_SCORE)
        print(f"[BOOT] init baseline mean_act={INIT_BEST_SCORE:.4f}")
    except Exception as e:
        INIT_BEST_SCORE = float("-inf")
        print(f"[BOOT][WARN] baseline eval failed: {e}")

    return BEST_DIR_PATH

def _patch_noop_adapter_api_on(obj):
    base = getattr(obj, "pretrained_model", obj)
    for n in ("disable_adapter", "disable_adapters", "enable_adapter", "enable_adapters"):
        if not hasattr(base, n):
            setattr(base, n, (lambda *a, **k: None))

def _class_level_patch():
    try:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        for n in ("disable_adapter","disable_adapters","enable_adapter","enable_adapters"):
            if not hasattr(LlamaForCausalLM, n):
                setattr(LlamaForCausalLM, n, (lambda self, *a, **k: None))
        print("[PATCH] LlamaForCausalLM class patched.")
    except Exception as e:
        print("[PATCH][warn] class-level patch skipped:", e)

def _assert_model_finite(trainer: PPOTrainer):
    with torch.no_grad():
        for name, p in trainer.model.named_parameters():
            if p.requires_grad and (not torch.isfinite(p).all()):
                raise RuntimeError(f"NaN/Inf in params, first bad: {name}")

def _pick(stats_dict, keys, default=np.nan):
    for k in keys:
        if k in stats_dict:
            try:
                return float(stats_dict[k])
            except Exception:
                pass
    return float(default)

def _mean_gen_len(r_all: torch.Tensor, fallback: int) -> float:
    try:
        if isinstance(r_all, torch.Tensor) and r_all.dim() == 2:
            return float(r_all.size(1))
    except Exception:
        pass
    return float(fallback)

def _set_trl_kl_coef(trainer: PPOTrainer, new_coef: float) -> None:
    new_coef = float(new_coef)

    wrote = []
    if hasattr(trainer, "kl_ctl") and hasattr(trainer.kl_ctl, "value"):
        trainer.kl_ctl.value = new_coef
        wrote.append("kl_ctl.value")

    if hasattr(trainer, "config") and hasattr(trainer.config, "init_kl_coef"):
        trainer.config.init_kl_coef = new_coef
        wrote.append("config.init_kl_coef")

    if hasattr(trainer, "kl_coef"):
        try:
            trainer.kl_coef = new_coef
            wrote.append("trainer.kl_coef")
        except Exception:
            pass

    real_ctl = getattr(getattr(trainer, "kl_ctl", None), "value", None)
    real_init = getattr(getattr(trainer, "config", None), "init_kl_coef", None)
    real_target = getattr(getattr(trainer, "config", None), "target_kl", None)

    print(f"[KL-SET] want={new_coef:.6f} wrote={wrote} "
          f"ctl={real_ctl} init_kl_coef={real_init} target_kl={real_target}")

def make_ppo_config_safe(**kwargs) -> PPOConfig:
    sig = inspect.signature(PPOConfig.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = {k: v for k, v in kwargs.items() if k not in allowed}
    if dropped:
        print("[PPOCFG][DROP]", sorted(list(dropped.keys())))
    return PPOConfig(**filtered)


# Reward resources
activity_pred = ActivityPredictor(model_path=PREDICTOR_CKPT)

_all_motifs, _bg = read_meme(MEME_PATH)
_motifs = _all_motifs
print(f"[CHK] motifs loaded: {len(_motifs)}")

_global_df = pd.read_csv(GLOBAL_VEC_CSV)
if _global_df.shape[0] == 0:
    raise ValueError(f"GLOBAL_VEC_CSV 空: {GLOBAL_VEC_CSV}")
_GT_COLS = list(_global_df.columns)
_GT_VEC  = _global_df.iloc[0].to_numpy(dtype=np.float32)
_GT_STD  = float(np.std(_GT_VEC)) if np.isfinite(np.std(_GT_VEC)) else 0.0
print(f"[CHK] global tf vec: shape={_global_df.shape} std={_GT_STD:.6f}")


# Reward functions
@torch.no_grad()
def reward_a_activity(seqs: List[str]) -> torch.Tensor:
    pop = activity_pred.pre_seqs([{"sequence": s, "expression": None} for s in seqs])
    vals = []
    for p in pop:
        expr = p.get("expression")
        vals.append(float(expr) if expr is not None else 0.0)
    t = torch.tensor(vals, dtype=torch.float32)
    return torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)

@torch.no_grad()
def reward_b_tfbs_corr(seqs: List[str]) -> torch.Tensor:
    n = len(seqs)
    if n == 0 or _GT_STD == 0.0 or _GT_VEC is None or len(_GT_VEC) == 0:
        return torch.zeros(n, dtype=torch.float32)

    tfbs = scan(seqs, _motifs, _bg, threshold=FIMO_THRESHOLD, both_strands=True)
    if tfbs is None or len(tfbs) == 0:
        z = torch.zeros(n, dtype=torch.float32)
        return -z if INVERT_TFBS else z

    freq = pd.pivot_table(tfbs, values="start", index="SeqID", columns="Matrix_id", aggfunc="count").fillna(0)
    freq = freq.reindex(columns=_GT_COLS, fill_value=0)

    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        key = str(i)
        if key not in freq.index:
            out[i] = 0.0
            continue
        v = freq.loc[key].to_numpy(dtype=np.float32, copy=False)
        if float(np.std(v)) == 0.0:
            out[i] = 0.0
            continue
        r = spstats.pearsonr(v, _GT_VEC)[0]
        out[i] = r if np.isfinite(r) else 0.0

    t = torch.tensor(out, dtype=torch.float32)
    return -t if INVERT_TFBS else t

@torch.no_grad()
def reward_d_base_types(seqs: List[str]) -> torch.Tensor:

    scores = []
    for s in seqs:
        s = _to_168_then_165(s)
        if not s:
            scores.append(0.0)
            continue

        u = len(set(s))
        collapse_penalty = max(0.0, 3.0 - float(u)) / 2.0
        r_d = 1.0 - collapse_penalty
        scores.append(r_d)

    t = torch.tensor(scores, dtype=torch.float32)
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

@torch.no_grad()
def sequence_entropy(policy, input_ids, response_ids, attention_mask):
    device = next(policy.parameters()).device
    input_ids, response_ids, attention_mask = input_ids.to(device), response_ids.to(device), attention_mask.to(device)
    cat_ids  = torch.cat([input_ids, response_ids], dim=1)
    cat_mask = torch.cat([attention_mask, torch.ones_like(response_ids)], dim=1)
    start = max(input_ids.size(1) - 1, 0)
    end   = start + response_ids.size(1)

    out = policy(input_ids=cat_ids, attention_mask=cat_mask, use_cache=False, return_dict=True)
    logits = out.logits if hasattr(out, "logits") else out[0]
    logp = logits.float().log_softmax(-1)[:, :-1, :][:, start:end, :]
    p = logp.exp()
    ent_each = -(p * logp).sum(-1).mean(1)
    return torch.nan_to_num(ent_each, nan=0.0, posinf=0.0, neginf=0.0)


# Sampling
@torch.no_grad()
def generate_batch_like_training_style(
    model,
    tok,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
):
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    queries = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    attn = torch.ones_like(queries)

    bad_words = []
    if tok.eos_token_id is not None:
        bad_words.append([tok.eos_token_id])

    for attempt in range(3):
        gen = model.generate(
            input_ids=queries,
            attention_mask=attn,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
            eos_token_id=None,
            bad_words_ids=(bad_words or None),
            use_cache=False,
        )
        responses = gen[:, queries.size(1):]
        seqs = [_to_168_then_165(s) for s in tok.batch_decode(responses, skip_special_tokens=True)]
        bad_ratio = sum(len(set(s)) <= 2 for s in seqs) / max(1, len(seqs))
        if bad_ratio <= 0.3 or attempt == 2:
            return queries, attn, responses, seqs

    return queries, attn, responses, seqs


# Evaluation
def _run_generate_for_eval(model_dir_for_eval: str, num_seqs: int) -> List[str]:
    outdir = tempfile.mkdtemp(prefix="ppo_eval_gen_")
    try:
        cmd = [
            sys.executable, GEN_SCRIPT,
            "--model_dir", model_dir_for_eval,
            "--outdir", outdir,
            "--num_seqs", str(num_seqs),
            "--max_new_tokens", str(TOKENS_165_RAW),
            "--temperature", str(GEN_TEMP),
            "--top_p", str(GEN_TOP_P),
            "--batch_size", str(EVAL_GEN_BATCH_SIZE),
            "--repetition_penalty", str(EVAL_REP_PENALTY),
            "--resample_tries", str(EVAL_RESAMPLE_TRIES),
            "--seed", str(GLOBAL_SEED),
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if proc.returncode != 0:
            print("[WARN] generate.py failed:")
            print(proc.stderr)
            return []

        txt_dir = os.path.join(outdir, "TXT")
        if not os.path.isdir(txt_dir):
            print("[WARN] No TXT dir")
            return []

        files = sorted(
            [
                os.path.join(txt_dir, f)
                for f in os.listdir(txt_dir)
                if f.startswith("generated_") and f.endswith(".txt")
            ]
        )
        if not files:
            print("[WARN] No generated_*.txt")
            return []

        latest = files[-1]
        seqs = []
        with open(latest, "r") as fh:
            for line in fh:
                s = _clean_acgt(line.strip())
                if len(s) >= TARGET_LEN:
                    seqs.append(s[:TARGET_LEN])

        return seqs

    finally:
        shutil.rmtree(outdir, ignore_errors=True)

@torch.no_grad()
def eval_model_dir_activity(model_dir_for_eval: str, num_seqs: int) -> float:
    seqs = _run_generate_for_eval(model_dir_for_eval, num_seqs=num_seqs)
    if not seqs:
        return float("-inf")

    vals = reward_a_activity(seqs).cpu().numpy()
    score = float(np.mean(vals)) if vals.size else float("-inf")

    print(f"[EVAL-GEN] got_seqs={len(seqs)} mean_act={score:.6f}")
    return score


def seed_everything(seed: int, deterministic: bool = False):
    import os, random, numpy as np, torch
    from transformers import set_seed

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Training
def main():
    seed_everything(GLOBAL_SEED, deterministic=RNG["DETERMINISTIC"])
    global INIT_BEST_SCORE, replay

    start_dir = bootstrap_best_if_needed()
    print(f"[LOAD] start_dir = {start_dir}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    tok = AutoTokenizer.from_pretrained(start_dir, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        start_dir, trust_remote_code=True, local_files_only=True, torch_dtype=dtype, device_map="auto"
    )
    ref_for_trl = create_reference_model(policy).eval()

    _class_level_patch()
    _patch_noop_adapter_api_on(policy)
    _patch_noop_adapter_api_on(ref_for_trl)

    device = next(policy.parameters()).device

    # predictor 冒烟
    try:
        _ = activity_pred.pre_seqs([{"sequence": "ACGT"*42, "expression": None}])
    except Exception as e:
        print("[PREDICTOR SMOKE TEST] failed:", e)
        return

    # Keep only PPOConfig fields supported by the installed TRL version.
    ppo_cfg = make_ppo_config_safe(
        learning_rate=2e-6,
        batch_size=16,
        mini_batch_size=4,
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
        seed=GLOBAL_SEED,

        whiten_rewards=True,
        use_score_norm=True,
        use_score_scaling=False,

        kl_penalty="kl",
        target_kl=0.03,
        init_kl_coef=INIT_KL_COEF,
    )

    trainer = PPOTrainer(ppo_cfg, policy, ref_for_trl, tok)

    # Initialize the adaptive KL coefficient.
    kl_coef = float(INIT_KL_COEF)
    _set_trl_kl_coef(trainer, kl_coef)
    print(f"[KL-CTRL] KL penalty enabled. target_kl_per_tok={TARGET_KL_PER_TOKEN} init_kl_coef={kl_coef}")

    # Track reward/advantage association.
    orig_compute_advantages = trainer.compute_advantages
    def wrapped_compute_advantages(self, values, rewards, masks):
        v, a, rets = orig_compute_advantages(values, rewards, masks)
        try:
            r_flat = rewards.detach().float().reshape(-1).cpu().numpy()
            a_flat = a.detach().float().reshape(-1).cpu().numpy()
            r_flat = np.nan_to_num(r_flat)
            a_flat = np.nan_to_num(a_flat)
            corr = 0.0
            if np.std(r_flat) > 1e-8 and np.std(a_flat) > 1e-8:
                corr = float(np.corrcoef(r_flat, a_flat)[0, 1])
            print(f"[DBG-ADV] corr(r,adv)={corr:.4f} mean_r={r_flat.mean():.4f} std_r={r_flat.std():.4f} mean_adv={a_flat.mean():.4f}")
        except Exception as e:
            print(f"[DBG-ADV] failed: {e}")
        return v, a, rets

    trainer.compute_advantages = types.MethodType(wrapped_compute_advantages, trainer)
    print("[DBG] patched compute_advantages")

    # Optimize all trainable parameters.
    for _, p in trainer.model.named_parameters():
        p.requires_grad_(True)

    # Use a lower learning rate for the value head.
    from torch.optim import AdamW
    value_params = []
    other_params = []
    for n, p in trainer.model.named_parameters():
        if not p.requires_grad:
            continue
        if ("v_head" in n) or ("value" in n) or ("critic" in n):
            value_params.append(p)
        else:
            other_params.append(p)

    trainer.optimizer = AdamW(
        [
            {"params": other_params, "lr": ppo_cfg.learning_rate, "eps": 1e-8},
            {"params": value_params, "lr": ppo_cfg.learning_rate * 0.1, "eps": 1e-8},
        ]
    )

    best_score_cache = _load_best_score()
    act_ema = None
    no_improve = 0
    LOG_INTERVAL = 50

    for step in range(NUM_UPDATES):
        _assert_model_finite(trainer)

        B = trainer.config.batch_size

        # Mix fresh samples with a small recent replay batch.
        use_replay = len(replay) >= REPLAY_MIN_FILL
        replay_B = min(REPLAY_FIXED_K, B) if use_replay else 0
        fresh_B = B - replay_B

        # (1) fresh sampling
        q_f, attn_f, r_f_ids, seqs_f = generate_batch_like_training_style(
            model=trainer.model,
            tok=tok,
            batch_size=fresh_B,
            max_new_tokens=TOKENS_GEN,
            temperature=GEN_TEMP,
            top_p=GEN_TOP_P,
            device=device,
        )
        q_f, attn_f, r_f_ids = q_f.to(device), attn_f.to(device), r_f_ids.to(device)
        base_stats_f = _batch_base_type_stats(seqs_f)

        r_a = reward_a_activity(seqs_f)
        r_b = reward_b_tfbs_corr(seqs_f)
        r_d = reward_d_base_types(seqs_f)
        ent = sequence_entropy(trainer.model, q_f, r_f_ids, attn_f).to(r_a.device)

        r_a_c = torch.clamp(r_a, -10.0, 10.0)
        r_b_c = torch.clamp(r_b, -1.0, 1.0)
        r_d_c = torch.clamp(r_d, 0.0, 1.0)

        # Reference-point scalarization.

        pen_b = torch.exp(ALPHA_B * (TAU_B - r_b_c))
        pen_d = torch.exp(ALPHA_D * (TAU_D - r_d_c))

        soft_penalty = torch.log1p(pen_b + pen_d)

        raw_total = (r_a_c - LAMBDA_CONSTR * soft_penalty).detach()

        rewards_f = raw_total + ENTROPY_BETA * ent

        # Baseline shift preserves sample ordering.
        reward_center = rewards_f.mean().detach()
        rewards_f = rewards_f - reward_center

        # Guard against near-zero reward variance.
        if float(rewards_f.std(unbiased=False).item()) < 1e-3:
            rewards_f = rewards_f + 1e-3 * torch.randn_like(rewards_f)

        # (2) TACO replay concat：priority 从 elite 抽 replay_B 条，拼进 trainer.step
        samp = None
        if replay_B > 0:
            samp = replay.sample(replay_B, recent_window=REPLAY_RECENT_WINDOW)
            if samp is None:
                replay_B = 0

        if replay_B > 0:
            q_rep = torch.stack([s["q"] for s in samp], 0).to(device)
            r_rep = torch.stack([s["r"] for s in samp], 0).to(device)

            # 用存的 score 构造 replay reward，并对齐当前 step 的中心（避免跨 step 漂）
            rew_rep = torch.tensor([s["score"] for s in samp], dtype=torch.float32, device=r_a.device)
            rew_rep = rew_rep - raw_total.mean().detach()  # 对齐当前 fresh 的 raw_total mean（类似 baseline）

            q_all   = torch.cat([q_f, q_rep], 0)
            r_all   = torch.cat([r_f_ids, r_rep], 0)
            rew_all = torch.cat([rewards_f, rew_rep], 0)
        else:
            q_all, r_all, rew_all = q_f, r_f_ids, rewards_f

        assert q_all.size(0) == B and r_all.size(0) == B and rew_all.size(0) == B
        pre_mean = float(rew_all.mean().item())

        # TRL applies the KL penalty inside the PPO step.
        stats = trainer.step(
            list(q_all.unbind(0)),
            list(r_all.unbind(0)),
            [x.detach().to(r_a.device) for x in rew_all],
        )

        # KL reported by TRL.
        obj_kl_sum = _pick(stats, ["objective/kl"], default=np.nan)

        # Generated length.
        gen_len = _pick(stats, ["tokens/responses_len_mean"], default=_mean_gen_len(r_all, fallback=TOKENS_GEN))

        # Approximate KL is monitoring-only.
        approx_kl = _pick(stats, ["ppo/policy/approxkl", "ppo/policy/approx_kl", "approx_kl"], default=np.nan)

        # Per-token KL.
        kl_per_tok_bt = float(obj_kl_sum / max(gen_len * max(B, 1), 1.0)) if np.isfinite(obj_kl_sum) else float("nan")
        kl_per_tok_t  = float(obj_kl_sum / max(gen_len, 1.0)) if np.isfinite(obj_kl_sum) else float("nan")

        if np.isfinite(kl_per_tok_t) and kl_per_tok_t > 0.5:
            kl_per_tok = kl_per_tok_bt
            kl_mode = "obj/(B*len)"
        else:
            kl_per_tok = kl_per_tok_t
            kl_mode = "obj/len"

        # Additional diagnostics.
        ent_trl  = _pick(stats, ["objective/entropy", "ppo/entropy", "policy/entropy", "entropy"], default=np.nan)
        clipfrac = _pick(stats, ["ppo/policy/clipfrac", "ppo/clipfrac", "clipfrac"], default=np.nan)
        policy_kl = _pick(stats, ["ppo/policy/policykl", "ppo/policy/policy_kl", "policy_kl"], default=np.nan)
        pg_loss   = _pick(stats, ["ppo/loss/policy", "loss/policy", "policy_loss"], default=np.nan)
        v_loss    = _pick(stats, ["ppo/loss/value", "loss/value", "value_loss"], default=np.nan)

        print("[KL-KEYS]", {
            "objective/kl": float(obj_kl_sum) if np.isfinite(obj_kl_sum) else None,
            "gen_len": float(gen_len) if np.isfinite(gen_len) else None,
            "B": int(B),
            "kl_per_tok": float(kl_per_tok) if np.isfinite(kl_per_tok) else None,
            "kl_mode": kl_mode,
            "approxkl(mon)": float(approx_kl) if np.isfinite(approx_kl) else None,
            "kl_coef": float(kl_coef),
        })

        # Adaptive KL update.
        if np.isfinite(kl_per_tok) and TARGET_KL_PER_TOKEN > 0:
            prev_kl_coef = kl_coef
            ratio = float(kl_per_tok / TARGET_KL_PER_TOKEN)
            ratio = float(np.clip(ratio, 0.2, 5.0))

            kl_coef *= float(np.exp(KL_ADAPT_SPEED * (ratio - 1.0)))
            kl_coef = float(np.clip(kl_coef, KL_COEF_MIN, KL_COEF_MAX))
            kl_coef = float(np.clip(kl_coef, prev_kl_coef * 0.5, prev_kl_coef * 2.0))
            _set_trl_kl_coef(trainer, kl_coef)

        # (4) add to elite replay (only fresh) —— 存 fresh top50% 的 raw_total（作为 score）
        with torch.no_grad():
            k = max(1, int(np.ceil(fresh_B * REPLAY_ADD_FRAC)))
            top_idx = torch.topk(raw_total.detach().cpu(), k=k, largest=True).indices.tolist()

        replay.add_many(
            dnas=[seqs_f[i] for i in top_idx],
            q_ids=[q_f[i] for i in top_idx],
            r_ids=[r_f_ids[i] for i in top_idx],
            scores=[raw_total[i].item() for i in top_idx],
            step=step,
        )

        # (4.5) 快速验证：replay 抽到的分数是否明显更高
        if replay_B > 0 and samp is not None:
            rep_scores = [s["score"] for s in samp]
            try:
                print(f"[REPLAY] picked mean_score={float(np.mean(rep_scores)):.4f} max={float(np.max(rep_scores)):.4f} "
                      f"fresh_mean_raw={float(raw_total.mean().item()):.4f} replay_size={len(replay)}")
            except Exception:
                pass

        # # (5) post sampling quick check (activity only)
        # with torch.no_grad():
        #     _, _, _, seqs_after = generate_batch_like_training_style(
        #         model=trainer.model, tok=tok, batch_size=B,
        #         max_new_tokens=TOKENS_84BP, temperature=GEN_TEMP, top_p=GEN_TOP_P, device=device
        #     )
        #     post_act = float(torch.clamp(reward_a_activity(seqs_after), -10.0, 10.0).mean().item())

        cur_act_mean = float(r_a.mean().item())
        act_ema = cur_act_mean if act_ema is None else (0.9 * act_ema + 0.1 * cur_act_mean)

        if (step % LOG_INTERVAL == 0) or (step + 1 == NUM_UPDATES):
            rb_min = float(r_b.min().item())
            rb_max = float(r_b.max().item())
            rb_std = float(r_b.std(unbiased=False).item())
            rb_nonzero_ratio = float((r_b.abs() > 1e-6).float().mean().item())

            pen_mean = float(soft_penalty.mean().item())
            total_contrib = float(rewards_f.abs().mean().item())
            ratio_contrib = (pen_mean / total_contrib) if total_contrib > 1e-9 else 0.0

            print(
                f"[TFBS-DIAG] Range=[{rb_min:.3f}, {rb_max:.3f}] "
                f"Std={rb_std:.3f} "
                f"Cover(!=0)={rb_nonzero_ratio:.1%} "
                f"Penalty_Ratio={ratio_contrib:.1%}"
            )
            if rb_std < 1e-6 and rb_max < 1e-6:
                print(_c("[WARN] TFBS reward is SILENT (all zeros)! Check FIMO/Motifs.", "red", False))

            print(
                f"[DBG-REWARD] r_a(mean={float(r_a.mean()):.3f}, std={float(r_a.std(unbiased=False)):.3f}) "
                f"[SCALAR] penalty_mean={soft_penalty.mean().item():.4f}"
                f"r_b(mean={float(r_b.mean()):.3f}) r_d(mean={float(r_d.mean()):.3f}) "
                f"ent_bonus_mean={float(ent.mean()):.3f} reward_center={float(reward_center.item()):.3f}"
            )
            print(
                f"[BASE-TYPE] uniq_mean={base_stats_f['uniq_mean']:.3f} "
                f"u1={base_stats_f['uniq1_ratio']:.1%} "
                f"u2={base_stats_f['uniq2_ratio']:.1%} "
                f"u3={base_stats_f['uniq3_ratio']:.1%} "
                f"u4={base_stats_f['uniq4_ratio']:.1%} "
                f"GC(mean={base_stats_f['gc_mean']:.3f}, std={base_stats_f['gc_std']:.3f})"
            )
            print(
                "[DBG-PPO] "
                f"approx_kl={approx_kl:.4e} policy_kl={policy_kl:.4e} "
                f"clipfrac={clipfrac:.3f} policy_loss={pg_loss:.4f} value_loss={v_loss:.4f}"
            )
            print(f"[ABS-ACT] mean_fresh={cur_act_mean:.4f} ema={act_ema:.4f}")
            print(
                f"[KL-MON] kl_per_tok={kl_per_tok:.6e} (target={TARGET_KL_PER_TOKEN:.3e}) "
                f"objective/kl={obj_kl_sum:.6e} gen_len≈{gen_len:.1f} approxkl(mon)={approx_kl:.6e} "
                f"kl_coef={kl_coef:.6f} mode={kl_mode}"
            )
            print(
                f"[step {step+1:3d}/{NUM_UPDATES}] reward_pre_mean={pre_mean:.4f} "
                f"a_mean={cur_act_mean:.4f} b_mean={float(r_b.mean()):.4f} d_mean={float(r_d.mean()):.4f} "
                f"TRL_entropy={ent_trl:.4f} replay={len(replay)} used_replay={replay_B} best≈{best_score_cache:.6f}"
            )

        # (6) periodic eval + early stop
        if ((step + 1) % EVAL_EVERY == 0) or (step + 1 == NUM_UPDATES):
            tmp_dir = os.path.abspath(os.path.join(RL_OUT_ROOT, f"_tmp_eval_{_now_tag()}_{step+1:06d}"))
            export_policy_hf(trainer.model, tok, tmp_dir)

            new_score = eval_model_dir_activity(tmp_dir, num_seqs=NUM_EVAL_SEQS)
            old_best = _load_best_score()

            baseline = INIT_BEST_SCORE if (INIT_BEST_SCORE is not None and np.isfinite(INIT_BEST_SCORE)) else old_best
            delta_init = new_score - baseline if (baseline is not None and np.isfinite(baseline)) else float("nan")
            delta_best = new_score - old_best if np.isfinite(old_best) else float("nan")

            msg = f"[EVAL] step={step + 1} mean_act={new_score:.4f} Δ_vs_init={delta_init:.4f} Δ_vs_best={delta_best:.4f}"
            print(_c(msg, "green", True))

            improved = (not np.isfinite(old_best)) or (new_score > old_best + IMPROVE_EPS)
            if improved:
                if os.path.isdir(BEST_DIR_PATH):
                    shutil.rmtree(BEST_DIR_PATH)
                shutil.move(tmp_dir, BEST_DIR_PATH)
                _write_best_meta(BEST_DIR_PATH, new_score)
                best_score_cache = new_score
                no_improve = 0
                msg = f"[BEST] promoted {BEST_DIR_PATH} score={new_score:.6f}"
                print(_c(msg, "green", True))
                save_best_run_config(
                    os.path.join(BEST_DIR_PATH, "best_config.json"),
                    ppo_cfg
                )
            else:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                no_improve += 1
                msg = f"[BEST] keep {old_best:.6f} vs {new_score:.6f}, no_improve={no_improve}/{PATIENCE_EVALS}"
                print(_c(msg, "yellow", True))
                if no_improve >= PATIENCE_EVALS:
                    print(_c("[EARLY-STOP] no significant improvement, stopping.", "red", True))
                    break

    print("✅ PPO 完成")
    print(f"📌 最佳模型目录：{BEST_DIR_PATH if os.path.isdir(BEST_DIR_PATH) else '(暂无)'}")
    print(f"📄 最佳记录：{BEST_META if os.path.isfile(BEST_META) else '(暂无)'}")


if __name__ == "__main__":
    main()
