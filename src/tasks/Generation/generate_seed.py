#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import csv
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList


# Sequence-length constants
TOKEN_BP = 6
TOKENS_FOR_168 = 28
REQUIRED_BP = TOKENS_FOR_168 * TOKEN_BP   # 168
TARGET_LEN = 165
MAX_RESAMPLE = 3

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("generate_seed_config.json")


def _is_absolute_path(value):
    return (
        Path(value).is_absolute()
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    )


def _load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    for key in ("model_dir", "outdir"):
        value = config.get(key)
        if value and not _is_absolute_path(value):
            config[key] = str(PROJECT_ROOT / value)

    print(f"[CONFIG] Loaded: {config_path}")
    return config


# Utilities
def clean_acgt(s: str) -> str:
    return re.sub(r"[^ACGTacgt]", "", s or "").upper()


def _auto_seed() -> int:
    raw = int.from_bytes(os.urandom(8), "little") ^ (time.time_ns() & 0xFFFFFFFF)
    return raw & 0xFFFFFFFF


def hard_reset_rng(seed: int, device: str) -> int:
    seed = int(seed) & 0xFFFFFFFF
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)
    return seed


def load_tokenizer(model_dir: str):
    try:
        tok = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            k=TOKEN_BP,
        )
    except TypeError:
        try:
            tok = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            tok = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
            )

    # 兼容某些 tokenizer 缺少 _added_tokens_decoder
    if not hasattr(tok, "_added_tokens_decoder"):
        try:
            base = object.__getattribute__(tok, "added_tokens_decoder")
        except Exception:
            base = {}
        if base is None or not isinstance(base, dict):
            base = {}
        tok._added_tokens_decoder = base

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok


def load_model(model_dir: str, device: str):
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    return model.to(device).eval()


# 6-mer vocabulary constraint
class VocabMaskProcessor(LogitsProcessor):
    def __init__(self, vocab_size: int, allow_ids: torch.LongTensor):
        super().__init__()
        self.vocab_size = int(vocab_size)
        mask = torch.full((self.vocab_size,), float("-inf"))
        mask[allow_ids.long()] = 0.0
        self.registered_mask = mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        mask = self.registered_mask.to(scores.device)
        return scores + mask


def build_allowed_6mer_ids(tok) -> torch.LongTensor:
    allow = []

    vocab_size = getattr(tok, "vocab_size", None)
    if vocab_size is None:
        items = tok.get_vocab().items()
        for token, tid in items:
            s = str(token)
            if len(s) == 6 and re.fullmatch(r"[ACGT]+", s):
                allow.append(int(tid))
    else:
        ids = list(range(vocab_size))
        toks = tok.convert_ids_to_tokens(ids)
        for tid, s in zip(ids, toks):
            s = "" if s is None else str(s)
            if len(s) == 6 and re.fullmatch(r"[ACGT]+", s):
                allow.append(int(tid))

    if not allow:
        raise RuntimeError("未找到任何符合条件的 6-mer ACGT token，请确认词表是否为 6-mer。")

    return torch.LongTensor(sorted(set(allow)))


# CLI
def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, remaining_args = config_parser.parse_known_args()

    config_path = config_args.config
    if config_path is None and not remaining_args:
        config_path = str(DEFAULT_CONFIG_PATH)
    config = _load_config(config_path) if config_path else {}

    ap = argparse.ArgumentParser()

    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--num_seqs", type=int, default=100)
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=TOKENS_FOR_168,
        help="每个样本生成的 token 数；默认 28，对应约 168bp。",
    )
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top_p", type=float, default=0.90)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)

    ap.add_argument(
        "--restrict_6mer",
        dest="restrict_6mer",
        action="store_true",
        help="仅允许 6-mer 的 A/C/G/T 词元被采样。",
    )
    ap.add_argument(
        "--no_restrict_6mer",
        dest="restrict_6mer",
        action="store_false",
        help="关闭 6-mer ACGT 词表限制。",
    )
    ap.set_defaults(restrict_6mer=True)

    ap.add_argument("--seed", type=int, default=-1, help="-1=自动随机；>=0=固定种子")
    ap.add_argument("--pad_to_6", action="store_true", help="已废弃；本脚本不会做任何 A-padding")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--resample_tries",
        type=int,
        default=MAX_RESAMPLE,
        help="单条样本若清洗后长度 < 168bp，最多重采样多少次。",
    )

    config_cli_args = []
    for key, value in config.items():
        if key == "restrict_6mer":
            config_cli_args.append("--restrict_6mer" if value else "--no_restrict_6mer")
        elif key == "pad_to_6":
            if value:
                config_cli_args.append("--pad_to_6")
        elif value is not None:
            config_cli_args.extend([f"--{key}", str(value)])

    args = ap.parse_args(config_cli_args + remaining_args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "没有可用的 GPU，请检查环境配置。"

    if args.seed is None or args.seed < 0:
        base_seed = _auto_seed()
        reproducible = False
    else:
        base_seed = int(args.seed) & 0xFFFFFFFF
        reproducible = True

    hard_reset_rng(base_seed, device)

    if device.startswith("cuda"):
        torch.backends.cudnn.deterministic = reproducible
        torch.backends.cudnn.benchmark = not reproducible

    outdir = Path(args.outdir)
    (outdir / "FASTA").mkdir(parents=True, exist_ok=True)
    (outdir / "CSV").mkdir(parents=True, exist_ok=True)
    (outdir / "TXT").mkdir(parents=True, exist_ok=True)

    if args.pad_to_6:
        print("⚠️  `--pad_to_6` 已被忽略：本脚本不做任何 A-padding。")

    print(f"[INFO] Loading tokenizer from {args.model_dir}")
    tok = load_tokenizer(args.model_dir)

    print(f"[INFO] Loading model from {args.model_dir} on {device}")
    model = load_model(args.model_dir, device)

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model.config.pad_token_id = tok.pad_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.eos_token_id

    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id

    logits_processors = LogitsProcessorList()
    if args.restrict_6mer:
        allow_ids = build_allowed_6mer_ids(tok)
        vocab_size = getattr(tok, "vocab_size", None)
        if vocab_size is None:
            vocab_size = int(max(allow_ids).item()) + 1

        logits_processors.append(
            VocabMaskProcessor(
                vocab_size=vocab_size,
                allow_ids=allow_ids,
            )
        )
        print(f"[INFO] Restricting vocab to {allow_ids.numel()} valid 6-mer ACGT tokens.")

    seqs = []
    B = max(1, int(args.batch_size))
    total = int(args.num_seqs)

    if args.max_new_tokens != TOKENS_FOR_168:
        print(
            f"⚠️  当前 `--max_new_tokens={args.max_new_tokens}`；"
            f"若你要严格执行 168→165 协议，建议设为 {TOKENS_FOR_168}。"
        )

    mode_str = "fixed" if reproducible else "auto"
    print(f"[INFO] RNG mode={mode_str}, base_seed={base_seed}")
    print(
        f"[INFO] Generation policy: generate {args.max_new_tokens} tokens "
        f"(≈{args.max_new_tokens * TOKEN_BP}bp), clean, keep only sequences with "
        f"length >= {REQUIRED_BP}bp, then trim to {TARGET_LEN}bp; "
        f"resample up to {args.resample_tries} times if too short; no A-padding."
    )
    print(
        f"[INFO] total={total}, batch_size={B}, max_new_tokens={args.max_new_tokens}, "
        f"temp={args.temperature}, top_p={args.top_p}, repetition_penalty={args.repetition_penalty}"
    )

    def gen_once(input_ids, attn_mask, offset: int):
        """
        对一次 generate 调用进行独立 seed 控制。
        优先尝试给 model.generate 显式传 generator；
        若当前 transformers 版本不支持，则退回到全局 RNG reset。
        """
        if reproducible:
            local_seed = (base_seed + offset) & 0xFFFFFFFF
        else:
            local_seed = _auto_seed()

        hard_reset_rng(local_seed, device)

        bad_words = None
        if eos_id is not None:
            bad_words = [[eos_id]]

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=None,
            bad_words_ids=bad_words,
            use_cache=False,
            logits_processor=logits_processors,
            repetition_penalty=args.repetition_penalty,
        )

        hard_reset_rng(local_seed, device)

        # Some supported transformers versions do not accept generator here.
        g_out = model.generate(**gen_kwargs)

        new_only = g_out[:, input_ids.size(1):]
        outs = tok.batch_decode(new_only, skip_special_tokens=True)
        return [clean_acgt(s) for s in outs]

    with torch.no_grad():
        global_offset = 0
        kept = 0

        while kept < total:
            b = min(B, total - kept)

            input_ids = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
            attn_mask = torch.ones((b, 1), dtype=torch.long, device=device)

            batch = gen_once(input_ids, attn_mask, offset=global_offset)
            global_offset += 1

            for i in range(b):
                s = batch[i]
                tries = 0

                while len(s) < REQUIRED_BP and tries < args.resample_tries:
                    one = gen_once(
                        input_ids[i:i + 1],
                        attn_mask[i:i + 1],
                        offset=global_offset,
                    )[0]
                    global_offset += 1
                    s = one
                    tries += 1

                if len(s) >= REQUIRED_BP:
                    seqs.append(s[:TARGET_LEN])   # 168 -> 165
                    kept += 1
                # 若仍不足 168bp：直接丢弃，不做 A-padding

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fasta_path = outdir / f"FASTA/generated_{ts}.fasta"
    csv_path = outdir / f"CSV/generated_{ts}.csv"
    txt_path = outdir / f"TXT/generated_{ts}.txt"

    with open(fasta_path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">gen_{i}\n{s}\n")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence"])
        for s in seqs:
            w.writerow([s])

    with open(txt_path, "w") as f:
        for s in seqs:
            f.write(s + "\n")

    print(f"[OK] Wrote {len(seqs)} sequences (each {TARGET_LEN} bp; strategy: 168→165; no A-padding):")
    print(f"     FASTA: {fasta_path}")
    print(f"     CSV  : {csv_path}")
    print(f"     TXT  : {txt_path}")


if __name__ == "__main__":
    main()
