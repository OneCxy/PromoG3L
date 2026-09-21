#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, sys, time, torch, subprocess, inspect, threading, shutil
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, TrainerCallback,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# Model loading
def _prepare_load_kwargs(name_or_path: str):
    is_local = os.path.isdir(name_or_path)
    if is_local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        load_kwargs = dict(trust_remote_code=True, local_files_only=True)
    else:
        load_kwargs = dict(trust_remote_code=True)
    return name_or_path, load_kwargs


# GPU monitoring
def monitor_gpu(interval=60):
    """后台线程定时打印显存使用"""
    def loop():
        while True:
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                )
                used, total = map(int, r.stdout.strip().split(","))
                print(f"[GPU] memory used: {used:4d} MiB / {total:4d} MiB")
            except Exception:
                pass
            time.sleep(interval)
    t = threading.Thread(target=loop, daemon=True)
    t.start()


# Best-model tracking and early stopping
class BestOnlySaverEarlyStop(TrainerCallback):
    """
    - 每次 eval：如果 eval_loss 更低 -> 覆盖保存到 output_dir/_tmp_best
    - 如果连续 patience 次未改善（改善阈值 threshold） -> 提前停止训练
    """
    def __init__(self, out_dir, tokenizer, metric="eval_loss", patience=3, threshold=0.0):
        self.best = None
        self.metric = metric
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.bad = 0
        self.out_dir = out_dir
        self.best_dir = os.path.join(out_dir, "_tmp_best")
        self.tokenizer = tokenizer
        os.makedirs(self.best_dir, exist_ok=True)

    def _is_improved(self, cur):
        # metric = eval_loss，越小越好；threshold 表示至少下降多少才算“提升”
        if self.best is None:
            return True
        return (self.best - cur) > self.threshold

    def _wipe_dir(self, d):
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            return
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)

    def on_evaluate(self, args, state, control, metrics=None, model=None, **_):
        if not metrics or self.metric not in metrics:
            return
        cur = float(metrics[self.metric])

        if self._is_improved(cur):
            self.best = cur
            self.bad = 0
            self._wipe_dir(self.best_dir)
            model.save_pretrained(self.best_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(self.best_dir)
            print(f"💾 New best {self.metric}={cur:.6f} (step={state.global_step}) saved → {self.best_dir}")
        else:
            self.bad += 1
            print(f"⏳ No improvement: cur={cur:.6f}, best={self.best:.6f}, bad={self.bad}/{self.patience}")
            if self.bad >= self.patience:
                print(f"🛑 Early stop: {self.metric} not improved for {self.patience} evals.")
                control.should_training_stop = True


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("fine_tuning_lora_config.json")


def _load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    for key in ("model_name", "train_csv", "validation_csv", "output_dir"):
        value = config.get(key)
        if value and not Path(value).is_absolute():
            config[key] = str(PROJECT_ROOT / value)

    if config.pop("wandb_disabled", False):
        os.environ["WANDB_DISABLED"] = "true"

    print(f"[CONFIG] Loaded: {config_path}")
    return config


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    config_args, remaining_args = config_parser.parse_known_args()
    config = _load_config(config_args.config)

    p = argparse.ArgumentParser("LoRA fine-tuning (local/remote auto-compatible)")
    p.add_argument("--model_name", type=str, required=True, help="HF 仓库名或本地目录")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--validation_csv", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--max_length", type=int, default=200)
    p.add_argument("--pad_to_multiple_of_six", action="store_true")
    p.add_argument("--val_ratio_when_missing", type=float, default=0.05)
    p.add_argument("--monitor_interval", type=int, default=60)
    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj")
    p.add_argument("--merge_lora_on_save", action="store_true")

    p.add_argument("--early_stopping_patience", type=int, default=3,
                   help="连续多少次 eval 指标不提升就早停（按 evaluation_strategy 的频率）")
    p.add_argument("--early_stopping_threshold", type=float, default=0.0,
                   help="提升幅度阈值（eval_loss 至少下降这么多才算提升）")

    boolean_flags = {
        "pad_to_multiple_of_six",
        "use_lora",
        "merge_lora_on_save",
    }
    config_cli_args = []
    for key, value in config.items():
        option = f"--{key}"
        if key in boolean_flags:
            if value:
                config_cli_args.append(option)
        elif value is not None:
            config_cli_args.extend([option, str(value)])

    return p.parse_args(config_cli_args + remaining_args)


# Tokenizer
def setup_tokenizer(name_or_path: str):
    src, kw = _prepare_load_kwargs(name_or_path)
    tok = AutoTokenizer.from_pretrained(src, **kw)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# Dataset
def _detect_col(ex):
    for k in ("sequence", "seq", "dna_sequence", "dna_seq", "text"):
        if k in ex:
            return k
    raise ValueError("No sequence column found")

def setup_dataset(train_csv, val_csv, tokenizer, max_len, pad6, val_ratio):
    files = {"train": train_csv}
    if val_csv:
        files["validation"] = val_csv
    ds = load_dataset("csv", data_files=files)

    def tok_fn(batch):
        col = _detect_col(batch)
        seqs = batch[col]
        eff_len = max_len + ((6 - (max_len % 6)) % 6) if pad6 else max_len
        return tokenizer(
            seqs,
            truncation=True,
            max_length=eff_len,
            padding="longest",
            pad_to_multiple_of=(6 if pad6 else None),
            return_attention_mask=True,
        )

    ds = {k: v.map(tok_fn, batched=True, remove_columns=v.column_names) for k, v in ds.items()}
    train_ds = ds.get("train")
    eval_ds = ds.get("validation")
    if eval_ds is None:
        tmp = train_ds.train_test_split(test_size=val_ratio, seed=42)
        train_ds, eval_ds = tmp["train"], tmp["test"]
        print(f"🪄 Auto-split {val_ratio*100:.1f}% for eval")
    return train_ds, eval_ds


# Model
def setup_model(args):
    src, kw = _prepare_load_kwargs(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(src, **kw)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    if args.use_lora:
        if not HAS_PEFT:
            raise RuntimeError("PEFT not installed. Run: pip install peft")
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
        )
        model = get_peft_model(model, cfg)
        print("✅ LoRA enabled.")
    return model


# Training
def train(args):
    monitor_gpu(args.monitor_interval)

    tok = setup_tokenizer(args.model_name)
    train_ds, val_ds = setup_dataset(args.train_csv, args.validation_csv,
                                     tok, args.max_length,
                                     args.pad_to_multiple_of_six,
                                     args.val_ratio_when_missing)
    model = setup_model(args)
    os.makedirs(args.output_dir, exist_ok=True)

    sig = inspect.signature(TrainingArguments.__init__)
    kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        report_to="none",
    )

    if "bf16" in sig.parameters:
        kwargs["bf16"] = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    # Evaluation is required for best-model selection and early stopping.
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        print("⚠️ Your transformers version is very old (<4.10), skipping eval_strategy.")

    # Keep only the in-memory best state.
    if "save_strategy" in sig.parameters:
        kwargs["save_strategy"] = "no"

    t_args = TrainingArguments(**kwargs)

    best_cb = BestOnlySaverEarlyStop(
        args.output_dir, tok,
        metric="eval_loss",
        patience=args.early_stopping_patience,
        threshold=args.early_stopping_threshold
    )

    trainer = Trainer(
        model=model,
        args=t_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
        callbacks=[best_cb],
    )

    print("🏋️ Start training...")
    t0 = time.time()
    trainer.train()
    print(f"✅ Training done in {(time.time()-t0)/60:.2f} min")

    # Export only the best model.
    best_dir = os.path.join(args.output_dir, "_tmp_best")
    if not os.path.isdir(best_dir) or len(os.listdir(best_dir)) == 0:
        raise RuntimeError(f"[FATAL] best_dir is empty: {best_dir}")

    if args.use_lora and args.merge_lora_on_save:
        # Reload the best adapter before merging.
        if not HAS_PEFT:
            raise RuntimeError("PEFT not installed. Run: pip install peft")
        base_src, base_kw = _prepare_load_kwargs(args.model_name)
        base = AutoModelForCausalLM.from_pretrained(base_src, **base_kw)
        best_model = PeftModel.from_pretrained(base, best_dir)
        try:
            best_model = best_model.merge_and_unload()
            print("🔗 LoRA merged into base model (BEST).")
        except Exception as e:
            raise RuntimeError(f"merge_and_unload failed: {e}")

        for name in os.listdir(args.output_dir):
            p = os.path.join(args.output_dir, name)
            if os.path.abspath(p) == os.path.abspath(best_dir):
                continue
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)

        best_model.save_pretrained(args.output_dir, safe_serialization=True)
        tok.save_pretrained(args.output_dir)
        shutil.rmtree(best_dir, ignore_errors=True)
        print("✨ Best (merged) model saved successfully.")
    else:
        for name in os.listdir(args.output_dir):
            p = os.path.join(args.output_dir, name)
            if os.path.abspath(p) == os.path.abspath(best_dir):
                continue
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)

        for name in os.listdir(best_dir):
            src = os.path.join(best_dir, name)
            dst = os.path.join(args.output_dir, name)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)

        shutil.rmtree(best_dir, ignore_errors=True)
        print("✨ Best model saved successfully.")


if __name__ == "__main__":
    train(parse_args())
