from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.common import load_yaml, read_jsonl, require_path, resolve_path, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="监督微调用户模拟器。")
    parser.add_argument("--config", default="configs/sft.yaml")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml(resolve_path(args.config, project_root))
    set_seed(int(config["seed"]))

    model_path = require_path(resolve_path(config["model_path"], project_root), "基础模型")
    train_path = require_path(resolve_path(config["train_file"], project_root), "监督微调训练集")
    eval_path = resolve_path(config["eval_file"], project_root)
    output_dir = resolve_path(config["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if config.get("bf16", True) and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    peft_config = None
    if config.get("use_lora", True):
        peft_config = LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules=list(config["lora_target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        max_length=int(config["max_length"]),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        warmup_ratio=float(config["warmup_ratio"]),
        weight_decay=float(config["weight_decay"]),
        logging_steps=int(config["logging_steps"]),
        eval_strategy="steps" if eval_path.exists() else "no",
        eval_steps=int(config["eval_steps"]),
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        bf16=bool(config.get("bf16", True) and torch.cuda.is_available()),
        fp16=bool(not config.get("bf16", True) and torch.cuda.is_available()),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        completion_only_loss=True,
        packing=False,
        report_to="tensorboard",
        seed=int(config["seed"]),
    )
    train_dataset = Dataset.from_list(read_jsonl(train_path))
    eval_dataset = Dataset.from_list(read_jsonl(eval_path)) if eval_path.exists() else None
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_or_model_dir = output_dir / ("adapter" if peft_config else "merged")
    trainer.save_model(str(adapter_or_model_dir))
    trainer.accelerator.wait_for_everyone()
    is_main_process = trainer.is_world_process_zero()
    if is_main_process:
        tokenizer.save_pretrained(adapter_or_model_dir)
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if peft_config and is_main_process:
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True, local_files_only=True
        )
        merged = PeftModel.from_pretrained(base, output_dir / "adapter").merge_and_unload()
        merged.save_pretrained(output_dir / "merged", safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(output_dir / "merged")


if __name__ == "__main__":
    main()
