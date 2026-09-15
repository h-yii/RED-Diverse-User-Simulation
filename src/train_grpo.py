from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.common import load_yaml, read_jsonl, require_path, resolve_path, set_seed
from src.rewards import UserSimulatorRewards


def main() -> None:
    parser = argparse.ArgumentParser(description="使用组相对策略优化训练用户模拟器。")
    parser.add_argument("--config", default="configs/grpo.yaml")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml(resolve_path(args.config, project_root))
    set_seed(int(config["seed"]))

    model_path = require_path(resolve_path(config["sft_model_path"], project_root), "监督微调模型")
    train_path = require_path(resolve_path(config["train_file"], project_root), "组相对策略优化训练集")
    eval_path = resolve_path(config["eval_file"], project_root)
    output_dir = resolve_path(config["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_size = int(config.get("num_generations", 8))
    if group_size != 8:
        raise ValueError("当前实现固定每组采样 8 次，请将 num_generations 设为 8。")
    if float(config.get("beta", 0.0)) <= 0:
        raise ValueError("beta 必须大于零，才能保留相对监督微调模型的散度惩罚项。")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rewards = UserSimulatorRewards(config["embedding"], config["reward"], group_size)
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

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch = world_size * int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"])
    if effective_batch % group_size:
        raise ValueError("单卡批大小乘梯度累积步数必须能被每组采样数 8 整除。")
    training_args = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        num_generations=group_size,
        num_generations_eval=group_size,
        # max_prompt_length=int(config["max_prompt_length"]),
        max_completion_length=int(config["max_completion_length"]),
        num_iterations=int(config["num_iterations"]),
        epsilon=float(config["epsilon"]),
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        chat_template_kwargs={"enable_thinking": False},
        beta=float(config["beta"]),
        reward_weights=[float(config["diversity_reward_weight"])],
        scale_rewards=config.get("scale_rewards", "group"),
        multi_objective_aggregation="sum_then_normalize",
        loss_type=str(config.get("loss_type", "dapo")),
        logging_steps=int(config["logging_steps"]),
        eval_strategy="no",
        # eval_steps=5,
        # eval_strategy="steps" if eval_path.exists() else "no",
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        bf16=bool(config.get("bf16", True) and torch.cuda.is_available()),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
        report_to="tensorboard",
        seed=int(config["seed"]),
        model_init_kwargs={
            "torch_dtype": "bfloat16" if config.get("bf16", True) and torch.cuda.is_available() else torch.float32,
            "trust_remote_code": True,
            "local_files_only": True,
            "attn_implementation": "sdpa",
        },
        use_vllm=False,
        vllm_mode="colocate",
    )
    train_dataset = Dataset.from_list(read_jsonl(train_path))
    eval_dataset = Dataset.from_list(read_jsonl(eval_path)) if eval_path.exists() else None
    trainer = GRPOTrainer(
        model=str(model_path),
        reward_funcs=[rewards.diversity_marginal_reward],
        # reward_funcs=[],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_or_model_dir = output_dir / ("adapter" if peft_config else "final")
    trainer.save_model(str(adapter_or_model_dir))
    tokenizer.save_pretrained(adapter_or_model_dir)
    trainer.accelerator.wait_for_everyone()
    is_main_process = trainer.is_world_process_zero()
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if peft_config and is_main_process:
        dtype = torch.bfloat16 if config.get("bf16", True) and torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        merged = PeftModel.from_pretrained(base, adapter_or_model_dir).merge_and_unload()
        merged.save_pretrained(output_dir / "final", safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(output_dir / "final")


if __name__ == "__main__":
    main()
