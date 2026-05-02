"""
SFT training for 1-shot experiments with GEM or CE loss.

Uses HuggingFace Trainer with a custom loss override for GEM.
Each "step" = one full pass over the dataset (1 epoch).

Usage:
    torchrun --nproc_per_node=4 src/gem/train.py \
        --model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --dataset data/sft_1shot_datasets/standard_pi1/problem_0000.parquet \
        --loss gem --num_steps 1000
"""

import argparse
import json
import os
import random
import time

import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from gem.gem_loss import ce_loss_causal_lm, gem_loss_causal_lm


def parse_args():
    p = argparse.ArgumentParser(description="1-Shot SFT Training (GEM/CE)")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="models/sft_1shot")
    p.add_argument("--loss", type=str, choices=["gem", "ce"], default="gem")
    p.add_argument("--gem_beta", type=float, default=0.7)
    p.add_argument("--num_steps", type=int, default=1000, help="Number of epochs (each = full dataset pass)")
    p.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_every_n_steps", type=int, default=100)
    p.add_argument("--eval_every_n_steps", type=int, default=50)
    p.add_argument("--eval_dataset", type=str, default="data/math500.parquet")
    p.add_argument("--eval_samples", type=int, default=50)
    p.add_argument("--eval_max_new_tokens", type=int, default=2048)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def tokenize_dataset(dataset, tokenizer, max_seq_length):
    def tokenize_fn(examples):
        all_input_ids, all_attention_mask, all_labels = [], [], []
        response_key = "response" if "response" in examples else "completion"

        for prompt, response in zip(examples["prompt"], examples[response_key]):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                response_ids = response_ids + [tokenizer.eos_token_id]

            combined = prompt_ids + response_ids
            if len(combined) > max_seq_length:
                combined = combined[:max_seq_length]
                prompt_len = min(len(prompt_ids), max_seq_length)
            else:
                prompt_len = len(prompt_ids)

            labels = [-100] * prompt_len + combined[prompt_len:]
            pad_len = max_seq_length - len(combined)
            pad_id = tokenizer.pad_token_id or 0

            all_input_ids.append(combined + [pad_id] * pad_len)
            all_attention_mask.append([1] * len(combined) + [0] * pad_len)
            all_labels.append(labels + [-100] * pad_len)

        return {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels}

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")
    dataset.set_format("torch")
    return dataset


# ---------------------------------------------------------------------------
# Custom Trainer with GEM loss
# ---------------------------------------------------------------------------

class GEMTrainer(Trainer):
    def __init__(self, gem_beta=0.7, loss_type="gem", **kwargs):
        super().__init__(**kwargs)
        self.gem_beta = gem_beta
        self.loss_type = loss_type

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.loss_type == "gem":
            loss = gem_loss_causal_lm(logits, labels, beta=self.gem_beta)
        else:
            loss = ce_loss_causal_lm(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Eval callback — runs MATH500 evaluation at specified step intervals
# ---------------------------------------------------------------------------

def load_math500_problems(path, num_samples=0):
    df = pd.read_parquet(path)
    problems = []
    for _, row in df.iterrows():
        prompt_raw = row["prompt"]
        if isinstance(prompt_raw, str):
            prompt_raw = json.loads(prompt_raw)
        gt = row.get("reward_model", {})
        if isinstance(gt, str):
            gt = json.loads(gt)
        gt_answer = gt.get("ground_truth", str(gt))
        prompt_text = "\n".join(m["content"] for m in prompt_raw if m["role"] in ("user", "system"))
        problems.append({"prompt_text": prompt_text, "ground_truth": gt_answer})

    if num_samples > 0:
        random.seed(42)
        problems = random.sample(problems, min(num_samples, len(problems)))
    return problems


def extract_boxed_answer(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i, depth = idx, 0
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[text.index("{", idx) + 1:i]
        i += 1
    return None


class Math500EvalCallback(TrainerCallback):
    def __init__(self, eval_problems, tokenizer, eval_every_n_epochs, max_new_tokens, steps_per_epoch):
        self.eval_problems = eval_problems
        self.tokenizer = tokenizer
        self.eval_every_n_epochs = eval_every_n_epochs
        self.max_new_tokens = max_new_tokens
        self.steps_per_epoch = steps_per_epoch
        self.eval_batch_size = 8

    def _run_eval(self, model, step, device):
        from rlvr_grokking.rewards.deepscaler_reward import compute_score

        model.eval()
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        correct, total = 0, len(self.eval_problems)
        eval_start = time.time()
        all_prompts = [p["prompt_text"] for p in self.eval_problems]

        for batch_start in range(0, total, self.eval_batch_size):
            batch_prompts = all_prompts[batch_start:batch_start + self.eval_batch_size]
            batch_problems = self.eval_problems[batch_start:batch_start + self.eval_batch_size]

            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

            for j, problem in enumerate(batch_problems):
                prompt_len = (inputs["attention_mask"][j] == 1).sum().item()
                response = self.tokenizer.decode(outputs[j][prompt_len:], skip_special_tokens=True)
                score = compute_score("math", response, problem["ground_truth"])
                if score > 0:
                    correct += 1
                idx = batch_start + j
                if idx < 3:
                    print(f"  [Eval {idx}] GT={problem['ground_truth']} | Extracted={extract_boxed_answer(response)} | Score={score}")

        self.tokenizer.padding_side = original_padding_side
        accuracy = correct / total * 100 if total > 0 else 0
        eval_time = time.time() - eval_start
        print(f"  MATH500 Eval (step {step}): {correct}/{total} = {accuracy:.1f}% ({eval_time:.1f}s)")
        model.train()
        return {"eval/math500_accuracy": accuracy, "eval/math500_correct": correct, "eval/math500_total": total}

    def _log_to_wandb(self, results, step):
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(results, step=step)
        except ImportError:
            pass

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero and self.eval_problems:
            print("\n--- Baseline Eval (step 0) ---")
            results = self._run_eval(model, step=0, device=model.device)
            if state.is_world_process_zero:
                self._log_to_wandb(results, step=0)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        if self.eval_every_n_epochs > 0 and epoch % self.eval_every_n_epochs == 0:
            if state.is_world_process_zero and self.eval_problems:
                print(f"\n--- Eval (step {epoch}) ---")
                results = self._run_eval(model, step=epoch, device=model.device)
                if state.is_world_process_zero:
                    self._log_to_wandb(results, step=epoch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # Load and tokenize dataset
    dataset = load_dataset("parquet", data_files=args.dataset, split="train")
    print(f"Dataset: {len(dataset)} examples from {args.dataset}")
    dataset = tokenize_dataset(dataset, tokenizer, args.max_seq_length)

    # Calculate steps per epoch for save/eval scheduling
    n_devices = int(os.environ.get("WORLD_SIZE", 1))
    samples_per_device = len(dataset) // n_devices
    steps_per_epoch = max(1, samples_per_device // (args.batch_size * args.gradient_accumulation_steps))

    # Training arguments — num_steps epochs, each epoch = 1 full dataset pass
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10 * steps_per_epoch,  # log every ~10 epochs
        save_strategy="epoch",
        save_steps=args.save_every_n_steps,  # not used with save_strategy=epoch
        save_total_limit=12,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.wandb_run_name,
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # Set wandb project via env
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # Load eval problems
    eval_problems = None
    if args.eval_every_n_steps > 0 and os.path.exists(args.eval_dataset):
        eval_problems = load_math500_problems(args.eval_dataset, args.eval_samples)
        print(f"Loaded {len(eval_problems)} eval problems")

    callbacks = []
    if eval_problems:
        callbacks.append(Math500EvalCallback(
            eval_problems=eval_problems,
            tokenizer=tokenizer,
            eval_every_n_epochs=args.eval_every_n_steps,
            max_new_tokens=args.eval_max_new_tokens,
            steps_per_epoch=steps_per_epoch,
        ))

    # Custom save callback — only save every N epochs
    class SaveEveryNEpochsCallback(TrainerCallback):
        def __init__(self, save_every):
            self.save_every = save_every

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(state.epoch)
            if epoch % self.save_every == 0:
                control.should_save = True
            else:
                control.should_save = False

    callbacks.append(SaveEveryNEpochsCallback(args.save_every_n_steps))

    trainer = GEMTrainer(
        gem_beta=args.gem_beta,
        loss_type=args.loss,
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    print(f"Loss: {args.loss}" + (f" (beta={args.gem_beta})" if args.loss == "gem" else ""))
    print(f"Model: {args.model}")
    print(f"Devices: {n_devices}")
    print(f"Batch: {args.batch_size} x {args.gradient_accumulation_steps} x {n_devices} = {args.batch_size * args.gradient_accumulation_steps * n_devices} effective")
    print(f"Epochs: {args.num_steps}, steps_per_epoch: {steps_per_epoch}")
    print(f"Save every {args.save_every_n_steps} epochs, eval every {args.eval_every_n_steps} epochs")

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print("Training complete!")


if __name__ == "__main__":
    main()
