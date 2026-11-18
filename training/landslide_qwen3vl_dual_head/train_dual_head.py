#!/usr/bin/env python3
"""Train Qwen3-VL with its native LM head and an 8-way task head.

The organization follows CLSGen: the causal LM head learns the assistant
description while a two-layer MLP reads the final input-prefix hidden state.
Both losses update the same LoRA parameters. Canonical labels exactly match
the original descriptions, notably ``Earthflow`` and ``Rock slide``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_linear_schedule_with_warmup

from labels import LABELS, parse_record
from modeling_dual_head import ClassificationHead, get_hidden_size


DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct/Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_DATA_ROOT = "/root/autodl-tmp/loradataset"
DEFAULT_OUTPUT = "/root/autodl-tmp/landslide_qwen3vl_dual_head"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one durable, timestamped metrics event."""
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


class JsonlDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, media_dir: str | Path):
        self.jsonl_path = Path(jsonl_path)
        self.media_dir = Path(media_dir)
        self.records: list[dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.jsonl_path}:{line_number}") from exc

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = parse_record(self.records[index])
        item["record_index"] = index
        item["source"] = str(self.jsonl_path)
        if item["label"] is not None and item["answer_label"] is not None and item["label"] != item["answer_label"]:
            raise ValueError(
                f"Filename/answer label mismatch at {self.jsonl_path}:{index + 1}: "
                f"{item['label']} != {item['answer_label']}"
            )
        return item


class TextOnlyDataset(JsonlDataset):
    """Optional general instruction corpus used only for the LM loss."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        if item["image_paths"]:
            raise ValueError(
                "--text-jsonl must contain text-only records; found images at "
                f"{item['source']}:{item['record_index'] + 1}"
            )
        item["label"] = None
        item["label_id"] = None
        item["answer_label"] = None
        return item


def load_image(media_dir: Path, relative_path: str) -> Image.Image:
    image_path = media_dir / relative_path
    if not image_path.exists():
        fallback = Path("/root/autodl-tmp") / relative_path
        if fallback.exists():
            image_path = fallback
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as image:
        return image.convert("RGB")


def make_chat_inputs(processor: Any, item: dict[str, Any], media_dir: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    prompt_messages = copy.deepcopy(item["prompt_messages"])
    answer = item["answer"]
    if not prompt_messages:
        raise ValueError("Record has no user prompt")
    if not answer:
        raise ValueError("Record has no assistant answer")
    full_messages = prompt_messages + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    images = [load_image(media_dir, path) for path in item["image_paths"]]
    prompt_kwargs: dict[str, Any] = {"text": prompt_text, "return_tensors": "pt"}
    full_kwargs: dict[str, Any] = {"text": full_text, "return_tensors": "pt"}
    if images:
        prompt_kwargs["images"] = images
        full_kwargs["images"] = images
    prompt_inputs = processor(**prompt_kwargs)
    full_inputs = processor(**full_kwargs)
    input_ids = full_inputs["input_ids"]
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    if prompt_len > input_ids.shape[-1]:
        raise ValueError(f"Prompt is longer than full sequence: {prompt_len} > {input_ids.shape[-1]}")
    lm_labels = input_ids.clone()
    lm_labels[:, :prompt_len] = -100
    if "attention_mask" in full_inputs:
        lm_labels = lm_labels.masked_fill(full_inputs["attention_mask"] == 0, -100)
    return dict(full_inputs), lm_labels, prompt_len


def move_tensors(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


class DualHeadModel(nn.Module):
    def __init__(self, base_model: Any, classification_head: ClassificationHead):
        super().__init__()
        self.base_model = base_model
        self.classification_head = classification_head

    def forward(self, model_inputs: dict[str, Any], lm_labels: torch.Tensor, prompt_len: int) -> tuple[Any, torch.Tensor]:
        outputs = self.base_model(
            **model_inputs, labels=lm_labels, output_hidden_states=True,
            return_dict=True, use_cache=False,
        )
        # This position is before the target; causal attention prevents leakage.
        class_hidden = outputs.hidden_states[-1][:, prompt_len - 1, :]
        return outputs, self.classification_head(class_hidden)


def build_model(args: argparse.Namespace, device: torch.device) -> DualHeadModel:
    dtype = torch.float32 if device.type != "cuda" else (torch.bfloat16 if args.bf16 else torch.float16)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    )
    base_model.config.use_cache = False
    head = ClassificationHead(
        get_hidden_size(base_model), len(LABELS), args.head_hidden_size, args.head_dropout
    )
    if args.resume_from_checkpoint:
        checkpoint_dir = Path(args.resume_from_checkpoint)
        metadata = json.loads(
            (checkpoint_dir / "dual_head_config.json").read_text(encoding="utf-8")
        )
        if metadata.get("labels") != LABELS:
            raise ValueError(
                f"Checkpoint labels do not match current labels: {metadata.get('labels')}"
            )
        base_model = PeftModel.from_pretrained(
            base_model, checkpoint_dir / "adapter", is_trainable=True
        )
        head.load_state_dict(
            torch.load(
                checkpoint_dir / "classification_head.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        base_model.print_trainable_parameters()
    elif args.use_lora:
        base_model = get_peft_model(
            base_model,
            LoraConfig(
                r=args.lora_rank, lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules.split(","),
                task_type=TaskType.CAUSAL_LM, bias="none",
            ),
        )
        base_model.print_trainable_parameters()
    elif not args.allow_full_finetune:
        raise ValueError("Pass --allow-full-finetune explicitly with --no-lora")
    model = DualHeadModel(base_model, head)
    model.to(device)
    return model


def trainable_parameters(model: DualHeadModel) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def restore_training_state(
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    loader_length: int,
) -> tuple[int, int, float, float]:
    """Restore optimizer/RNG state from a completed-epoch LoRA checkpoint."""
    trainer_state = json.loads(
        (checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8")
    )
    if int(trainer_state["micro_step_in_epoch"]) != loader_length:
        raise ValueError(
            "Continuation currently requires an end-of-epoch checkpoint; got "
            f"micro_step_in_epoch={trainer_state['micro_step_in_epoch']} and "
            f"loader_length={loader_length}."
        )
    optimizer.load_state_dict(
        torch.load(
            checkpoint_dir / "optimizer.pt",
            map_location=device,
            weights_only=False,
        )
    )
    scaler_path = checkpoint_dir / "scaler.pt"
    if scaler.is_enabled() and scaler_path.exists():
        scaler.load_state_dict(
            torch.load(scaler_path, map_location=device, weights_only=False)
        )
    rng_state = torch.load(
        checkpoint_dir / "rng_state.pth", map_location="cpu", weights_only=False
    )
    random.setstate(rng_state["python"])
    torch.set_rng_state(rng_state["torch"])
    if device.type == "cuda" and "cuda" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    metrics = trainer_state.get("metrics", {})
    return (
        int(trainer_state["epoch"]),
        int(trainer_state["global_step"]),
        float(metrics.get("accuracy", -math.inf)),
        float(metrics.get("macro_f1", -math.inf)),
    )


def effective_number_weights(class_counts: list[int], beta: float, device: torch.device) -> torch.Tensor:
    """Class-Balanced Loss weights from Cui et al., CVPR 2019."""
    if not 0.0 <= beta < 1.0:
        raise ValueError("--class-balance-beta must be in [0, 1)")
    weights = []
    for count in class_counts:
        if count <= 0:
            weights.append(0.0)
        elif beta == 0.0:
            weights.append(1.0)
        else:
            weights.append((1.0 - beta) / (1.0 - beta**count))
    positive = sum(weight > 0 for weight in weights)
    scale = positive / sum(weights) if positive and sum(weights) else 1.0
    return torch.tensor([weight * scale for weight in weights], dtype=torch.float32, device=device)


def metrics_from_confusion(confusion: list[list[int]]) -> dict[str, Any]:
    per_class = []
    recalls, f1_values = [], []
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(LABELS)))
    for class_id, label in enumerate(LABELS):
        tp = confusion[class_id][class_id]
        support = sum(confusion[class_id])
        predicted = sum(row[class_id] for row in confusion)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "label": label, "support": support, "predicted": predicted,
            "correct": tp, "precision": precision, "recall": recall, "f1": f1,
        })
        if support:
            recalls.append(recall)
            f1_values.append(f1)
    return {
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "correct": correct, "total": total,
        "confusion_matrix": confusion, "per_class": per_class,
    }


@torch.no_grad()
def evaluate(model: DualHeadModel, processor: Any, dataset: Dataset, media_dir: Path, device: torch.device, max_samples: int | None = None) -> dict[str, Any]:
    model.eval()
    confusion = [[0] * len(LABELS) for _ in LABELS]
    lm_loss_total, evaluated = 0.0, 0
    limit = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for index in range(limit):
        item = dataset[index]
        if item["label_id"] is None:
            continue
        model_inputs, lm_labels, prompt_len = make_chat_inputs(processor, item, media_dir)
        model_inputs = move_tensors(model_inputs, device)
        outputs, logits = model(model_inputs, lm_labels.to(device), prompt_len)
        predicted, actual = int(logits.argmax(-1).item()), int(item["label_id"])
        confusion[actual][predicted] += 1
        lm_loss_total += float(outputs.loss.detach().cpu())
        evaluated += 1
    metrics = metrics_from_confusion(confusion)
    metrics["lm_loss"] = lm_loss_total / evaluated if evaluated else 0.0
    model.train()
    return metrics


def save_checkpoint(
    model: DualHeadModel,
    args: argparse.Namespace,
    output_dir: Path,
    name: str,
    metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    micro_step_in_epoch: int,
    global_step: int,
    total_updates: int,
) -> Path:
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(model.base_model, PeftModel):
        raise RuntimeError(
            "Checkpoint saving is restricted to PEFT/LoRA runs so the frozen "
            "Qwen base-model parameters are never written."
        )
    # PEFT writes only adapter tensors, never the frozen Qwen base weights.
    model.base_model.save_pretrained(checkpoint_dir / "adapter")
    torch.save(model.classification_head.state_dict(), checkpoint_dir / "classification_head.pt")
    # Optimizer groups contain only trainable LoRA and classification-head parameters.
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    if scaler.is_enabled():
        torch.save(scaler.state_dict(), checkpoint_dir / "scaler.pt")
    rng_state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng_state, checkpoint_dir / "rng_state.pth")
    trainer_state = {
        "epoch": epoch,
        "micro_step_in_epoch": micro_step_in_epoch,
        "global_step": global_step,
        "total_updates": total_updates,
        "metrics": metrics,
    }
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(trainer_state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "base_model": args.model, "labels": LABELS, "num_labels": len(LABELS),
        "classification_feature": "last_hidden_state at prompt_len - 1",
        "classification_head": {
            "type": "LayerNorm-Linear-GELU-Dropout-Linear",
            "input_size": model.classification_head.input_size,
            "hidden_size": model.classification_head.hidden_size,
            "dropout": model.classification_head.dropout_probability,
        },
        "head_initialization": "PyTorch random initialization",
        "use_lora": args.use_lora,
        "class_balanced_loss": args.class_balanced_loss,
        "class_balance_beta": args.class_balance_beta,
        "checkpoint": {
            "epoch": epoch,
            "micro_step_in_epoch": micro_step_in_epoch,
            "global_step": global_step,
            "total_updates": total_updates,
            "contains_base_model_weights": False,
            "contains_optimizer_state": True,
        },
        "metrics": metrics,
    }
    (checkpoint_dir / "dual_head_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return checkpoint_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL LM + 8-way head")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--val-jsonl", default=None)
    parser.add_argument("--text-jsonl", default=None)
    parser.add_argument("--max-text-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="End-of-epoch dual-head checkpoint; --epochs means additional epochs",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lm-loss-weight", type=float, default=1.0)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--eval-save-steps", type=int, default=150,
        help="Validate and save every N optimizer updates; the final update is always saved",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-hidden-size", type=int, default=2048)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--class-balance-beta", type=float, default=0.999)
    parser.add_argument("--no-class-balanced-loss", dest="class_balanced_loss", action="store_false")
    parser.set_defaults(class_balanced_loss=True)
    parser.add_argument("--allow-full-finetune", action="store_true")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.set_defaults(use_lora=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj", help="Language attention only; visual tower remains frozen")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", dest="bf16", action="store_true")
    precision.add_argument("--fp16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume_from_checkpoint and not args.use_lora:
        raise ValueError("--resume-from-checkpoint requires LoRA")
    if args.batch_size != 1:
        raise ValueError("Use --batch-size 1 and gradient accumulation for variable-length image sequences")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.eval_save_steps < 1:
        raise ValueError("--eval-save-steps must be positive")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    seed_everything(args.seed)
    data_root = Path(args.data_root)
    train_jsonl = Path(args.train_jsonl or data_root / "qwen3vl_landslide_train.jsonl")
    val_jsonl = Path(args.val_jsonl or data_root / "qwen3vl_landslide_val.jsonl")
    media_dir = data_root
    image_dataset: Dataset = JsonlDataset(train_jsonl, media_dir)
    train_dataset: Dataset = image_dataset
    if args.text_jsonl:
        text_dataset: Dataset = TextOnlyDataset(args.text_jsonl, media_dir)
        if args.max_text_samples is not None:
            text_dataset = torch.utils.data.Subset(text_dataset, range(min(args.max_text_samples, len(text_dataset))))
        train_dataset = ConcatDataset([image_dataset, text_dataset])
    val_dataset = JsonlDataset(val_jsonl, media_dir)
    if args.max_train_samples is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))

    class_counts = [0] * len(LABELS)
    for index in range(len(train_dataset)):
        item = train_dataset[index]
        if item["label_id"] is not None:
            class_counts[item["label_id"]] += 1
    print("Training class counts:")
    for label, count in zip(LABELS, class_counts):
        print(f"  {label}: {count}")
    print(f"Training records: {len(train_dataset)}; validation records: {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 is default; pass --fp16 on unsupported hardware")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = build_model(args, device)
    if args.gradient_checkpointing:
        model.base_model.gradient_checkpointing_enable()
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "classification_head" not in n]
    head_params = list(model.classification_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.learning_rate},
        {"params": head_params, "lr": args.head_learning_rate},
    ], weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.bf16)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = output_dir / "training_metrics.jsonl"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda batch: batch[0])
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    start_epoch, start_global_step = 0, 0
    best_accuracy, best_macro_f1 = -math.inf, -math.inf
    if args.resume_from_checkpoint:
        checkpoint_dir = Path(args.resume_from_checkpoint)
        start_epoch, start_global_step, best_accuracy, best_macro_f1 = restore_training_state(
            checkpoint_dir, optimizer, scaler, device, len(loader)
        )
        # Keep AdamW moments but deliberately restart a lower-LR continuation schedule.
        optimizer.param_groups[0]["lr"] = args.learning_rate
        optimizer.param_groups[0]["initial_lr"] = args.learning_rate
        optimizer.param_groups[1]["lr"] = args.head_learning_rate
        optimizer.param_groups[1]["initial_lr"] = args.head_learning_rate
    stage_updates = updates_per_epoch * args.epochs
    total_updates = start_global_step + stage_updates
    warmup_steps = int(stage_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, max(stage_updates, 1)
    )
    class_weights = effective_number_weights(class_counts, args.class_balance_beta, device) if args.class_balanced_loss else None
    if class_weights is not None:
        print("Class-balanced CE weights:")
        for label, weight in zip(LABELS, class_weights.tolist()):
            print(f"  {label}: {weight:.4f}")
    print(
        f"Continuation updates: {stage_updates}; global endpoint: {total_updates}; "
        f"warmup updates: {warmup_steps}"
    )
    append_jsonl(metrics_jsonl, {
        "event": "run_start",
        "run_id": run_id,
        "training_records": len(train_dataset),
        "validation_records": len(val_dataset),
        "labels": LABELS,
        "class_counts": dict(zip(LABELS, class_counts)),
        "total_updates": total_updates,
        "stage_updates": stage_updates,
        "start_epoch": start_epoch,
        "start_global_step": start_global_step,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "warmup_updates": warmup_steps,
        "args": vars(args),
    })

    global_step = start_global_step
    last_eval_step = -1
    accumulation = args.gradient_accumulation_steps
    remainder = len(loader) % accumulation

    def validate_and_save(epoch: int, micro_step_in_epoch: int) -> None:
        nonlocal best_accuracy, best_macro_f1, last_eval_step
        metrics = evaluate(
            model, processor, val_dataset, media_dir, device, args.max_val_samples
        )
        print(
            f"update={global_step}/{total_updates} epoch={epoch} validation "
            f"accuracy={metrics['correct']}/{metrics['total']}="
            f"{metrics['accuracy'] * 100:.2f}% macro_f1={metrics['macro_f1']:.4f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
            f"lm_loss={metrics['lm_loss']:.4f}"
        )
        print("Confusion matrix rows=actual, columns=predicted:")
        print(json.dumps(metrics["confusion_matrix"], ensure_ascii=False))
        checkpoint_name = f"step-{global_step}"
        improved_accuracy = metrics["accuracy"] > best_accuracy
        improved_macro_f1 = metrics["macro_f1"] > best_macro_f1
        save_kwargs = dict(
            model=model,
            args=args,
            output_dir=output_dir,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            micro_step_in_epoch=micro_step_in_epoch,
            global_step=global_step,
            total_updates=total_updates,
        )
        save_checkpoint(name=checkpoint_name, **save_kwargs)
        if improved_accuracy:
            best_accuracy = metrics["accuracy"]
            save_checkpoint(name="best", **save_kwargs)
        if improved_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            save_checkpoint(name="best_macro_f1", **save_kwargs)
        append_jsonl(metrics_jsonl, {
            "event": "validation",
            "run_id": run_id,
            "epoch": epoch,
            "micro_step_in_epoch": micro_step_in_epoch,
            "global_step": global_step,
            "total_updates": total_updates,
            "checkpoint": str(output_dir / checkpoint_name),
            "is_best_accuracy": improved_accuracy,
            "is_best_macro_f1": improved_macro_f1,
            **metrics,
        })
        last_eval_step = global_step

    final_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch + 1, final_epoch + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_lm, running_cls = 0.0, 0.0
        update_lm, update_cls, update_joint, update_micro_count = 0.0, 0.0, 0.0, 0
        for step, item in enumerate(loader, start=1):
            model_inputs, lm_labels, prompt_len = make_chat_inputs(processor, item, media_dir)
            model_inputs = move_tensors(model_inputs, device)
            lm_labels = lm_labels.to(device)
            divisor = remainder if remainder and step > len(loader) - remainder else accumulation
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                outputs, class_logits = model(model_inputs, lm_labels, prompt_len)
                lm_loss = outputs.loss
                if item["label_id"] is None:
                    cls_loss = torch.zeros((), device=device)
                else:
                    target = torch.tensor([item["label_id"]], device=device)
                    # With micro-batch size 1, CrossEntropy(weight=..., mean)
                    # divides by the target weight and cancels reweighting.
                    # Apply the normalized effective-number weight explicitly.
                    cls_loss = F.cross_entropy(
                        class_logits.float(), target, reduction="none"
                    )
                    if class_weights is not None:
                        cls_loss = cls_loss * class_weights.index_select(0, target)
                    cls_loss = cls_loss.mean()
                joint_loss = args.lm_loss_weight * lm_loss + args.cls_loss_weight * cls_loss
                loss = joint_loss / divisor
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_lm += float(lm_loss.detach().cpu())
            running_cls += float(cls_loss.detach().cpu())
            update_lm += float(lm_loss.detach().cpu())
            update_cls += float(cls_loss.detach().cpu())
            update_joint += float(joint_loss.detach().cpu())
            update_micro_count += 1
            if step % accumulation == 0 or step == len(loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
                    .detach().cpu()
                )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                append_jsonl(metrics_jsonl, {
                    "event": "train_update",
                    "run_id": run_id,
                    "epoch": epoch,
                    "micro_step_in_epoch": step,
                    "global_step": global_step,
                    "total_updates": total_updates,
                    "micro_batches": update_micro_count,
                    "lm_loss": update_lm / update_micro_count,
                    "classification_loss": update_cls / update_micro_count,
                    "joint_loss": update_joint / update_micro_count,
                    "gradient_norm_before_clipping": gradient_norm,
                    "lora_learning_rate": scheduler.get_last_lr()[0],
                    "classification_head_learning_rate": scheduler.get_last_lr()[1],
                })
                update_lm, update_cls, update_joint, update_micro_count = 0.0, 0.0, 0.0, 0
                if global_step % args.eval_save_steps == 0:
                    validate_and_save(epoch, step)
            if step % 50 == 0 or step == len(loader):
                print(
                    f"epoch={epoch} step={step}/{len(loader)} update={global_step}/{total_updates} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} lm_loss={running_lm / step:.4f} "
                    f"cls_loss={running_cls / step:.4f}"
                )

    # The last update is validated and saved even when it is not divisible by the interval.
    if global_step != last_eval_step:
        validate_and_save(final_epoch, len(loader))
    append_jsonl(metrics_jsonl, {
        "event": "run_end",
        "run_id": run_id,
        "global_step": global_step,
        "total_updates": total_updates,
        "best_validation_accuracy": best_accuracy,
        "best_validation_macro_f1": best_macro_f1,
    })
    print(f"Training complete. Best validation accuracy: {best_accuracy * 100:.2f}%; best macro-F1: {best_macro_f1:.4f}")
    print(f"Metrics JSONL: {metrics_jsonl}")
    print(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
