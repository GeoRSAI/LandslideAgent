#!/usr/bin/env python3
"""Run classification and text generation from a dual-head checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from labels import LABELS, label_from_answer, parse_record
from modeling_dual_head import ClassificationHead, get_hidden_size


DEFAULT_MODEL = os.getenv("LLM_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
DEFAULT_DATA_ROOT = os.getenv("DUAL_HEAD_DATA_ROOT", "data")


def load_image(media_dir: Path, relative_path: str) -> Image.Image:
    image_path = media_dir / relative_path
    if not image_path.exists():
        fallback = Path(relative_path)
        if fallback.exists():
            image_path = fallback
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        return image.convert("RGB")


def build_prompt_inputs(
    processor: Any,
    item: dict[str, Any],
    media_dir: Path,
) -> tuple[dict[str, Any], int]:
    prompt_messages = copy.deepcopy(item["prompt_messages"])
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images = [load_image(media_dir, path) for path in item["image_paths"]]
    kwargs: dict[str, Any] = {"text": prompt_text, "return_tensors": "pt"}
    if images:
        kwargs["images"] = images
    inputs = processor(**kwargs)
    return dict(inputs), int(inputs["input_ids"].shape[-1])


def load_dual_head(
    args: argparse.Namespace,
    processor: Any,
    device: torch.device,
) -> tuple[Any, ClassificationHead]:
    checkpoint = Path(args.checkpoint)
    metadata_path = checkpoint / "dual_head_config.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing checkpoint metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("labels") != LABELS:
        raise ValueError(
            f"Checkpoint labels {metadata.get('labels')} do not match runtime labels {LABELS}"
        )
    head_config = metadata.get("classification_head", {})
    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    adapter_dir = checkpoint / "adapter"
    if adapter_dir.exists():
        base = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    elif (checkpoint / "base_model").exists():
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint / "base_model",
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        raise FileNotFoundError(
            f"No adapter/ or base_model/ under checkpoint: {args.checkpoint}"
        )

    head = ClassificationHead(
        input_size=get_hidden_size(base),
        num_labels=len(LABELS),
        hidden_size=int(head_config.get("hidden_size", 2048)),
        dropout=float(head_config.get("dropout", 0.1)),
    )
    head.load_state_dict(
        torch.load(
            checkpoint / "classification_head.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    base.to(device).eval()
    head.to(device).eval()
    return base, head


@torch.no_grad()
def run_one(
    base: Any,
    head: ClassificationHead,
    processor: Any,
    item: dict[str, Any],
    media_dir: Path,
    device: torch.device,
    max_new_tokens: int,
    classification_only: bool = False,
) -> dict[str, Any]:
    prompt_inputs, input_len = build_prompt_inputs(processor, item, media_dir)
    prompt_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in prompt_inputs.items()
    }
    outputs = base(
        **prompt_inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden = outputs.hidden_states[-1][:, -1, :].float()
    logits = head(hidden)
    probabilities = torch.softmax(logits.float(), dim=-1)
    predicted_id = int(logits.argmax(dim=-1).item())

    result = {
        "image": item["image_paths"][0] if item["image_paths"] else None,
        "actual": item["label"],
        "predicted": LABELS[predicted_id],
        "class_probabilities": {
            label: float(probability)
            for label, probability in zip(LABELS, probabilities[0].cpu())
        },
        "class_logits": {
            label: float(logit)
            for label, logit in zip(LABELS, logits[0].float().cpu())
        },
    }
    if classification_only:
        return result
    generated_ids = base.generate(
        **prompt_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_text = processor.tokenizer.decode(
        generated_ids[0][input_len:],
        skip_special_tokens=True,
    ).strip()
    generated_label = label_from_answer(generated_text)
    result.update({
        "generated_label": generated_label,
        "head_text_agreement": (
            generated_label == LABELS[predicted_id]
            if generated_label is not None
            else None
        ),
        "generated_text": generated_text,
    })
    return result


def classification_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = [[0] * len(LABELS) for _ in LABELS]
    for result in results:
        if result["actual"] not in LABELS:
            continue
        actual = LABELS.index(result["actual"])
        predicted = LABELS.index(result["predicted"])
        confusion[actual][predicted] += 1
    per_class = []
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(LABELS)))
    for index, label in enumerate(LABELS):
        tp = confusion[index][index]
        support = sum(confusion[index])
        predicted_count = sum(row[index] for row in confusion)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "label": label,
            "support": support,
            "predicted": predicted_count,
            "correct": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    supported = [item for item in per_class if item["support"]]
    normalized = [
        [value / sum(row) if sum(row) else 0.0 for value in row]
        for row in confusion
    ]
    return {
        "labels": LABELS,
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": sum(item["recall"] for item in supported) / len(supported) if supported else 0.0,
        "macro_f1": sum(item["f1"] for item in supported) / len(supported) if supported else 0.0,
        "correct": correct,
        "total": total,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "row_normalized_confusion_matrix": normalized,
    }


def generation_comparison_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [result for result in results if result.get("actual") in LABELS]
    parsed = [result for result in labeled if result.get("generated_label") in LABELS]
    generated_as_predictions = [
        {**result, "predicted": result["generated_label"]} for result in parsed
    ]
    text_correct = sum(
        result["generated_label"] == result["actual"] for result in parsed
    )
    agreements = sum(result.get("head_text_agreement") is True for result in labeled)
    return {
        "classification_head": classification_metrics(labeled),
        "generated_text": {
            "parsed": len(parsed),
            "parse_failures": len(labeled) - len(parsed),
            "parse_rate": len(parsed) / len(labeled) if labeled else 0.0,
            "correct": text_correct,
            "total": len(labeled),
            "accuracy_over_all_samples": text_correct / len(labeled) if labeled else 0.0,
            "metrics_on_parsed_samples": classification_metrics(generated_as_predictions),
        },
        "head_text_agreement": {
            "agree": agreements,
            "total": len(labeled),
            "agreement_rate_over_all_samples": agreements / len(labeled) if labeled else 0.0,
            "agreement_rate_among_parsed_samples": agreements / len(parsed) if parsed else 0.0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--classification-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.jsonl and not args.image:
        raise ValueError("Provide --jsonl or --image")
    if args.image and not args.prompt:
        raise ValueError("--prompt is required with --image")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    base, head = load_dual_head(args, processor, device)
    media_dir = Path(args.data_root)

    items: list[dict[str, Any]] = []
    if args.jsonl:
        with Path(args.jsonl).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    items.append(parse_record(json.loads(line)))
    else:
        items.append(
            {
                "prompt_messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": args.image},
                            {"type": "text", "text": args.prompt},
                        ],
                    }
                ],
                "answer": "",
                "image_paths": [args.image],
                "label": None,
                "label_id": None,
                "answer_label": None,
            }
        )

    if args.max_samples is not None:
        items = items[: args.max_samples]
    results = []
    for index, item in enumerate(items):
        result = run_one(
            base,
            head,
            processor,
            item,
            media_dir,
            device,
            args.max_new_tokens,
            args.classification_only,
        )
        result["id"] = index
        results.append(result)
        if args.classification_only or args.summary_only:
            if (index + 1) % 25 == 0 or index + 1 == len(items):
                print(f"Processed {index + 1}/{len(items)}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Saved: {output_path}")

    if args.jsonl:
        metrics = (
            classification_metrics(results)
            if args.classification_only
            else generation_comparison_metrics(results)
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        if args.metrics_output:
            metrics_path = Path(args.metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
