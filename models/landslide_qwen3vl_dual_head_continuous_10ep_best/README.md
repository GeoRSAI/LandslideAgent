# Dual-head inference sources

This directory contains inference code, label definitions, architecture and adapter metadata only. No trained weights are distributed.

Supply `adapter/adapter_model.safetensors` and `classification_head.pt` yourself, plus the compatible Qwen3-VL-8B-Instruct base model. `LLM_DUAL_HEAD_PATH` must point to this directory. Install PyTorch, PEFT, Transformers with Qwen3-VL support and Pillow in the model environment.

The standalone script uses the dual-head LoRA for both classification and text generation. The framework selects adapters separately per request; see [adapter usage](../../docs/DUAL_HEAD_USAGE.md).

```bash
python models/landslide_qwen3vl_dual_head_continuous_10ep_best/infer_dual_head.py --checkpoint models/landslide_qwen3vl_dual_head_continuous_10ep_best --model /path/to/Qwen3-VL-8B-Instruct --image /path/to/image.png
```

The original checkpoint metadata records training/evaluation information, including optimizer-state metadata. This source-only directory contains no optimizer state or checkpoint tensors. Recorded metrics are historical metadata and have not been independently reproduced for this release.
