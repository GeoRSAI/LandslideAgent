# Landslide Agent

An open-source multimodal landslide dataset and domain-rule augmented agent framework for remote sensing-based landslide identification, geospatial reasoning, and structured disaster report generation.

This repository provides a remote-sensing landslide analysis agent built around a FastAPI frontend, a JSON-RPC tool registry, and segmentation-guided multi-stage reasoning.

## Features

- FastAPI web frontend and OpenAI-style chat endpoint
- JSON-RPC 2.0 tool protocol for agent-tool interaction
- TIFF metadata reading, segmentation, candidate-region refinement, classification, geo-background query, decision fusion, and optional report writing
- OpenMMLab-compatible adapters for MMSegmentation and MMPreTrain
- Mock LLM mode for running the framework without local large-model weights

## Dataset

The open-source dataset is available on Google Drive: [Download dataset](https://drive.google.com/file/d/1wibzr3qJ4LTCzQzh_jSfEXs48Zla4Nwd/view?usp=sharing).

## Architecture

Main flow:

1. `tiff.info`: read image metadata
2. `llm.first_pass`: scene-level first-pass judgement
3. `seg.run`: semantic segmentation
4. `seg.refine`: segmentation-guided candidate-region refinement
5. `cls.run`: scene/image classification
6. `geo.background` / `geo.nearby`: optional geographic context
7. `fuse.decision`: final decision fusion
8. `report.write`: optional report JSON output

## Quick Start

Install core dependencies:

```bash
pip install -r requirements.txt
```

Run the protocol demo:

```bash
python -m scripts.run_protocol_demo --image data/sample.tif --out outputs/report.protocol.json
```

Start the FastAPI frontend in mock mode:

```bash
LLM_MOCK=1 bash scripts/start_frontend_all.sh
```

Then open `http://127.0.0.1:8003/`.

## Configuration

Copy `.env.example` and fill in paths for your own environment:

```bash
cp .env.example .env
```

Important variables:

- `LLM_MODEL_PATH`: local large-model path. Leave empty when `LLM_MOCK=1`.
- `LLM_LORA_PATH`: optional LoRA adapter path.
- `MMSEG_CONFIG_PATH` and `MMSEG_CHECKPOINT_PATH`: MMSegmentation config and checkpoint.
- `MMPRETRAIN_ROOT`, `CLS_CONFIG_PATH`, and `CLS_CHECKPOINT_PATH`: MMPreTrain source/config/checkpoint for classification.
- `OPENTOPOGRAPHY_API_KEY`: optional API key for terrain background queries.

Do not commit `.env`, private API keys, model weights, or checkpoints.

## External Models And Frameworks

The open-source dataset is provided through the Google Drive link above. This repository does not include large-model weights, OpenMMLab source trees, or trained checkpoints. Users should install and configure these external dependencies separately according to their own licenses:

- Large multimodal model, for example a local Qwen-VL compatible model
- `mmsegmentation` / `mmcv` / `mmengine` for segmentation
- `mmpretrain` for classification
- Segmentation and classification checkpoints trained or obtained by the user

## Using MMSegmentation

Set the segmentation environment variables before starting services:

```bash
export SEG_BACKEND=mmseg
export MMSEG_CONFIG_PATH=/path/to/mmseg_config.py
export MMSEG_CHECKPOINT_PATH=/path/to/mmseg_checkpoint.pth
export MMSEG_DEVICE=cuda:0
export MMSEG_LANDSLIDE_CLASS_INDEX=1
```

`MMSEG_LANDSLIDE_CLASS_INDEX` should match your dataset label mapping.

## Using MMPreTrain

Set the classification environment variables:

```bash
export CLS_ENV_PYTHON=/path/to/python
export MMPRETRAIN_ROOT=/path/to/mmpretrain
export CLS_CONFIG_PATH=/path/to/classification_config.py
export CLS_CHECKPOINT_PATH=/path/to/classification_checkpoint.pth
export CLS_DEVICE=cpu
# Optional class-id to class-name mapping file
# export CLS_CLASS_MAPPING_PATH=/path/to/class_mapping.txt
```

## Standard Agent Protocol

This project includes a minimal JSON-RPC 2.0 tool protocol:

- `tools/list`: list available tools and input schemas
- `tools/call`: call a tool with `{name, arguments}`

Protocol server and registry:

- `src/agent/protocol.py`
- `src/agent/default_server.py`

## Project Layout

- `configs/thresholds.json`: decision thresholds
- `data/sample.tif`: optional sample image for demo use
- `scripts/llm_service.py`: main FastAPI app, chat endpoint, and frontend entry
- `scripts/seg_service.py`: segmentation service
- `scripts/cls_service.py`: classification service
- `scripts/start_frontend_all.sh`: startup script
- `scripts/run_protocol_demo.py`: protocol-based CLI demo
- `src/agent/`: JSON-RPC protocol and tool registry
- `src/models/`: LLM, segmentation, and classification adapters
- `src/pipelines/`: pipeline stages and fusion logic
- `src/tools/`: TIFF, crop, OSM, and geo-background tools
- `static/index.html`: web UI

`logs/` and `outputs/` are runtime-generated directories and are intentionally ignored by git.

## Notes

- Use `LLM_MOCK=1` when model weights are unavailable.
- Runtime outputs are JSON-friendly dictionaries for downstream integration.
- This project is intended for research and demonstration. Validate model behavior, data rights, and deployment security before production use.
