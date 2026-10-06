# Deployment and usage

## 1. Install the framework

```bash
git clone https://github.com/GeoRSAI/LandslideAgent.git
cd LandslideAgent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Linux or WSL is recommended for the launch scripts. The core package installs the web service, workflow and image-processing dependencies. It does not deploy trained models or OpenMMLab environments.

## 2. Choose mock or real inference

For a frontend demonstration, retain `LLM_MOCK=1` and run `bash scripts/start_frontend_all.sh`. This skips actual LLM loading; it does not supply a real segmentation model or establish model accuracy.

For real inference, provision a compatible Qwen3-VL-8B-Instruct model environment with PyTorch, Transformers supporting Qwen3-VL, PEFT and the model's required image utilities. Set `LLM_ENV_PYTHON` to that environment's Python executable. The appropriate CUDA and package versions depend on your hardware and model setup.

The repository contains dual-head inference sources and configuration only. Supply these trained files yourself:

```text
models/landslide_qwen3vl_dual_head_continuous_10ep_best/
  adapter/adapter_model.safetensors
  classification_head.pt
```

The Qwen base-model weights must also be supplied separately. Do not commit local weights or credentials.

## 3. Deploy MMsegmentation and MMPreTrain yourself

Create compatible model environments using the installation instructions of [MMsegmentation](https://github.com/open-mmlab/mmsegmentation) and [MMPreTrain](https://github.com/open-mmlab/mmpretrain). Install their required PyTorch, MMEngine and MMCV dependencies according to the chosen versions. The repository's service scripts wrap those installations; they do not download or install them automatically.

Provide a segmentation config/checkpoint pair and, for ConvNeXt classification or the default second opinion, a classification config/checkpoint pair. The classifier also requires the MMPreTrain source root. Config files and checkpoints must match the trained architecture and label mapping.

Edit `.env` with paths for your machine:

```bash
LLM_MOCK=0
LLM_ENV_PYTHON=/path/to/llm-env/bin/python
LLM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
LLM_DUAL_HEAD_PATH=models/landslide_qwen3vl_dual_head_continuous_10ep_best
LLM_FIRST_PASS_ADAPTER=dualhead
LLM_DESCRIBE_ADAPTER=dualhead
LLM_VISUAL_EVIDENCE_ADAPTER=base
CLS_BACKEND=dualhead

SEG_ENV_PYTHON=/path/to/mmseg-env/bin/python
SEG_BACKEND=mmseg
MMSEG_CONFIG_PATH=/path/to/segmentation-config.py
MMSEG_CHECKPOINT_PATH=/path/to/segmentation-checkpoint.pth
MMSEG_DEVICE=cuda:0
MMSEG_LANDSLIDE_CLASS_INDEX=1

CLS_ENV_PYTHON=/path/to/mmpretrain-env/bin/python
MMPRETRAIN_ROOT=/path/to/mmpretrain
CLS_CONFIG_PATH=/path/to/classification-config.py
CLS_CHECKPOINT_PATH=/path/to/classification-checkpoint.pth
CLS_DEVICE=cuda:0
```

`CLS_SECOND_OPINION=1` is the default: dual-head classification also attempts a ConvNeXt opinion. Set it to `0` to use only the dual-head result when available. This does not eliminate the segmentation requirement. Set `CLS_BACKEND=convnext` to use the image classifier directly.

Without dual-head weights, set both `LLM_FIRST_PASS_ADAPTER=base` and `LLM_DESCRIBE_ADAPTER=base`, and configure ConvNeXt classification. `LLM_LORA_PATH` specifies a separate optional SFT adapter and is not a substitute for the dual-head directory. See [adapter routing](DUAL_HEAD_USAGE.md).

## 4. Start and check services

```bash
bash scripts/start_frontend_all.sh
```

In another terminal, wait for the model to finish loading:

```bash
curl http://127.0.0.1:8003/health
curl -X POST http://127.0.0.1:8003/admin/start_services
```

Check `model_status=ready` and `dual_head_loaded=true` for a real dual-head deployment. The admin endpoint starts the configured segmentation and classification subprocesses; it does not install dependencies or supply model weights. Inspect its response and service logs if startup fails.

Open `http://127.0.0.1:8003/`, upload an image, provide the requested location/radius information and start analysis. Missing geographic evidence or unavailable model services may cause a pause or a degraded report. Uploading through the UI puts images under the project directory; external images served through `/media` must be inside `IMAGE_ALLOWED_ROOT`.

The server has no authentication layer. Keep loopback binding or use an authenticated gateway. External elevation, geocoding and OpenStreetMap providers require network access; any optional API keys belong in your private `.env`.

Stop the project services with `bash scripts/stop_frontend_all.sh`. That script matches service command lines, so use it on a machine where these service names identify the intended project instances.
