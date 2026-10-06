# Landslide Agent

A tool-driven framework for landslide analysis in satellite and aerial imagery, combining multimodal reasoning, segmentation, classification, geographic evidence, and structured reports.

This release includes framework code and the web interface. Model weights, trained adapter weights, datasets, private imagery and experiment results are not included.

## Dataset

The previously published dataset remains available through the [original dataset download](https://drive.google.com/file/d/1wibzr3qJ4LTCzQzh_jSfEXs48Zla4Nwd/view?usp=sharing). Dataset files are distributed separately from this code release.

## Architecture

Image metadata -> visual assessment -> segmentation -> refinement / conditional review -> classification and geographic context -> fusion -> report.

The shared controller lives in `src/agent/controller.py`, the tool-calling loop in `src/orchestration/`, and the alternative LangGraph workflow in `src/graph/landslide_graph.py`. Service endpoints include `/v1/agent/analyze`, `/v1/graph/analyze`, and `/health`. See [methods](docs/METHODS.md).

See the [deployment and usage guide](docs/USAGE.md) for model provisioning, OpenMMLab setup, service startup and configuration.

## Installation

Use Python 3.10+. Linux or WSL is recommended for GPU services and shell scripts. From the checkout root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
bash scripts/start_frontend_all.sh
```

Open http://127.0.0.1:8003/ and inspect `/health`. The example enables `LLM_MOCK=1` for a service demonstration. Mock mode replaces language-model responses; real segmentation and classification still require their own models.

Windows PowerShell can run the mock frontend directly after installing dependencies:

```powershell
$env:LLM_MOCK = "1"
python -m uvicorn scripts.llm_service:app --host 127.0.0.1 --port 8003
```

The Bash launcher loads `.env`. Direct Uvicorn invocation uses process environment variables. Run from the checkout root and retain the source checkout for configuration and web assets.

## Model configuration

| Component | Environment variables |
| --- | --- |
| Multimodal model | `LLM_MOCK=0`, `LLM_MODEL_PATH`, optionally `LLM_LORA_PATH` |
| Dual-head adapter | `LLM_DUAL_HEAD_PATH`, `CLS_BACKEND=dualhead` |
| Segmentation | `SEG_ENV_PYTHON`, `MMSEG_CONFIG_PATH`, `MMSEG_CHECKPOINT_PATH`, `MMSEG_DEVICE` |
| MMPreTrain classifier | `CLS_BACKEND=convnext`, `CLS_ENV_PYTHON`, `MMPRETRAIN_ROOT`, `CLS_CONFIG_PATH`, `CLS_CHECKPOINT_PATH` |
| Elevation provider | `GEO_DEM_PROVIDER`, optionally `OPENTOPOGRAPHY_API_KEY` |
| Accessible images | `IMAGE_ALLOWED_ROOT` (defaults to checkout root) |

Defaults under `models/` are placeholders. Provision PyTorch, Transformers, PEFT and model-specific utilities in the LLM environment; provision MMsegmentation / MMPreTrain and compatible OpenMMLab dependencies separately. Compatible GPU package versions depend on your checkpoints. Core installation does not install these model environments.

Geographic tools contact external geocoding, elevation and OpenStreetMap services. Network access and provider availability affect results. Keep credentials in your local environment. The service has no authentication layer; use the default loopback binding or an authenticated gateway.

## Development

```bash
python -m pytest -q
```

Tests isolate agent rules and workflows through fake or injected dependencies. They do not establish trained-model accuracy. Configure thresholds in `configs/thresholds.json` and environment overrides.

- `src/agent/`: shared policy and JSON-RPC protocol
- `src/orchestration/`: tool-calling agent loop
- `src/graph/`: LangGraph workflow
- `src/models/`, `src/pipelines/`, `src/tools/`: inference, analysis, and geographic tools
- `scripts/`: services and batch utilities
- `static/`: web interface
- `tests/`: regression tests

See [release preparation](docs/RELEASE_PREPARATION.md) and [contribution guidance](CONTRIBUTING.md).

## License

[MIT](LICENSE). Model weights, datasets and third-party components retain their own licenses.

## Dual-head inference sources

Dual-head scripts and configuration are included under `models/landslide_qwen3vl_dual_head_continuous_10ep_best/`. Trained adapter and classification-head weights are excluded. See [adapter usage](docs/DUAL_HEAD_USAGE.md).

First-pass visual assessment defaults to `LLM_FIRST_PASS_ADAPTER=dualhead`. Supply the dual-head weights for real inference, or explicitly set this variable to `base`. Second-pass visual review continues to default to the base model.
