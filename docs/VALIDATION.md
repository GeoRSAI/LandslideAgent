# Local validation

Validated on Windows with Python 3.12 in a fresh virtual environment on 2026-10-06.

- Editable installation with development dependencies succeeded.
- `pip check`: no broken requirements.
- `python -m pytest -q`: 190 passed, 3 deprecation warnings (FastAPI startup hooks and LangGraph serialization defaults).
- Mock service startup: `/health`, `/`, and `/static/js/panels.js` returned HTTP 200; model status was ready in mock mode.
- Added regression coverage for configured image roots and rejection of sibling directories sharing a path prefix.
- Isolated unit tests from the live LLM server; tests can override the default stub with controlled responses.
- Embedded credential pattern scan found no matches in release text files. This is a pattern scan, not a comprehensive security audit.
- Source backup SHA-256 matched the container archive: `f7ca55630b8fcf2d6b8818d168eca2d022cd4ec582246e5b07df20fc10ad6e7f`.

Not validated: real model inference, GPU environments, provider availability, scientific accuracy, and Linux CI execution. The included GitHub Actions workflow will run isolated tests after publication.

First-pass adapter defaults to dualhead; second-pass remains base. Four request-payload regression cases verify first-pass, second-pass, scene description and report routing. Container source was updated with the same routing change.
