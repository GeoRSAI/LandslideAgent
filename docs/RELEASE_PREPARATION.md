# Release preparation

Prepared on 2026-10-06 from the current container working tree, including uncommitted source changes.

Included: source, service scripts, configuration, frontend assets, regression tests and methods documentation.

Excluded: container Git history, backup source files, experiment directories, coordinate manifests, logs, generated outputs, caches, private data and model weights. The full original local download remains separate.

Removed the hard-coded OpenTopography credential; replaced deployment paths with checkout-relative resources, configurable model placeholders and the active interpreter; created output directories at startup; fixed image-path containment checks; added installation metadata, MIT licensing and example configuration.

Before publication, rotate the credential present in the original deployment. Confirm the copyright holder and redistribution rights for code and web assets. Original history and the backup archive contain deployment details and must not be published as the release.

Real GPU inference and geographic service behaviour require validation with separately provisioned model assets. See VALIDATION.md for local checks.

The local update branch is based on GeoRSAI/LandslideAgent main at 3de28aa. The original GeoRSAI copyright notice and existing public dataset link are preserved. Upstream history is retained locally; the ZIP contains source files only. No remote changes were published.

Legacy test_agent_mode.py and test_followup_report_write_guard.py are absent from the container export. Current API/runtime regression suites are included instead; the old streaming helper expected a synchronous iterator and is not restored unchanged.

Dual-head inference scripts, labels and sanitized model configuration were added. LoRA/classification-head weights remain excluded. Python syntax and JSON parsing checks passed; standalone GPU inference remains unverified.
