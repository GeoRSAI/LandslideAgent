#!/usr/bin/env python3
"""Batch-test the landslide agent over a folder of images.

Talks to the ALREADY-RUNNING local service (scripts/start_frontend_all.sh /
relaunch_llm.sh) over HTTP -- it does NOT load the model itself, so it runs
under any Python (system python is fine) and needs no extra packages.

For every image it POSTs /v1/agent/analyze, saves the full JSON response and the
final report text per image, and writes a summary.csv / summary.jsonl.

Usage (from the repo root, service must be up and model_status=ready):

    python scripts/batch_test.py --images data --mode agent
    python scripts/batch_test.py --images models/test-image-qwen --mode free --lat 29.6 --lon 103.0
    python scripts/batch_test.py --manifest cases.csv --mode agent      # cases.csv: image,lat,lon

Modes:  agent = rule/contract arm (rule container) | free = free arm (free container)
        graph = deterministic LangGraph pipeline (debug leftover; usually ignore)

Results land in:  outputs/batch_<mode>_<timestamp>/
    <image>.response.json   full API response
    <image>.report.txt      final report text (message content)
    summary.jsonl           one JSON line per image
    summary.csv             image, status, has_landslide, classification_confidence,
                            landslide_type, elapsed_s, report_path, error
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_HOST = os.getenv("BATCH_HOST", "127.0.0.1:8003")


def _http_json(url: str, payload: dict | None, timeout: float):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_health(host: str, wait_s: int) -> dict:
    url = f"http://{host}/health"
    deadline = time.time() + max(0, wait_s)
    last = {}
    while True:
        try:
            last = _http_json(url, None, timeout=15)
            if last.get("model_ready") or str(last.get("model_status")) == "ready":
                return last
        except Exception as exc:  # noqa: BLE001
            last = {"error": str(exc)}
        if time.time() >= deadline:
            return last
        time.sleep(5)


def _collect_images(images_dir: str) -> list[dict]:
    root = Path(images_dir)
    if not root.exists():
        sys.exit(f"images dir not found: {root}")
    items = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            items.append({"image": str(p.resolve()), "lat": None, "lon": None})
    return items


def _collect_manifest(manifest: str) -> list[dict]:
    items = []
    with open(manifest, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            img = (row.get("image") or row.get("path") or "").strip()
            if not img:
                continue
            lat = row.get("lat") or row.get("latitude")
            lon = row.get("lon") or row.get("longitude")
            items.append({
                "image": str(Path(img).resolve()),
                "lat": float(lat) if lat not in (None, "") else None,
                "lon": float(lon) if lon not in (None, "") else None,
                "expected_type": (row.get("expected_type") or row.get("label") or "").strip(),
            })
    return items


def _workflow(response: dict) -> list[dict]:
    """Ordered tool calls the agent actually made, from agent_trace."""
    steps = []
    for item in response.get("agent_trace") or []:
        if not isinstance(item, dict):
            continue
        steps.append({
            "tool": item.get("tool"),
            "status": item.get("status"),
            "execution_state": item.get("execution_state"),
        })
    return steps


def _report_object(response: dict, report_path: str) -> tuple[dict | None, str]:
    """Resolve the final structured report and record where it came from."""
    report = None
    if report_path and os.path.isfile(report_path):
        try:
            with open(report_path, encoding="utf-8") as fh:
                report = json.load(fh)
        except Exception:  # noqa: BLE001
            report = None
    if isinstance(report, dict):
        return report, "written_report"
    # The free arm may legitimately end without report.write.  Preserve the
    # actual fusion output instead of fabricating a report or silently losing it.
    # A rules-controlled run may try fusion early, receive a precondition
    # error, collect more evidence, then succeed on a later attempt.  Search
    # backward and accept only an error-free fusion result; an error observation
    # is process evidence, never a final report.
    for item in reversed(response.get("agent_trace") or []):
        if str(item.get("tool")) == "fuse.decision":
            candidate = item.get("output") or item.get("data") or item
            if isinstance(candidate, dict) and not candidate.get("error") and "has_landslide" in candidate:
                return candidate, "fuse_decision_trace"
    # Free arm: fuse.decision only signals "write the report"; the verdict is the
    # one stated in the report written from the agent's own evidence.
    answer = response.get("composed_answer")
    if isinstance(answer, dict) and answer.get("has_landslide") is not None:
        return answer, "composed_report"
    return None, "unavailable"


def _report_fields(report: dict | None) -> dict:
    """Pull comparable headline fields from a resolved structured report."""
    out = {"has_landslide": "", "classification_confidence": "",
           "landslide_type": ""}
    if isinstance(report, dict):
        for k in out:
            if k in report and report[k] is not None:
                out[k] = report[k]
    return out


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _workflow_text(trace: list[dict]) -> str:
    """Human-readable audit trail; workflow.json retains the full raw trace."""
    lines = []
    for index, item in enumerate(trace, 1):
        tool = item.get("tool") or item.get("name") or "(non-tool event)"
        state = item.get("execution_state") or item.get("status") or "unknown"
        lines.append(f"{index}. {tool}  [{state}]")
        for key in ("arguments", "verification", "error", "message"):
            if key in item and item[key] not in (None, "", {}, []):
                lines.append("   " + key + ": " + json.dumps(
                    item[key], ensure_ascii=False, default=str))
    return "\n".join(lines) or "(no tool calls recorded)"


def _case_id(index: int, image_path: str) -> str:
    """Avoid overwriting results when different folders contain the same stem."""
    return f"{index:04d}_{Path(image_path).stem}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-test the landslide agent.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--images", help="directory of images (recursed)")
    src.add_argument("--manifest", help="CSV with columns image[,lat,lon]")
    ap.add_argument("--mode", default=os.getenv("BATCH_MODE", "agent"),
                    choices=["agent", "free", "graph"],
                    help="agent=rule/contract arm, free=free arm, graph=langgraph")
    ap.add_argument("--host", default=DEFAULT_HOST, help="service host:port")
    ap.add_argument("--out", default="", help="output dir (default outputs/batch_<mode>_<ts>)")
    ap.add_argument("--lat", type=float, default=None, help="latitude for all images")
    ap.add_argument("--lon", type=float, default=None, help="longitude for all images")
    ap.add_argument("--nearby-radius", type=int, default=300)
    ap.add_argument("--max-turns", type=int, default=30,
                    help="maximum model calls per image (default: 30)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of images (0=all)")
    ap.add_argument("--timeout", type=float, default=1800, help="per-image seconds")
    ap.add_argument("--health-wait", type=int, default=0,
                    help="seconds to wait for model_ready before starting")
    ap.add_argument("--prompt", default="Analyse this scene for landslide evidence.")
    ap.add_argument("--allow-non-dualhead", action="store_true",
                    help="proceed even if the service is not using the dual-head model")
    args = ap.parse_args()

    if args.manifest:
        cases = _collect_manifest(args.manifest)
    else:
        cases = _collect_images(args.images or "data")
    if args.lat is not None or args.lon is not None:
        for c in cases:
            c["lat"] = args.lat if c["lat"] is None else c["lat"]
            c["lon"] = args.lon if c["lon"] is None else c["lon"]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        sys.exit("no images found")

    health = _wait_health(args.host, args.health_wait)
    if not (health.get("model_ready") or str(health.get("model_status")) == "ready"):
        print("WARNING: service not reporting model_ready:", json.dumps(health))
        print("Start it first:  bash scripts/start_frontend_all.sh   (or ~/relaunch_llm.sh)")
        print("Then check:      curl -s http://%s/health" % args.host)
        if args.health_wait <= 0:
            sys.exit(1)

    # Confirm the service is actually running on the dual-head model.
    dual_head = bool(health.get("dual_head_loaded"))
    print("model_path=%s dual_head_loaded=%s adapter_always_on=%s labels=%s mock=%s" % (
        health.get("model_path"), dual_head,
        health.get("dual_head_adapter_always_on"),
        health.get("dual_head_labels"), health.get("mock_mode")))
    if not dual_head and not health.get("mock_mode") and not args.allow_non_dualhead:
        print("ERROR: service is NOT using the dual-head model (dual_head_loaded=false).")
        print("start_frontend_all.sh keeps CLS_BACKEND=dualhead and "
              "LLM_DUAL_HEAD_ADAPTER_ALWAYS_ON=1 by default; restart with those, or pass "
              "--allow-non-dualhead to override.")
        sys.exit(2)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out or f"outputs/batch_{args.mode}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    case_manifest = []
    for index, case in enumerate(cases, 1):
        image_path = case["image"]
        case_manifest.append({
            "case_id": _case_id(index, image_path), "index": index,
            "image": image_path, "image_sha256": _file_sha256(image_path),
            "image_bytes": os.path.getsize(image_path),
            "latitude": case["lat"], "longitude": case["lon"],
            "expected_type": case.get("expected_type") or None,
        })
    (out_dir / "cases.json").write_text(
        json.dumps(case_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Provenance for the whole run: which model/backend produced these reports.
    (out_dir / "run_meta.json").write_text(json.dumps({
        "timestamp": ts, "mode": args.mode, "host": args.host,
        "prompt": args.prompt, "nearby_radius": args.nearby_radius,
        "max_turns": args.max_turns, "timeout_s": args.timeout,
        "num_images": len(cases),
        "dual_head_loaded": dual_head,
        "dual_head_adapter_always_on": health.get("dual_head_adapter_always_on"),
        "dual_head_labels": health.get("dual_head_labels"),
        "model_path": health.get("model_path"),
        "lora_loaded": health.get("lora_loaded"),
        "mock_mode": health.get("mock_mode"),
        "batch_script_sha256": _file_sha256(__file__),
        "case_manifest": "cases.json",
        "health": health,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    url = f"http://{args.host}/v1/agent/analyze"
    summary_jsonl = open(out_dir / "summary.jsonl", "w", encoding="utf-8")
    csv_fh = open(out_dir / "summary.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_fh)
    writer.writerow(["case_id", "image", "expected_type", "mode", "degraded", "has_landslide",
                     "classification_confidence", "landslide_type",
                     "n_tool_calls", "workflow", "elapsed_s", "report_path",
                     "report_record", "report_source", "response_record",
                     "workflow_record", "error"])

    print(f"mode={args.mode} host={args.host} images={len(cases)} -> {out_dir}")
    for i, case in enumerate(cases, 1):
        img = case["image"]
        case_id = _case_id(i, img)
        content = [{"type": "image", "image_path": img}, {"type": "text", "text": args.prompt}]
        payload = {
            "messages": [{"role": "user", "content": content}],
            "agent_mode": args.mode,
            "nearby_radius": args.nearby_radius,
            "max_turns": args.max_turns,
        }
        if case["lat"] is not None:
            payload["latitude"] = case["lat"]
        if case["lon"] is not None:
            payload["longitude"] = case["lon"]

        t0 = time.time()
        err = ""
        response: dict = {}
        try:
            response = _http_json(url, payload, timeout=args.timeout)
        except urllib.error.HTTPError as exc:
            err = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
        elapsed = round(time.time() - t0, 1)

        response_record = out_dir / f"{case_id}.response.json"
        response_record.write_text(
            json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
        msg = (((response.get("choices") or [{}])[0].get("message") or {}).get("content")
               if response else "")
        if isinstance(msg, list):
            msg = " ".join(str(p.get("text", "")) for p in msg if isinstance(p, dict))
        (out_dir / f"{case_id}.report.txt").write_text(str(msg or ""), encoding="utf-8")
        if isinstance(response.get("structured_report"), dict):
            (out_dir / f"{case_id}.structured_report.json").write_text(
                json.dumps(response["structured_report"], ensure_ascii=False, indent=2), encoding="utf-8")

        report_path = str(response.get("report_path") or "")
        report, report_source = _report_object(response, report_path)
        report_record = out_dir / f"{case_id}.final_report.json"
        report_record.write_text(json.dumps({
            "source": report_source, "reported_path": report_path or None,
            "report": report,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        fields = _report_fields(report)
        trace = [item for item in (response.get("agent_trace") or []) if isinstance(item, dict)]
        steps = _workflow(response)
        wf_compact = " > ".join(
            f"{s['tool']}:{s.get('execution_state') or s.get('status')}" for s in steps)
        workflow_record = out_dir / f"{case_id}.workflow.json"
        workflow_record.write_text(json.dumps({
            "case_id": case_id, "image": img,
            "expected_type": case.get("expected_type") or None, "request": payload,
            "elapsed_s": elapsed, "error": err, "agent_trace": trace,
            "critic": response.get("critic"), "history": response.get("history"),
            "artifacts": response.get("artifacts"), "report_source": report_source,
            "report_record": str(report_record),
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (out_dir / f"{case_id}.workflow.txt").write_text(
            _workflow_text(trace), encoding="utf-8")
        degraded = bool((response.get("critic") or {}).get("degraded"))
        row = {
            "case_id": case_id, "image": img,
            "expected_type": case.get("expected_type") or None,
            "mode": args.mode, "degraded": degraded,
            "n_tool_calls": len(steps), "workflow": steps,
            "elapsed_s": elapsed, "report_path": report_path,
            "report_record": str(report_record), "report_source": report_source,
            "response_record": str(response_record), "workflow_record": str(workflow_record),
            "error": err, **fields,
        }
        summary_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary_jsonl.flush()
        writer.writerow([case_id, img, case.get("expected_type") or "", args.mode, degraded, fields["has_landslide"],
                         fields["classification_confidence"], fields["landslide_type"],
                         len(steps), wf_compact,
                         elapsed, report_path, str(report_record), report_source,
                         str(response_record), str(workflow_record), err])
        csv_fh.flush()
        status = "ERR " + err if err else ("degraded" if degraded else "ok")
        print(f"[{i}/{len(cases)}] {case_id}: {status}  has_landslide={fields['has_landslide']} "
              f"cls_conf={fields['classification_confidence']} ({elapsed}s)")

    summary_jsonl.close()
    csv_fh.close()
    print(f"done -> {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
