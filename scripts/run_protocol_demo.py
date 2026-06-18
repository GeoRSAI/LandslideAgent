from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent.default_server import create_default_server


def rpc(server, req_id: int, method: str, params: dict | None = None) -> dict:
    return server.handle({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run landslide pipeline via JSON-RPC tool protocol")
    parser.add_argument("--image", required=True, help="Input tif path")
    parser.add_argument("--lat", type=float, required=True, help="Observation latitude for geo tools")
    parser.add_argument("--lon", type=float, required=True, help="Observation longitude for geo tools")
    parser.add_argument("--radius", type=int, default=300, help="Nearby search radius in meters")
    parser.add_argument("--out", default="outputs/report.protocol.json", help="Output report path")
    parser.add_argument("--thresholds", default="configs/thresholds.json", help="Threshold config")
    args = parser.parse_args()

    server = create_default_server(args.thresholds)

    _ = rpc(server, 1, "tools/list")
    image_info = rpc(server, 2, "tools/call", {"name": "tiff.info", "arguments": {"image_path": args.image}})[
        "result"
    ]["content"]
    tiles = []
    if int(image_info.get("width", 0) or 0) > 1024 or int(image_info.get("height", 0) or 0) > 1024:
        tiles = rpc(server, 3, "tools/call", {"name": "image.tile", "arguments": {"image_info": image_info}})["result"][
            "content"
        ]["tiles"]
    stage1 = rpc(server, 4, "tools/call", {"name": "llm.first_pass", "arguments": {"image_info": image_info}})[
        "result"
    ]["content"]
    stage2 = rpc(server, 5, "tools/call", {"name": "seg.run", "arguments": {"image_info": image_info}})["result"][
        "content"
    ]
    refine_args = {"tiles": tiles, "image_info": image_info, "segmentation": stage2} if tiles else {
        "image_info": image_info,
        "segmentation": stage2,
    }
    refinement = rpc(server, 6, "tools/call", {"name": "seg.refine", "arguments": refine_args})["result"]["content"]
    classification = rpc(
        server,
        7,
        "tools/call",
        {"name": "cls.run", "arguments": {"image_info": image_info}},
    )["result"]["content"]
    geo_nearby = rpc(
        server,
        8,
        "tools/call",
        {"name": "geo.nearby", "arguments": {"lat": args.lat, "lon": args.lon, "radius": args.radius}},
    )["result"]["content"]
    geo_background = rpc(
        server,
        9,
        "tools/call",
        {"name": "geo.background", "arguments": {"lat": args.lat, "lon": args.lon}},
    )["result"]["content"]

    final = rpc(
        server,
        10,
        "tools/call",
        {
            "name": "fuse.decision",
            "arguments": {
                "stage1": stage1,
                "refinement": refinement,
                "segmentation": stage2,
                "classification": classification,
                "geo_context": {
                    "background": geo_background,
                    "nearby": geo_nearby,
                },
            },
        },
    )["result"]["content"]

    report = {
        "image_info": image_info,
        "stage1_llm_judge": stage1,
        "stage2_segmentation": stage2,
        "stage3_segmentation_guided_refine": refinement,
        "stage4_classification": classification,
        "geo_context": {
            "background": geo_background,
            "nearby": geo_nearby,
        },
        "final": final,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"final": final, "report": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
