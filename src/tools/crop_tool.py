from __future__ import annotations


def crop_or_tile(image_info: dict, tile_size: int = 512) -> list[dict]:
    width = image_info["width"]
    height = image_info["height"]
    image_path = image_info.get("image_path", "")
    tiles = []
    tile_id = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tiles.append(
                {
                    "tile_id": tile_id,
                    "x": x,
                    "y": y,
                    "w": min(tile_size, width - x),
                    "h": min(tile_size, height - y),
                    "image_path": image_path,
                }
            )
            tile_id += 1
    return tiles
