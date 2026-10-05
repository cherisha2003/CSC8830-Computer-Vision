"""Rebuild the example images shown on the website.

    python generate_examples.py

This reuses the saved SAM2 masks. To rerun SAM2 too, install Meta's package and
use `python generate_examples.py --run-sam2`. Model weights are downloaded on
first use, or supply the original SAM2 Tiny weights with --checkpoint PATH.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from segmentation import rgb_mask, thermal_mask, overlay, scores, fourier_views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-sam2", action="store_true")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    folder = Path(__file__).parent / "examples"
    settings = json.loads((folder / "settings.json").read_text())
    for kind, config in settings.items():
        image = np.asarray(Image.open(folder / f"{kind}.jpg").convert("RGB"))
        if kind == "rgb":
            mask, raw = rgb_mask(image, config["box"], config["foreground"],
                                 config["background"], config["kernel_size"])
            seeds = image.copy()
            x0, y0, x1, y1 = config["box"]
            cv2.rectangle(seeds, (x0, y0), (x1, y1), (255, 200, 0), 1)
            for label, color in [("foreground", (0, 255, 50)), ("background", (255, 70, 70))]:
                for x, y in config[label]:
                    cv2.circle(seeds, (x, y), 4, color, -1)
            Image.fromarray(seeds).save(folder / "rgb_seeds.png")
        else:
            mask, raw, threshold = thermal_mask(image, config["box"],
                                                config["polarity"], config["kernel_size"])
            config["threshold"] = threshold
        if args.run_sam2:
            from sam2_reference import make_reference
            sam = make_reference(image, config["box"], config.get("foreground"),
                                 config.get("background"), args.checkpoint)
            Image.fromarray(sam).save(folder / f"{kind}_sam2_mask.png")
        else:
            sam = np.asarray(Image.open(folder / f"{kind}_sam2_mask.png").convert("L"))
        images = {
            "raw": raw,
            "classical_mask": mask,
            "classical": overlay(image, mask),
            "sam2": overlay(image, sam, (0, 220, 90)),
            "comparison": overlay(overlay(image, mask), sam, (0, 220, 90)),
        }
        for name, result in images.items():
            Image.fromarray(result).save(folder / f"{kind}_{name}.png")
        config["scores"] = scores(mask, sam)
        print(kind, config["scores"])
    image = np.asarray(Image.open(folder / "rgb.jpg").convert("RGB"))
    for name, result in zip(["spectrum", "low_pass", "high_pass"], fourier_views(image)):
        Image.fromarray(result).save(folder / f"fourier_{name}.png")
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
