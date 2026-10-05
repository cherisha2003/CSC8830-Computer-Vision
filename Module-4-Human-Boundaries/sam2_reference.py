"""Make a reference mask with SAM2, separately from the classical code.

Example:
    python sam2_reference.py image.jpg mask.png 20 10 400 500

Install Meta's official sam2 package first. A first run downloads its weights.
For the saved examples, use generate_examples.py --run-sam2 instead.
"""
import argparse
import numpy as np
from PIL import Image


def make_reference(image, box, foreground=None, background=None, checkpoint=None):
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint:
        from sam2.build_sam import build_sam2
        model = build_sam2("configs/sam2/sam2_hiera_t.yaml", checkpoint,
                           device=device, apply_postprocessing=False)
        predictor = SAM2ImagePredictor(model)
    else:
        predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-tiny", device=device, apply_postprocessing=False
        )
    points = (foreground or []) + (background or [])
    kwargs = {}
    if points:
        kwargs["point_coords"] = np.asarray(points, dtype=np.float32)
        kwargs["point_labels"] = np.asarray([1]*len(foreground or []) + [0]*len(background or []))
    with torch.inference_mode():
        predictor.set_image(image.copy())
        masks, _, _ = predictor.predict(box=np.asarray(box), multimask_output=False, **kwargs)
    return masks[0].astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser(description="Generate a SAM2 comparison mask")
    parser.add_argument("image")
    parser.add_argument("output")
    parser.add_argument("box", nargs=4, type=int)
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    h, w = image.shape[:2]
    x0, y0, x1, y1 = args.box
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        parser.error(f"Box must fit image dimensions {w}x{h}")
    mask = make_reference(image, args.box)
    Image.fromarray(mask).save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
