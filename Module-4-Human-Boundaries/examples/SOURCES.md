# Example sources

## RGB

- File: `rgb.jpg`, unmodified copy of OpenCV's `messi5.jpg` sample.
- Source: https://github.com/opencv/opencv/blob/master/samples/data/messi5.jpg
- Original photo credit visible in the image: WEEkmedia. The watermark is retained.
- Included for this academic computer vision demonstration; no ownership of the photograph is claimed.

## Thermal

- Dataset: LLVIP: A Visible-infrared Paired Dataset for Low-light Vision.
- X. Jia et al., ICCV Workshops, 2021.
- Dataset website: https://bupt-ai-cz.github.io/LLVIP/
- Original frame: https://github.com/bupt-ai-cz/LLVIP/blob/main/FusionGAN/Train_LLVIP_ir/010001.jpg
- `thermal.jpg` is the crop (left=440, top=345, right=580, bottom=635), saved as JPEG at quality 95. Both algorithms use this exact cropped file.
- This is a real infrared intensity image. It is not a temperature-calibrated array.
- License: see `LLVIP_LICENSE.md`. Noncommercial academic use. Cite LLVIP for this frame and any derivative results.

## Outputs

All classical masks and overlays were computed by the included scripts. Reference masks were computed with the original SAM2 Hiera Tiny model from Meta, on CPU, with multimask output and optional postprocessing disabled. These are experimental outputs, not official LLVIP annotations or manually labeled ground truth.

- Model source: https://github.com/facebookresearch/sam2
- Checkpoint: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
- RGB foreground/background points and target boxes: `settings.json`.
