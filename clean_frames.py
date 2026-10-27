"""Remove the storyboard's baked-in labels/icons from each frame via inpainting.

Usage: python3 clean_frames.py <upload_dir> <out_dir>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# (source file, output name, [(x0, y0, x1, y1, mode)]) in 768x1376 source pixels.
# mode "text": mask only glyph pixels inside the box; "rect": mask the whole box.
FRAMES = [
    ("da675793-image.jpg", "01_reveal.png", [
        (185, 160, 585, 230, "text"), (280, 228, 490, 290, "text"),
        (170, 1100, 600, 1160, "text"), (310, 1158, 460, 1205, "text"),
        (25, 1285, 90, 1345, "rect"), (680, 1285, 745, 1345, "rect"),
    ]),
    ("8f1b3151-image.jpg", "02_detail.png", [
        (160, 145, 610, 220, "text"), (170, 218, 600, 270, "text"),
        (190, 1240, 580, 1350, "text"), (668, 1275, 750, 1350, "rect"),
    ]),
    ("b18c3373-image.jpg", "03_placement.png", [
        (205, 125, 570, 195, "text"), (255, 192, 520, 240, "text"),
        (195, 945, 575, 1005, "text"), (178, 1172, 590, 1234, "rect"),
        (660, 1275, 740, 1340, "rect"),
    ]),
    ("8043363e-image.jpg", "04_grip.png", [
        (170, 185, 595, 255, "text"), (235, 252, 535, 310, "text"),
        (190, 985, 580, 1055, "text"), (225, 1052, 550, 1110, "text"),
        (236, 1156, 534, 1214, "rect"), (675, 1285, 750, 1355, "rect"),
    ]),
    ("24543bf0-image.jpg", "05_final.png", [
        (170, 70, 600, 140, "text"), (190, 137, 580, 205, "text"),
        (165, 1245, 610, 1350, "text"), (660, 1265, 745, 1345, "rect"),
    ]),
]


def build_mask(img, boxes):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    background = cv2.medianBlur(gray.astype(np.uint8), 41).astype(np.int16)
    glyphs = (np.abs(gray - background) > 28).astype(np.uint8) * 255
    mask = np.zeros(gray.shape, np.uint8)
    for x0, y0, x1, y1, mode in boxes:
        if mode == "rect":
            mask[y0:y1, x0:x1] = 255
        else:
            mask[y0:y1, x0:x1] = glyphs[y0:y1, x0:x1]
    # Grow the mask to also cover anti-aliasing and drop shadows.
    return cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)


def main(src_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for src, name, boxes in FRAMES:
        img = cv2.imread(str(Path(src_dir) / src))
        mask = build_mask(img, boxes)
        clean = cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)
        # Soften the patched areas so they blend into the surrounding bokeh.
        soft = cv2.GaussianBlur(clean, (0, 0), 3)
        alpha = cv2.GaussianBlur(mask, (0, 0), 6)[..., None] / 255.0
        clean = (clean * (1 - alpha) + soft * alpha).astype(np.uint8)
        cv2.imwrite(str(out / name), clean)
        cv2.imwrite(str(out / name.replace(".png", "_mask.png")), mask)


if __name__ == "__main__":
    main(*sys.argv[1:3])
