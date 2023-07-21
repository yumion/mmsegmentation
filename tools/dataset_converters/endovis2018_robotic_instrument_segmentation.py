from pathlib import Path

from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np


def main():
    # convert 3 color part of instrument
    # RoboticInstrumentSegmentation(EndoVis2018) -> SurgToolLoc(EndoVis2023)
    original_palette = [
        [0, 0, 0],  # background-tissue
        [0, 255, 0],  # instrument-shaft
        [0, 255, 255],  # instrument-clasper
        [125, 255, 12],  # instrument-wrist
        [255, 55, 0],  # kidney-parenchyma
        [24, 55, 125],  # converd-kidney
        [187, 155, 25],  # thread
        [0, 255, 125],  # clamps
        [255, 255, 125],  # suturing-needle
        [123, 15, 175],  # suction-instrument
        [124, 155, 5],  # intestine
        [12, 255, 141],  # ultrasound-probe
    ]
    cmap = {
        # 0: 0,  # background
        2: 1,  # clasper
        3: 2,  # wrist
        1: 3,  # shaft
        9: 3,  # suction
        # 11: 3,  # probe
    }
    # copy to CustomDataset.PALETTE
    # order is RGB
    palette = [
        [0, 0, 0],  # background
        [0, 255, 255],  # clasper
        [255, 255, 0],  # wrist
        [0, 255, 0],  # shaft
    ]

    parent_dir = Path("/data1/shared/miccai/EndoVis2018/test")
    child_dir_ptn = "seq_*"
    seg_dir_name = "labels"
    save_dir_name = "3colors"

    save_as_palette_endovis2018(
        parent_dir, child_dir_ptn, seg_dir_name, save_dir_name, cmap, original_palette, palette
    )


def save_as_palette_endovis2018(
    parent_dir, child_dir_ptn, seg_dir_name, save_dir_name, cmap, original_palette, palette, suffix=".png"
):
    for video_dir in parent_dir.glob(child_dir_ptn):
        if not video_dir.is_dir():
            print(f"{video_dir} is not a directory")
            continue

        save_dir = video_dir / save_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for label_file in tqdm(video_dir.glob(f"{seg_dir_name}/*{suffix}"), desc=video_dir.name):
            label_rgb = cv2.imread(str(label_file))[..., ::-1]
            seg_img = generate_palette(label_rgb, cmap, original_palette, palette)
            seg_img.save(save_dir / label_file.name)


def generate_palette(label_rgb, cmap, original_palette, palette):
    seg_map = np.zeros_like(label_rgb[..., 0])
    for label_id, palette_id in cmap.items():
        label_color = original_palette[label_id]
        seg_map += np.where(np.all(label_rgb == label_color, axis=-1), palette_id, 0).astype(seg_map.dtype)
    seg_img = Image.fromarray(seg_map).convert("P")
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    return seg_img


if __name__ == "__main__":
    main()
