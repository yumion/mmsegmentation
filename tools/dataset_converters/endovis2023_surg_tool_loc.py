from pathlib import Path

from utils import save_as_palette


def main():
    # convert 3 color part of instrument
    # 3色 + 術具ナンバリング -> 3色のみsemantic segmentation mask
    cmap = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 1,
        5: 2,
        6: 3,
        7: 1,
        8: 2,
        9: 3,
        10: 1,
        11: 2,
        12: 3,
    }
    # copy to CustomDataset.PALETTE
    # order is RGB
    palette = [
        [0, 0, 0],  # background
        [0, 255, 255],  # clasper
        [255, 255, 0],  # wrist
        [0, 255, 0],  # shaft
    ]

    parent_dir = Path("/data1/shared/miccai/EndoVis2023/SurgToolLoc/v1.2")
    child_dir_ptn = "*_clip_*"
    seg_dir_name = "mask"
    save_dir_name = "semantic_mask"

    save_as_palette(parent_dir, child_dir_ptn, seg_dir_name, save_dir_name, cmap, palette)


if __name__ == "__main__":
    main()
