from pathlib import Path

from utils import save_as_palette


def main():
    # convert 3 color part of instrument
    # SAR-RARP50(EndoVis2022) -> SurgToolLoc(EndoVis2023)
    cmap = {
        # 0: 0,  # background
        # 1: 1,  # clasper
        # 2: 2,  # wrist
        # 3: 3,  # shaft
        4: 0,  # needle
        5: 0,  # thread
        6: 2,  # suction
        7: 2,  # needle holder
        8: 0,  # clamps
        9: 0,  # catheter
    }
    # copy to CustomDataset.PALETTE
    # order is RGB
    palette = [
        [0, 0, 0],  # background
        [0, 255, 255],  # clasper
        [255, 255, 0],  # wrist
        [0, 255, 0],  # shaft
    ]

    parent_dir = Path("/data1/shared/miccai/EndoVis2022")
    child_dir_ptn = "video_*"
    seg_dir_name = "segmentation"
    save_dir_name = "3colors"

    save_as_palette(parent_dir, child_dir_ptn, seg_dir_name, save_dir_name, cmap, palette)


if __name__ == "__main__":
    main()
