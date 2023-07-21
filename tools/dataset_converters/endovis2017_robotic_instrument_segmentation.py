from pathlib import Path

import cv2
from tqdm import tqdm

from utils import generate_palette, save_as_palette


def main():
    # convert 3 color part of instrument
    # RoboticInstrumentSegmentation(EndoVis2017) -> SurgToolLoc(EndoVis2023)
    cmap = {
        # 0: 0,  # background
        30: 1,  # clasper
        20: 2,  # wrist
        10: 3,  # shaft
        # 40: 0,  # Probe
    }
    # copy to CustomDataset.PALETTE
    # order is RGB
    palette = [
        [0, 0, 0],  # background
        [0, 255, 255],  # clasper(水色)
        [255, 255, 0],  # wrist(黄色)
        [0, 255, 0],  # shaft(緑)
    ]

    dataset_type = "train"
    parent_dir = Path(f"/data1/shared/miccai/EndoVis2017/{dataset_type}")
    child_dir_ptn = "instrument_dataset_*"
    seg_dir_name = "ground_truth"
    save_dir_name = "3colors"

    if dataset_type == "train":
        save_as_palette_for_train(parent_dir, child_dir_ptn, seg_dir_name, save_dir_name, cmap, palette)
    elif dataset_type == "test":
        save_as_palette(parent_dir, child_dir_ptn, seg_dir_name, save_dir_name, cmap, palette)


def save_as_palette_for_train(parent_dir, child_dir_ptn, seg_dir_name, save_dir_name, cmap, palette):
    for video_dir in parent_dir.glob(child_dir_ptn):
        if not video_dir.is_dir():
            print(f"{video_dir} is not a directory")
            continue

        save_dir = video_dir / save_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)

        sub_seg_dirs = list((video_dir / seg_dir_name).iterdir())
        for label_file in tqdm(sub_seg_dirs[0].glob("*.png"), desc=video_dir.name):
            # 術具ごとに画像ファイルが分かれているので1枚にする
            seg_map = cv2.imread(str(label_file), 0)
            for subdir in sub_seg_dirs[1:]:
                tmp = cv2.imread(str(subdir / label_file.name), 0)
                seg_map = cv2.add(seg_map, tmp)
            seg_img = generate_palette(seg_map, cmap, palette)
            seg_img.save(save_dir / label_file.name)


if __name__ == "__main__":
    main()
