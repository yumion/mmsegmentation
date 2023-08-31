import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_palette(label, cmap, palette):
    # print("label:", np.unique(label))
    seg_map = np.zeros_like(label)
    for label_id, palette_id in cmap.items():
        # print("before:", np.unique(seg_map))
        seg_map += np.where(np.asarray(label) == label_id, palette_id, 0).astype(seg_map.dtype)
        # print("after:", np.unique(seg_map))
    seg_img = Image.fromarray(seg_map).convert("P")
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    return seg_img


def save_as_palette(parent_dir, child_dir_ptn, seg_dir_name, save_dir_name, cmap, palette, suffix=".png"):
    for video_dir in parent_dir.glob(child_dir_ptn):
        if not video_dir.is_dir():
            print(f"{video_dir} is not a directory")
            continue

        save_dir = video_dir / save_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for label_file in tqdm(video_dir.glob(f"{seg_dir_name}/*{suffix}"), desc=video_dir.name):
            seg_map = cv2.imread(str(label_file), 0)
            # seg_map = Image.open(label_file).convert("P")
            seg_img = generate_palette(seg_map, cmap, palette)
            seg_img.save(save_dir / label_file.name)
