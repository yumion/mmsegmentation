import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import GroupKFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-dir",
        "--target_dir",
        type=Path,
        help="root directory path of dataset. It has to be organized as follows: target_dir/case_dir/img_dir/img.jpg, target_dir/case_dir/mask_dir/mask.png",  # noqa
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        type=Path,
        help="output directory path of dataset",
    )
    parser.add_argument("--n-splits", "--n_splits", type=int, default=3, help="number of folds")
    parser.add_argument(
        "--image-dirname",
        "--image_dirname",
        type=str,
        default="frame",
        help="image directory name",
    )
    parser.add_argument("--copy", action="store_true", help="copy split dataset")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    for nb_fold, (train_cases, val_cases) in enumerate(
        split_folds(args.target_dir, args.n_splits, args.image_dirname), 1
    ):
        print(f"nb_fold: {nb_fold}")
        print(f"train cases: {len(train_cases)}")
        print(f"val cases: {len(val_cases)}")
        copy_folds(train_cases, args.out_dir / f"cv{nb_fold}/train", not args.copy)
        copy_folds(val_cases, args.out_dir / f"cv{nb_fold}/val", not args.copy)


def split_folds(
    target_dir: Path,
    n_splits: int,
    image_dirname: str = "frame",
) -> Tuple[List[str], List[str]]:
    cases, images = [], []
    for case_dir in target_dir.iterdir():
        if not case_dir.is_dir():
            continue
        for img_path in case_dir.glob(f"{image_dirname}/*"):
            if not img_path.is_file():
                continue
            cases.append(str(case_dir))
            images.append(str(img_path))

    group_kfold = GroupKFold(n_splits=n_splits)
    for train_index, val_index in group_kfold.split(images, groups=cases):
        train_cases = {cases[i] for i in train_index}
        val_cases = {cases[i] for i in val_index}
        train_images = [images[i] for i in train_index]
        val_images = [images[i] for i in val_index]
        print(f"train images: {len(train_images)}")
        print(f"val images: {len(val_images)}")
        yield list(train_cases), list(val_cases)


def copy_folds(target_fold: List[str], out_dir: Path, is_symlink: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    for case_dir in target_fold:
        src_dir = Path(case_dir)
        dst_dir = out_dir / src_dir.name
        if is_symlink:
            if dst_dir.exists():
                dst_dir.unlink()
            dst_dir.symlink_to(src_dir, target_is_directory=True)
        else:
            shutil.copytree(src_dir, dst_dir)


if __name__ == "__main__":
    main()
