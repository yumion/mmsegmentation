# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List, Optional

import logging
from tqdm import tqdm

import mmengine
from mmengine.logging import print_log
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class EndoVisDataset(BaseSegDataset):
    """EndoVis dataset."""

    METAINFO = dict(
        classes=(
            "background",
            "clasper",
            "wrist",
            "shaft",
        ),
        palette=[
            [0, 0, 0],  # background
            [0, 255, 255],  # clasper
            [255, 255, 0],  # wrist
            [0, 255, 0],  # shaft
        ],
    )

    def __init__(
        self,
        ann_file: Optional[str] = None,
        img_suffix: str = ".png",
        seg_map_suffix: str = ".png",
        data_root: Optional[str] = None,
        **kwargs,
    ) -> None:
        # ignore `_join_prefix()`
        super().__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            data_root=None,
            lazy_init=True,
            **kwargs,
        )
        self.data_root = Path(data_root if data_root is not None else "")
        if not kwargs.get("lazy_init", False):
            self.full_init()

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get("img_path", None)
        ann_dir = self.data_prefix.get("seg_map_path", None)
        if ann_dir is not None:
            find_pattern = f"{ann_dir}/*{self.seg_map_suffix}"
        else:
            find_pattern = f"{img_dir}/*{self.img_suffix}"
        # in case there is train/val split text files
        ann_file = self.data_root / self.ann_file
        if not ann_file.is_dir() and self.ann_file:
            assert ann_file.is_file(), f"Failed to load `ann_file` {ann_file}"
            lines = mmengine.list_from_file(ann_file, backend_args=self.backend_args)
            for line in lines:
                video_name = line.strip()
                data_root = self.data_root / video_name
                with tqdm(data_root.glob(find_pattern)) as pbar:
                    for data_file in pbar:
                        img_name = data_file.stem
                        pbar.set_description(f"loading: {video_name}/{img_name}")
                        data_info = dict(
                            img_path=(data_root / img_dir / img_name).with_suffix(self.img_suffix),
                            label_map=self.label_map,
                            reduce_zero_label=self.reduce_zero_label,
                            seg_fields=[],
                        )
                        if ann_dir is not None:
                            data_info.update(
                                seg_map_path=(data_root / ann_dir / img_name).with_suffix(self.seg_map_suffix),
                            )
                        pbar.set_postfix(data_info)
                        data_list.append(data_info)
        # in case dataset were splitted train/val directory
        else:
            with tqdm(self.data_root.glob(f"*/{find_pattern}")) as pbar:
                for data_file in pbar:
                    img_name = data_file.stem
                    video_name = data_file.parts[-3]
                    pbar.set_description(f"loading: {video_name}/{img_name}")
                    data_info = dict(
                        img_path=(self.data_root / video_name / img_dir / img_name).with_suffix(self.img_suffix),
                        label_map=self.label_map,
                        reduce_zero_label=self.reduce_zero_label,
                        seg_fields=[],
                    )
                    if ann_dir is not None:
                        data_info.update(
                            seg_map_path=(self.data_root / video_name / ann_dir / img_name).with_suffix(
                                self.seg_map_suffix
                            )
                        )
                    pbar.set_postfix(data_info)
                    data_list.append(data_info)
        data_list = sorted(data_list, key=lambda x: x["img_path"])
        print_log(f'Loaded {len(data_list)} images', logger="current", level=logging.INFO)
        return data_list
