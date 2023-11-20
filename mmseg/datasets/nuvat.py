import csv
import json
import logging
from pathlib import Path
from typing import List, Optional

import mmengine
from mmengine.logging import MMLogger
from mmseg.registry import DATASETS
from tqdm import tqdm

from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class NuvatDataset(BaseSegDataset):
    """Nuvat dataset

    ├── data_root
    │   ├── 0000_0000
    │   │   ├── data_prefix.img_path
    │   │   │   ├── xxx{img_suffix}
    │   │   │   ├── yyy{img_suffix}
    │   │   │   ├── zzz{img_suffix}
    │   │   ├── data_prefix.seg_map_path
    │   │   │   ├── xxx{img_suffix}
    │   │   │   ├── yyy{img_suffix}
    │   │   │   ├── zzz{img_suffix}
    │   ├── 0000_0001
    │   │   ├── data_prefix.img_path
    │   │   │   ├── xxx{img_suffix}
    │   │   │   ├── yyy{img_suffix}
    │   │   │   ├── zzz{img_suffix}
    │   │   ├── data_prefix.seg_map_path
    │   │   │   ├── xxx{img_suffix}
    │   │   │   ├── yyy{img_suffix}
    │   │   │   ├── zzz{img_suffix}

    """

    METAINFO = dict(
        classes=("background",),
        palette=[
            [0, 0, 0],  # background
        ],
    )

    def __init__(
        self,
        ann_file: str = "",
        img_suffix: str = ".png",
        seg_map_suffix: str = ".png",
        data_root: Optional[str] = None,
        dump_path: Optional[str] = None,
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

        if dump_path is not None:
            dump_path = Path(dump_path)
            self.dump_annotations(dump_path)

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
                                seg_map_path=(data_root / ann_dir / img_name).with_suffix(
                                    self.seg_map_suffix
                                ),
                            )
                        # pbar.set_postfix(data_info)
                        data_list.append(data_info)
        # in case dataset were splitted train/val directory
        else:
            with tqdm(self.data_root.glob(f"*/{find_pattern}")) as pbar:
                for data_file in pbar:
                    img_name = data_file.stem
                    video_name = data_file.parts[-3]
                    pbar.set_description(f"loading: {video_name}/{img_name}")
                    data_info = dict(
                        img_path=(
                            self.data_root / video_name / img_dir / img_name
                        ).with_suffix(self.img_suffix),
                        label_map=self.label_map,
                        reduce_zero_label=self.reduce_zero_label,
                        seg_fields=[],
                    )
                    if ann_dir is not None:
                        data_info.update(
                            seg_map_path=(
                                self.data_root / video_name / ann_dir / img_name
                            ).with_suffix(self.seg_map_suffix)
                        )
                    # pbar.set_postfix(data_info)
                    data_list.append(data_info)

        assert (
            data_list
        ), f"ERROR: There is no data for loading. Is the data path wrong?: {self.data_root}"

        data_list = sorted(data_list, key=lambda x: x["img_path"])
        logger = MMLogger.get_current_instance()
        logger.info(f"Loaded {len(data_list)} images", logger="current", level=logging.INFO)
        return data_list

    def dump_annotations(self, dump_path: Path, label_map_file: str = "metainfo.json") -> None:
        annotations = []
        for idx in range(len(self)):
            data_info = self.get_data_info(idx)
            annotations.append([str(data_info["img_path"]), str(data_info["seg_map_path"])])
        with open(dump_path, "w") as fw:
            writer = csv.writer(fw)
            writer.writerows(annotations)
        with open(dump_path.parent / label_map_file, "w") as fw:
            json.dump(self._metainfo, fw, indent=4)
        logger = MMLogger.get_current_instance()
        logger.info(f"Dumped annotations to {dump_path}", logger="current", level=logging.INFO)
