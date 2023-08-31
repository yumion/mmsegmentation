import argparse
import os
from glob import glob
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.structures import PixelData
from mmseg.apis import inference_model, init_model
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg inferencer model")
    parser.add_argument("config", type=Path, help="train config file path")
    parser.add_argument("checkpoint", type=Path, help="checkpoint file")
    parser.add_argument(
        "device", type=int, default=0, help="device used for inference. `-1` means using cpu."
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        help="input directory of images to be predicted. It can be used wildcard.",
    )
    parser.add_argument(
        "--show-dir",
        type=Path,
        help="directory where painted images will be saved. "
        "If specified, it will be automatically saved "
        "to the work_dir/timestamp/show_dir",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="The directory to save output prediction for offline evaluation",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tta", action="store_true", help="Test time augmentation")
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main() -> None:
    args = parse_args()
    assert (
        args.show_dir is not None or args.out is not None
    ), "Either --show-dir or --out should be specified."

    model = MMSegInferencer(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        cfg_options=args.cfg_options,
        launcher=args.launcher,
        tta=args.tta,
    )

    with tqdm(sorted(glob(f"{args.target_dir}/*.png", recursive=True))) as pbar:
        for img_path in pbar:
            img_path = Path(img_path)

            # inputの画像のフォルダ構成を保ったまま保存する
            parent_dir = img_path.parent.name
            if "*" in args.target_dir:
                parent_dir = str(img_path.parent).replace(args.target_dir.split("*")[0], "")
            pbar.set_description(f"{parent_dir}/{img_path.name}")

            result = model(str(img_path))
            mask_p = model.putpalette(result)  # PIL palette
            blend = model.overlay(result, rgb_to_bgr=True)  # BGR ndarray

            verbose = {}
            if args.out is not None:
                out_dir = args.out / parent_dir / "preds"
                out_dir.mkdir(parents=True, exist_ok=True)
                mask_p.save(out_dir / img_path.name)
                verbose["pred"] = str(out_dir / img_path.name)

            if args.show_dir is not None:
                show_dir = args.show_dir / parent_dir / "vis"
                show_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(show_dir / img_path.name), blend)
                verbose["show"] = str(show_dir / img_path.name)

            pbar.set_postfix(verbose)


class MMSegInferencer:
    ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

    def __init__(
        self,
        config: Union[str, Path],
        checkpoint: Union[str, Path],
        device: int = 0,
        cfg_options: Optional[dict] = None,
        launcher: str = "none",
        tta: bool = False,
    ) -> None:
        self.cfg = self._build_config(config, checkpoint, cfg_options, launcher, tta)
        self.device = f"cuda:{device}" if device >= 0 and torch.cuda.is_available() else "cpu"
        self.model = init_model(self.cfg, self.cfg.load_from, device=self.device)
        self.cmap = np.array(self.model.dataset_meta.get("palette"), dtype=np.uint8)

    def __call__(self, image: ImageType) -> Union[SegDataSample, SampleList]:
        result = inference_model(self.model, image)
        result.set_data(dict(image=image))
        result = self.postprocess(result)
        return result

    def postprocess(
        self, result: Union[SegDataSample, SampleList]
    ) -> Union[SegDataSample, SampleList]:
        mask = self._get_pred_mask(result)
        # TODO: 小さい面積削除の後処理
        # TODO: binary maskにしてerode->孤立点を削除する
        result.update(PixelData(data=torch.from_numpy(mask).unsqueeze(0).to(self.device)))
        return result

    def overlay(
        self, result: SegDataSample, alpha: float = 0.3, rgb_to_bgr: bool = False
    ) -> np.ndarray:
        image = self._get_input_image(result)
        g_mask = self._get_pred_mask(result)

        _, mask = cv2.threshold(g_mask, 0, 255, cv2.THRESH_BINARY)
        _, inv_mask = cv2.threshold(g_mask, 0, 255, cv2.THRESH_BINARY_INV)

        c_bin_mask = np.dstack([mask, mask, mask])
        c_bin_inv_mask = np.dstack([inv_mask, inv_mask, inv_mask])

        fore_img = cv2.bitwise_and(image, c_bin_mask)
        back_img = cv2.bitwise_and(image, c_bin_inv_mask)

        c_mask = self.cmap[g_mask]
        blended = cv2.addWeighted(fore_img, 1 - alpha, c_mask, alpha, 0)
        overlayed = cv2.bitwise_or(back_img, blended)
        if rgb_to_bgr:
            overlayed = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
        return overlayed

    def putpalette(self, result: SegDataSample) -> Image:
        g_mask = self._get_pred_mask(result)
        mask_pil = Image.fromarray(g_mask).convert("P")
        if self.cmap is None:
            return mask_pil
        mask_pil.putpalette(self.cmap.flatten())
        return mask_pil

    def _build_config(
        self,
        config: Union[str, Path],
        checkpoint: Union[str, Path],
        cfg_options: Optional[dict] = None,
        launcher: str = "none",
        tta: bool = False,
    ) -> Config:
        # load config
        cfg = Config.fromfile(str(config))
        cfg.launcher = launcher
        if cfg_options is not None:
            cfg.merge_from_dict(cfg_options)
        # load weights
        cfg.load_from = str(checkpoint)
        # set TTA pipeline instead of test_pipeline
        cfg.tta = tta
        if tta:
            cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
            cfg.tta_model.module = cfg.model
            cfg.model = cfg.tta_model
        return cfg

    def _get_input_image(
        self, result: Union[SegDataSample, SampleList], bgr_to_rgb: bool = True
    ) -> Union[np.ndarray, Sequence[np.ndarray]]:
        if isinstance(result, SegDataSample):
            image = result.image
            return self._read_image(image, bgr_to_rgb)
        elif isinstance(result, SampleList):
            images = result.image
            return [self._read_image(image, bgr_to_rgb) for image in images]
        raise ValueError(f"Invalid type of result: {type(result)}")

    def _get_pred_mask(
        self, result: Union[SegDataSample, SampleList]
    ) -> Union[np.ndarray, Sequence[np.ndarray]]:
        if isinstance(result, SegDataSample):
            pred_mask = result.pred_sem_seg.data[0]
            return self._tensor2numpy(pred_mask)
        elif isinstance(result, SampleList):
            pred_masks = result.pred_sem_seg
            return [self._tensor2numpy(pred_mask.data[0]) for pred_mask in pred_masks]
        raise ValueError(f"Invalid type of result: {type(result)}")

    def _read_image(
        self, image: Union[np.ndarray, str, Path], bgr_to_rgb: bool = True
    ) -> np.ndarray:
        if isinstance(image, np.ndarray):
            if bgr_to_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        elif isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if bgr_to_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        raise ValueError(f"Invalid type of result: {type(image)}")

    def _tensor2numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().detach().numpy().astype(np.uint8)


if __name__ == "__main__":
    main()
