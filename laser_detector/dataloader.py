import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import tv_tensors


class SuperviselyLaserLineDataset(Dataset):
    def __init__(
        self,
        supervisely_project_path: str,
        transforms=None,
        sigma: float = 1,
        target_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            supervisely_project_path: Path to Supervisely project directory.
            transforms: Torchvision transforms.
            sigma: Sigma of heatmap gaussians.
            target_shape: Desired output heatmap shape (height, width). If None, the image size is used.
        """
        self.data_path = Path(supervisely_project_path)
        self.transforms = transforms
        self.sigma = sigma
        self.target_shape = target_shape

        self._image_paths = []
        self._ann_paths = []
        for datadir in self.data_path.iterdir():
            if not datadir.is_dir():
                continue
            self._image_paths += sorted((datadir / "img").iterdir())
            self._ann_paths += sorted((datadir / "ann").iterdir())

    def _load_ann(self, ann_file: str) -> np.ndarray:
        """
        Parses annotation JSON and generates a target heatmap.
        """
        with open(ann_file, "r") as f:
            ann = json.load(f)
        # generate gaussian distribution around line segments
        if self.target_shape is not None:
            mesh_x, mesh_y = np.meshgrid(
                np.arange(self.target_shape[1]),
                np.arange(self.target_shape[0]),
            )
            dst = np.zeros(
                (self.target_shape[0], self.target_shape[1]), dtype=np.float32
            )
            x_scale = self.target_shape[1] / ann["size"]["width"]
            y_scale = self.target_shape[0] / ann["size"]["height"]
        else:
            mesh_x, mesh_y = np.meshgrid(
                np.arange(ann["size"]["width"]),
                np.arange(ann["size"]["height"]),
            )
            dst = np.zeros(
                (ann["size"]["height"], ann["size"]["width"]), dtype=np.float32
            )
            x_scale = 1
            y_scale = 1
        dst[:] = np.Infinity
        y_range = np.arange(self.target_shape[0])
        for obj in ann["objects"]:
            if not obj["geometryType"] == "line":
                continue
            line_segments = zip(
                obj["points"]["exterior"][:-1], obj["points"]["exterior"][1:]
            )
            # calculate distance for each pixel to the nearest line segment
            for (x0, y0), (x1, y1) in line_segments:
                # ensure X values are always increasing
                if x0 > x1:
                    x0, y0, x1, y1 = x1, y1, x0, y0

                x0 = x0 * x_scale
                y0 = y0 * y_scale
                x1 = x1 * x_scale
                y1 = y1 * y_scale

                x0 = round(x0)
                x1 = round(x1)
                line_y_points = np.linspace(y0, y1, num=x1 - x0)
                dy = abs(
                    np.repeat(y_range[:, np.newaxis], x1 - x0, axis=1) - line_y_points
                )
                dst[:, x0:x1] = np.minimum(dst[:, x0:x1], dy)

                # px = x1 - x0
                # py = y1 - y0
                # norm = px**2 + py**2
                # u = ((mesh_x - x0) * px + (mesh_y - y0) * py) / norm
                # u = np.clip(u, 0, 1)
                # x = x0 + u * px
                # y = y0 + u * py
                # dx = x - mesh_x
                # dy = y - mesh_y
                # segment_dst = 0.5 * (dx**2 + dy**2)
                # dst = np.minimum(dst, segment_dst)

        target = np.exp(-((dst) ** 2 / (2 * self.sigma**2)))  # gaussian generation
        # target = tv_tensors.Mask(target)
        target = torch.Tensor(target)
        return target

    def __getitem__(self, index):
        image = Image.open(self._image_paths[index])
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = tv_tensors.Image(image)
        target = self._load_ann(self._ann_paths[index])
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self._image_paths)


if __name__ == "__main__":
    # some tests
    dataset = SuperviselyLaserLineDataset(
        "data/272599_Laser line detection", target_shape=(180, 320)
    )
    image, target = dataset[-1]
    # target = target.numpy()
    fig, ax = plt.subplots(nrows=2)
    ax[0].imshow(image)
    ax[1].imshow(target)
    plt.show()
