import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class SuperviselyLaserLineDataset(Dataset):
    def __init__(
        self, supervisely_project_path: str, transforms=None, sigma=3, kernel_size=15
    ):
        self.data_path = Path(supervisely_project_path)
        self.transforms = transforms
        self.sigma = sigma
        self.kernel_size = kernel_size

        self._image_paths = []
        self._ann_paths = []
        for datadir in self.data_path.iterdir():
            if not datadir.is_dir():
                continue
            self._image_paths += sorted((datadir / "img").iterdir())
            self._ann_paths += sorted((datadir / "ann").iterdir())

        x, y = np.meshgrid(
            np.linspace(-kernel_size / 2, kernel_size / 2, kernel_size),
            np.linspace(-kernel_size / 2, kernel_size / 2, kernel_size),
        )
        self._dst = np.sqrt(x**2 + y**2)

    @profile
    def _load_ann(self, ann_file):
        """
        Parses annotation JSON and generates a target heatmap.
        """
        with open(ann_file, "r") as f:
            ann = json.load(f)
        target = np.zeros(
            (ann["size"]["height"], ann["size"]["width"]), dtype=np.float32
        )
        mesh_x, mesh_y = np.meshgrid(
            np.arange(ann["size"]["width"]),
            np.arange(ann["size"]["height"]),
        )
        for obj in ann["objects"]:
            if not obj["geometryType"] == "line":
                continue
            line_segments = zip(
                obj["points"]["exterior"][:-1], obj["points"]["exterior"][1:]
            )
            for (x0, y0), (x1, y1) in line_segments:
                if x0 > x1:
                    x0, y0, x1, y1 = x1, y1, x0, y0
                for x in range(x0, x1 + 1):
                    if x1 == x0:
                        y = y0
                    else:
                        y = y0 + (x - x0) / (x1 - x0) * (y1 - y0)  # interpolate
                    self._dst = np.sqrt((mesh_x - x) ** 2 + (mesh_y - y) ** 2)
                    # generate gaussian
                    gauss = np.exp(-((self._dst) ** 2 / (2 * self.sigma**2)))
                    # paste into target heatmap
                    target = np.maximum(target, gauss)
        return target

    def __getitem__(self, index):
        image = Image.open(self._image_paths[index])
        image = ImageOps.exif_transpose(image)
        target = self._load_ann(self._ann_paths[index])
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        raise NotImplementedError


if __name__ == "__main__":
    # some tests
    dataset = SuperviselyLaserLineDataset("data/272599_Laser line detection")
    image, target = dataset[0]
    fig, ax = plt.subplots(nrows=2)
    ax[0].imshow(image)
    ax[1].imshow(target)
    plt.show()
