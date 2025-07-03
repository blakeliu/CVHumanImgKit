from typing import Any, List, Callable, Tuple, Union, Optional
import numpy as np
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.transforms import img_to_tensor
from abc import ABC, abstractmethod


def albu_gan_diversity(height: int, width: int):
    transformations = albu.Compose([
        albu.OneOf(transforms=[
            albu.LongestMaxSize(max_size=max((height, width)), p=1),
            albu.SmallestMaxSize(max_size=max((height, width)), p=1)], p=1),
        albu.PadIfNeeded(min_height=height, min_width=width,
                         border_mode=0, value=(0, 0, 0), p=1),
        albu.OneOf(transforms=[
            albu.RandomResizedCrop(height, width, scale=(
                0.5, 1.0), ratio=(0.75, 1.33), p=0.5),
            albu.RandomCrop(height, width, p=0.5)
        ], p=0.5),
        albu.Rotate(limit=10, interpolation=1,
                    border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.ColorJitter(0.2, 0.2, 0.2, p=0.01),
        albu.Resize(height=height, width=width,
                    interpolation=cv2.INTER_NEAREST, p=1),
    ])
    return transformations


def albu_gan_test(height: int, width: int):
    transformations = albu.Compose([
        albu.OneOf(transforms=[
            albu.LongestMaxSize(max_size=max((height, width)), p=1),
            # albu.SmallestMaxSize(max_size=max((height, width)), p=1)
            ], p=1),
        albu.PadIfNeeded(min_height=height, min_width=width,
                         border_mode=0, value=(0, 0, 0), p=1),
        albu.Resize(height=height, width=width,
                    interpolation=cv2.INTER_NEAREST, p=1),
    ])
    return transformations

class AlbuTransform(ABC):
    def __init__(self, height: int = 256, width: int = 256) -> None:
        self.height = height
        self.width = width

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    @property
    def size(self):
        return max(self.height, self.width)


class AlbuGanDiversity(AlbuTransform):
    def __init__(self, height: int = 256, width: int = 256) -> None:
        super().__init__(height, width)
        self.transform = albu_gan_diversity(height, width)

    def __call__(self, image) -> Any:
        transformed_data = self.transform(image=image)
        return transformed_data['image']
    
class AlbuGanTest(AlbuTransform):
    def __init__(self, height: int = 256, width: int = 256) -> None:
        super().__init__(height, width)
        self.transform = albu_gan_test(height, width)

    def __call__(self, image) -> Any:
        transformed_data = self.transform(image=image)
        return transformed_data['image']


def test_AlbuSegDiversity(img_path, show_count: int = 0):
    trans = AlbuGanDiversity(256, 256)
    # trans = AlbuGanTest(256, 256)
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(show_count):
        aug_img = trans(img)
        show_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("aug", show_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    img_file = "eyelid.jpg"
    test_AlbuSegDiversity(img_file, show_count=100)
