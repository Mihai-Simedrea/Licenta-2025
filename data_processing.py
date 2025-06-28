from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Callable
import numpy as np
from PIL import Image
from config import MAX_WORKERS
from models import BreastImageModel
import cv2
from skimage import measure


class Handler(ABC):
    @abstractmethod
    def set_next(self, handler: Handler) -> Handler:
        pass

    @abstractmethod
    def get_augmentations(self) -> List[Callable[[BreastImageModel], List[BreastImageModel]]]:
        pass


class AbstractHandler(Handler):
    _next_handler: Optional[Handler] = None

    def set_next(self, handler: Handler) -> Handler:
        self._next_handler = handler
        return self

    def get_augmentations(self) -> List[Callable[[BreastImageModel], List[BreastImageModel]]]:
        funcs = self._get_my_augmentations()
        if self._next_handler:
            funcs += self._next_handler.get_augmentations()
        return funcs

    @abstractmethod
    def _get_my_augmentations(self) -> List[Callable[[BreastImageModel], List[BreastImageModel]]]:
        pass


class RotationHandler(AbstractHandler):
    def __init__(self, allowed_rotations: list[int] = [0, 90, 180, 270]):
        self.allowed_rotations = allowed_rotations

    def _get_my_augmentations(self) -> List[Callable[[BreastImageModel], List[BreastImageModel]]]:
        def rotate(model: BreastImageModel) -> List[BreastImageModel]:
            results = []
            try:
                image = Image.fromarray(model.content)
                for angle in self.allowed_rotations:
                    rotated = image.rotate(angle, expand=True)
                    rotated_np = np.array(rotated)
                    rotated_name = f"{model.name.rsplit('.', 1)[0]}_rot{angle}.jpg"
                    results.append(BreastImageModel(rotated_name, rotated_np, rotated.size, model.label, model.bbox))
            except Exception as e:
                print(f"[ERROR] Rotation failed for {model.name}: {e}")
                results.append(model)
            return results
        return [rotate]


class PaddingHandler(AbstractHandler):
    def __init__(self, padded_width: int, padded_height: int):
        self.padded_width = padded_width
        self.padded_height = padded_height

    def _get_my_augmentations(self) -> List[Callable[[BreastImageModel], List[BreastImageModel]]]:
        def pad(model: BreastImageModel) -> List[BreastImageModel]:
            try:
                image = Image.fromarray(model.content)
                padded_image = Image.new("RGB", (self.padded_width, self.padded_height), (0, 0, 0))
                padded_image.paste(image, (0, 0))
                padded_np = np.array(padded_image)
                padded_name = f"{model.name.rsplit('.', 1)[0]}_padded.jpg"
                return [BreastImageModel(padded_name, padded_np, (self.padded_width, self.padded_height), model.label, model.bbox)]
            except Exception as e:
                print(f"[ERROR] Padding failed for {model.name}: {e}")
                return [model]
        return [pad]


class ResizeHandler(AbstractHandler):
    def __init__(self, target_width: int = 224, target_height: int = 224):
        self.target_width = target_width
        self.target_height = target_height

    def _get_my_augmentations(self) -> List[Callable[[BreastImageModel], List[BreastImageModel]]]:
        def resize(model: BreastImageModel) -> List[BreastImageModel]:
            try:
                orig_height, orig_width = model.content.shape[:2]
                image = Image.fromarray(model.content)
                resized_image = image.resize((self.target_width, self.target_height), Image.Resampling.LANCZOS)
                resized_np = np.array(resized_image)
                resized_name = f"{model.name.rsplit('.', 1)[0]}_resized.jpg"

                bbox = model.bbox
                if bbox is not None:
                    x, y, w, h = bbox
                    scale_x = self.target_width / orig_width
                    scale_y = self.target_height / orig_height
                    new_bbox = (
                        int(x * scale_x),
                        int(y * scale_y),
                        int(w * scale_x),
                        int(h * scale_y)
                    )
                else:
                    new_bbox = None

                return [BreastImageModel(resized_name, resized_np, (self.target_width, self.target_height), model.label, new_bbox)]
            except Exception as e:
                print(f"[ERROR] Resize failed for {model.name}: {e}")
                return [model]
        return [resize]


class NormalizeHandler(AbstractHandler):
    def _get_my_augmentations(self) -> List[Callable[[BreastImageModel], List[BreastImageModel]]]:
        def normalize(model: BreastImageModel) -> List[BreastImageModel]:
            try:
                normalized_array = model.content.astype(np.float32) / 255.0
                normalized_name = f"{model.name.rsplit('.', 1)[0]}_normalized.jpg"
                return [BreastImageModel(normalized_name, normalized_array, model.size, model.label, model.bbox)]
            except Exception as e:
                print(f"[ERROR] Normalization failed for {model.name}: {e}")
                return [model]
        return [normalize]


class BinaryMaskHandler(AbstractHandler):
    def _get_my_augmentations(self) -> List[Callable[[BreastImageModel], List[BreastImageModel]]]:
        def preprocess(model: BreastImageModel) -> List[BreastImageModel]:
            try:
                gray = model.content
                if len(gray.shape) == 3:
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                labels = measure.label(binary)
                props = measure.regionprops(labels)
                if not props:
                    raise ValueError("No regions found in binary image.")

                largest_region = max(props, key=lambda x: x.area)
                mask = np.zeros_like(binary, dtype=np.uint8)
                for y, x in largest_region.coords:
                    mask[y, x] = 255

                soft_mask = cv2.GaussianBlur(mask, (21, 21), 0)
                soft_mask_float = soft_mask.astype(np.float32) / 255.0
                gray_float = gray.astype(np.float32)

                masked_soft = (gray_float * soft_mask_float).astype(np.uint8)

                processed_name = f"{model.name.rsplit('.', 1)[0]}_masked.jpg"
                return [BreastImageModel(processed_name, masked_soft, model.size, model.label, model.bbox)]

            except Exception as e:
                print(f"[ERROR] Preprocessing failed for {model.name}: {e}")
                return [model]

        return [preprocess]
