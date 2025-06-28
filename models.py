from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class BreastImageModel:
    name: str
    content: any
    size: Tuple[int, int]
    label: int
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    