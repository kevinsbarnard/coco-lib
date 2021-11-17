from dataclasses import dataclass
from typing import List, Tuple

from dataclasses_json import dataclass_json

from .bases import Annotation, Category, Dataset
from .common import Info, Image, License


@dataclass_json
@dataclass
class ObjectDetectionAnnotation(Annotation):
    id: int
    image_id: int
    category_id: int
    segmentation: List[List[float]]
    area: float
    bbox: Tuple[float, float, float, float]
    iscrowd: int


@dataclass_json
@dataclass
class ObjectDetectionCategory(Category):
    id: int
    name: str
    supercategory: str


@dataclass_json
@dataclass
class ObjectDetectionDataset(Dataset):
    info: Info
    images: List[Image]
    licenses: List[License]
    annotations: List[ObjectDetectionAnnotation]
    categories: List[ObjectDetectionCategory]
