from dataclasses import dataclass
from typing import List, Tuple

from dataclasses_json import dataclass_json

from .bases import Dataset
from .common import Info, Image, License
from .objectdetection import ObjectDetectionAnnotation, ObjectDetectionCategory


@dataclass_json
@dataclass
class KeypointDetectionAnnotation(ObjectDetectionAnnotation):
    keypoints: List[float]
    num_keypoints: int


@dataclass_json
@dataclass
class KeypointDetectionCategory(ObjectDetectionCategory):
    keypoints: List[str]
    skeleton: List[Tuple[int, int]]


@dataclass_json
@dataclass
class KeypointDetectionDataset(Dataset):
    annotations: List[KeypointDetectionAnnotation]
    categories: List[KeypointDetectionCategory]
