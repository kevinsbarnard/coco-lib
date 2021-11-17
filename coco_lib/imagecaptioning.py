from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json

from .bases import Annotation, Dataset
from .common import Info, Image, License


@dataclass_json
@dataclass
class ImageCaptioningAnnotation(Annotation):
    id: int
    image_id: int
    caption: str


@dataclass_json
@dataclass
class ImageCaptioningDataset(Dataset):
    info: Info
    images: List[Image]
    licenses: List[License]
    annotations: List[ImageCaptioningAnnotation]