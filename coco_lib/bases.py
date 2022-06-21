from abc import ABC
from dataclasses import dataclass
from os import PathLike
from typing import List

from dataclasses_json import dataclass_json

from coco_lib.common import Info, Image, License


class Annotation(ABC):
    pass


class Category(ABC):
    pass


@dataclass_json
@dataclass
class Dataset(ABC):
    info: Info
    images: List[Image]
    licenses: List[License]
    
    def save(self, path: PathLike, **kwargs):
        with open(path, 'w') as f:
            f.write(self.to_json(**kwargs))

    @classmethod
    def load(cls, path: PathLike):
        with open(path, 'r') as f:
            return cls.from_json(f.read())