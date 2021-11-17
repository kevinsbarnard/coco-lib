from abc import ABC
from dataclasses import dataclass
from os import PathLike

from dataclasses_json import dataclass_json


class Annotation(ABC):
    pass


class Category(ABC):
    pass


@dataclass_json
@dataclass
class Dataset(ABC):
    def save(self, path: PathLike, **kwargs):
        with open(path, 'w') as f:
            f.write(self.to_json(**kwargs))

    @classmethod
    def load(cls, path: PathLike):
        with open(path, 'r') as f:
            return cls.from_json(f.read())