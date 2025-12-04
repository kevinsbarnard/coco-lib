"""Base classes for COCO dataset structures.

This module provides abstract base classes for annotations, categories, and datasets
that are extended by specific COCO format implementations.
"""

from abc import ABC
from dataclasses import dataclass
from os import PathLike
from typing import List, Type, TypeVar

from dataclasses_json import dataclass_json

from coco_lib.common import Info, Image, License


class Annotation(ABC):
    """Abstract base class for COCO annotations.

    This class serves as a base for all annotation types in COCO datasets,
    including object detection, keypoint detection, and panoptic segmentation.
    """

    pass


class Category(ABC):
    """Abstract base class for COCO categories.

    This class serves as a base for all category types in COCO datasets,
    defining the structure for object classes and their properties.
    """

    pass


DatasetT = TypeVar("DatasetT", bound="Dataset")


@dataclass_json
@dataclass
class Dataset(ABC):
    """Abstract base class for COCO datasets.

    This class provides the common structure and methods for all COCO dataset types,
    including metadata, images, and licenses.

    Attributes:
        info (Info): Dataset metadata including version, description, and creation date.
        images (List[Image]): List of images in the dataset.
        licenses (List[License]): List of image licenses.

    Example:
        >>> from coco_lib.objectdetection import ObjectDetectionDataset
        >>> from coco_lib.common import Info, Image, License
        >>> from coco_lib.objectdetection import ObjectDetectionAnnotation, ObjectDetectionCategory
        >>> info = Info(year=2023, version="1.0", description="Test dataset")
        >>> images = [Image(id=1, width=640, height=480, file_name="test.jpg")]
        >>> licenses = [License(id=1, name="CC BY 4.0")]
        >>> annotations = [ObjectDetectionAnnotation(
        ...     id=1, image_id=1, category_id=1,
        ...     segmentation=[[0, 0, 100, 0, 100, 100, 0, 100]],
        ...     area=10000.0, bbox=(10.0, 10.0, 100.0, 100.0), iscrowd=0
        ... )]
        >>> categories = [ObjectDetectionCategory(id=1, name="person", supercategory="human")]
        >>> dataset = ObjectDetectionDataset(info=info, images=images, licenses=licenses,
        ...                                   annotations=annotations, categories=categories)
        >>> len(dataset.images)
        1
    """

    info: Info
    images: List[Image]
    licenses: List[License]

    def save(self, path: PathLike, **kwargs) -> None:
        """Save the dataset to a JSON file.

        Args:
            path (PathLike): Path to save the dataset JSON file.
            **kwargs: Additional keyword arguments passed to the JSON encoder.

        Example:
            >>> import tempfile
            >>> from coco_lib.objectdetection import ObjectDetectionDataset
            >>> from coco_lib.common import Info, Image, License
            >>> from coco_lib.objectdetection import ObjectDetectionAnnotation, ObjectDetectionCategory
            >>> info = Info(year=2023)
            >>> images = [Image(id=1, width=640, height=480, file_name="test.jpg")]
            >>> licenses = [License(id=1, name="CC BY 4.0")]
            >>> annotations = [ObjectDetectionAnnotation(
            ...     id=1, image_id=1, category_id=1, segmentation=[[0.0, 0.0, 10.0, 0.0]],
            ...     area=100.0, bbox=(0.0, 0.0, 10.0, 10.0), iscrowd=0
            ... )]
            >>> categories = [ObjectDetectionCategory(id=1, name="object", supercategory="thing")]
            >>> dataset = ObjectDetectionDataset(info, images, licenses, annotations, categories)
            >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            ...     dataset.save(f.name)
        """
        with open(path, "w") as f:
            f.write(self.to_json(**kwargs))

    @classmethod
    def load(cls: Type[DatasetT], path: PathLike, **kwargs) -> DatasetT:
        """Load a dataset from a JSON file.

        Args:
            path (PathLike): Path to the dataset JSON file.
            **kwargs: Additional keyword arguments passed to the JSON decoder.

        Returns:
            DatasetT: An instance of the dataset class loaded from the file.

        Example:
            >>> import tempfile
            >>> import json
            >>> from coco_lib.objectdetection import ObjectDetectionDataset
            >>> # Create a temporary JSON file
            >>> data = {
            ...     "info": {"year": 2023},
            ...     "images": [{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg"}],
            ...     "licenses": [{"id": 1, "name": "CC BY 4.0"}],
            ...     "annotations": [{
            ...         "id": 1, "image_id": 1, "category_id": 1,
            ...         "segmentation": [[0.0, 0.0, 10.0, 0.0]], "area": 100.0,
            ...         "bbox": [0.0, 0.0, 10.0, 10.0], "iscrowd": 0
            ...     }],
            ...     "categories": [{"id": 1, "name": "object", "supercategory": "thing"}]
            ... }
            >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            ...     json.dump(data, f)
            ...     temp_path = f.name
            >>> dataset = ObjectDetectionDataset.load(temp_path)
            >>> len(dataset.images)
            1
        """
        with open(path, "r") as f:
            return cls.from_json(f.read(), **kwargs)
