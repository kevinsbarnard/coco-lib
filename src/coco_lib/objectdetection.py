"""Object detection COCO format structures.

This module provides dataclasses for working with COCO object detection datasets,
including annotations with bounding boxes and segmentation masks.
"""

from dataclasses import dataclass
from typing import List, Tuple

from dataclasses_json import dataclass_json

from coco_lib.bases import Annotation, Category, Dataset


@dataclass_json
@dataclass
class ObjectDetectionAnnotation(Annotation):
    """Annotation for object detection task.

    Contains bounding box, segmentation mask, and metadata for a single
    object instance in an image.

    Attributes:
        id (int): Unique annotation identifier.
        image_id (int): ID of the image this annotation belongs to.
        category_id (int): ID of the object category.
        segmentation (List[List[float]]): Polygon segmentation as list of [x,y] coordinates.
        area (float): Area of the segmentation mask in pixels.
        bbox (Tuple[float, float, float, float]): Bounding box as (x, y, width, height).
        iscrowd (int): Whether this annotation represents a crowd (0 or 1).

    Example:
        >>> ann = ObjectDetectionAnnotation(
        ...     id=1,
        ...     image_id=1,
        ...     category_id=1,
        ...     segmentation=[[100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0]],
        ...     area=10000.0,
        ...     bbox=(100.0, 100.0, 100.0, 100.0),
        ...     iscrowd=0
        ... )
        >>> ann.area
        10000.0
        >>> ann.bbox[2]  # width
        100.0
    """

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
    """Category definition for object detection.

    Defines an object class with its name and hierarchical relationship.

    Attributes:
        id (int): Unique category identifier.
        name (str): Category name (e.g., "person", "car").
        supercategory (str): Parent category name in the hierarchy.

    Example:
        >>> cat = ObjectDetectionCategory(
        ...     id=1,
        ...     name="car",
        ...     supercategory="vehicle"
        ... )
        >>> cat.name
        'car'
        >>> cat.supercategory
        'vehicle'
    """

    id: int
    name: str
    supercategory: str


@dataclass_json
@dataclass
class ObjectDetectionDataset(Dataset):
    """Complete object detection dataset.

    Contains all data for a COCO object detection dataset including
    annotations and category definitions.

    Attributes:
        annotations (List[ObjectDetectionAnnotation]): List of object annotations.
        categories (List[ObjectDetectionCategory]): List of object categories.

    Example:
        >>> from coco_lib.common import Info, Image, License
        >>> info = Info(year=2023, version="1.0")
        >>> images = [Image(id=1, width=640, height=480, file_name="test.jpg")]
        >>> licenses = [License(id=1, name="CC BY 4.0")]
        >>> annotations = [ObjectDetectionAnnotation(
        ...     id=1, image_id=1, category_id=1,
        ...     segmentation=[[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]],
        ...     area=10000.0, bbox=(0.0, 0.0, 100.0, 100.0), iscrowd=0
        ... )]
        >>> categories = [ObjectDetectionCategory(id=1, name="person", supercategory="human")]
        >>> dataset = ObjectDetectionDataset(
        ...     info=info, images=images, licenses=licenses,
        ...     annotations=annotations, categories=categories
        ... )
        >>> len(dataset.annotations)
        1
    """

    annotations: List[ObjectDetectionAnnotation]
    categories: List[ObjectDetectionCategory]
