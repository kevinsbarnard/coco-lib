"""Panoptic segmentation COCO format structures.

This module provides dataclasses for working with COCO panoptic segmentation datasets,
which combine instance segmentation (things) and semantic segmentation (stuff).
"""

from dataclasses import dataclass
from typing import List, Tuple

from dataclasses_json import dataclass_json

from coco_lib.bases import Annotation, Category, Dataset


@dataclass_json
@dataclass
class SegmentInfo:
    """Information about a single segment in panoptic segmentation.

    Represents one segmented region in a panoptic segmentation mask,
    which can be either a thing (countable object) or stuff (uncountable region).

    Attributes:
        id (int): Unique segment identifier within the annotation.
        category_id (int): ID of the category this segment belongs to.
        area (float): Area of the segment in pixels.
        bbox (Tuple[float, float, float, float]): Bounding box as (x, y, width, height).
        iscrowd (int): Whether this segment represents a crowd (0 or 1).

    Example:
        >>> segment = SegmentInfo(
        ...     id=1,
        ...     category_id=1,
        ...     area=5000.0,
        ...     bbox=(50.0, 50.0, 100.0, 50.0),
        ...     iscrowd=0
        ... )
        >>> segment.area
        5000.0
    """

    id: int
    category_id: int
    area: float
    bbox: Tuple[float, float, float, float]
    iscrowd: int


@dataclass_json
@dataclass
class PanopticSegmentationAnnotation(Annotation):
    """Annotation for panoptic segmentation task.

    Contains reference to a segmentation mask file and metadata about
    all segments within that mask.

    Attributes:
        image_id (int): ID of the image this annotation belongs to.
        file_name (str): Filename of the PNG segmentation mask.
        segments_info (List[SegmentInfo]): List of segments in the mask.

    Example:
        >>> ann = PanopticSegmentationAnnotation(
        ...     image_id=1,
        ...     file_name="mask_001.png",
        ...     segments_info=[
        ...         SegmentInfo(id=1, category_id=1, area=5000.0,
        ...                     bbox=(50.0, 50.0, 100.0, 50.0), iscrowd=0),
        ...         SegmentInfo(id=2, category_id=2, area=3000.0,
        ...                     bbox=(200.0, 200.0, 50.0, 60.0), iscrowd=0)
        ...     ]
        ... )
        >>> len(ann.segments_info)
        2
        >>> ann.file_name
        'mask_001.png'
    """

    image_id: int
    file_name: str
    segments_info: List[SegmentInfo]


@dataclass_json
@dataclass
class PanopticSegmentationCategory(Category):
    """Category definition for panoptic segmentation.

    Defines a category with distinction between things (countable objects)
    and stuff (uncountable regions), plus visualization color.

    Attributes:
        id (int): Unique category identifier.
        name (str): Category name (e.g., "person", "sky", "grass").
        supercategory (str): Parent category name in the hierarchy.
        isthing (int): Whether this is a thing (1) or stuff (0).
        color (Tuple[int, int, int]): RGB color for visualization.

    Example:
        >>> cat = PanopticSegmentationCategory(
        ...     id=1,
        ...     name="person",
        ...     supercategory="human",
        ...     isthing=1,
        ...     color=(220, 20, 60)
        ... )
        >>> cat.isthing
        1
        >>> cat.color
        (220, 20, 60)
    """

    id: int
    name: str
    supercategory: str
    isthing: int
    color: Tuple[int, int, int]


@dataclass_json
@dataclass
class PanopticSegmentationDataset(Dataset):
    """Complete panoptic segmentation dataset.

    Contains all data for a COCO panoptic segmentation dataset including
    annotations referencing mask files and category definitions.

    Attributes:
        annotations (List[PanopticSegmentationAnnotation]): List of panoptic annotations.
        categories (List[PanopticSegmentationCategory]): List of categories with thing/stuff info.

    Example:
        >>> from coco_lib.common import Info, Image, License
        >>> info = Info(year=2023, version="1.0")
        >>> images = [Image(id=1, width=640, height=480, file_name="test.jpg")]
        >>> licenses = [License(id=1, name="CC BY 4.0")]
        >>> annotations = [PanopticSegmentationAnnotation(
        ...     image_id=1,
        ...     file_name="mask_001.png",
        ...     segments_info=[
        ...         SegmentInfo(id=1, category_id=1, area=5000.0,
        ...                     bbox=(50.0, 50.0, 100.0, 50.0), iscrowd=0)
        ...     ]
        ... )]
        >>> categories = [PanopticSegmentationCategory(
        ...     id=1, name="person", supercategory="human",
        ...     isthing=1, color=(220, 20, 60)
        ... )]
        >>> dataset = PanopticSegmentationDataset(
        ...     info=info, images=images, licenses=licenses,
        ...     annotations=annotations, categories=categories
        ... )
        >>> dataset.categories[0].isthing
        1
    """

    annotations: List[PanopticSegmentationAnnotation]
    categories: List[PanopticSegmentationCategory]
