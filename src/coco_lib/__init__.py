"""COCO dataset library.

This package provides Python dataclasses for working with COCO format datasets,
including object detection, image captioning, keypoint detection, and panoptic segmentation.

Example:
    >>> from coco_lib.common import Info, Image, License
    >>> from coco_lib.objectdetection import ObjectDetectionDataset, ObjectDetectionCategory, ObjectDetectionAnnotation
    >>> # Create a simple dataset
    >>> info = Info(year=2023, version="1.0")
    >>> images = [Image(id=1, width=640, height=480, file_name="test.jpg")]
    >>> licenses = [License(id=1, name="CC BY 4.0")]
    >>> categories = [ObjectDetectionCategory(id=1, name="person", supercategory="human")]
    >>> annotations = [ObjectDetectionAnnotation(
    ...     id=1, image_id=1, category_id=1, segmentation=[[0.0, 0.0, 10.0, 10.0]],
    ...     area=100.0, bbox=(0.0, 0.0, 10.0, 10.0), iscrowd=0
    ... )]
    >>> dataset = ObjectDetectionDataset(info, images, licenses, annotations, categories)
    >>> len(dataset.images)
    1
"""

__version__ = "0.1.5"

__all__ = [
    "bases",
    "common",
    "imagecaptioning",
    "keypointdetection",
    "objectdetection",
    "panopticsegmentation",
]
