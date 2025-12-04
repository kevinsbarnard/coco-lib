"""Keypoint detection COCO format structures.

This module provides dataclasses for working with COCO keypoint detection datasets,
extending object detection with skeletal keypoint information for pose estimation.
"""

from dataclasses import dataclass
from typing import List, Tuple

from dataclasses_json import dataclass_json

from coco_lib.bases import Dataset
from coco_lib.objectdetection import ObjectDetectionAnnotation, ObjectDetectionCategory


@dataclass_json
@dataclass
class KeypointDetectionAnnotation(ObjectDetectionAnnotation):
    """Annotation for keypoint detection task.

    Extends object detection annotation with keypoint locations for
    pose estimation and skeletal tracking.

    Attributes:
        keypoints (List[float]): Flattened list of keypoint coordinates [x1, y1, v1, x2, y2, v2, ...],
            where v is visibility flag (0=not labeled, 1=labeled but not visible, 2=labeled and visible).
        num_keypoints (int): Number of labeled keypoints (v > 0).

    Example:
        >>> ann = KeypointDetectionAnnotation(
        ...     id=1,
        ...     image_id=1,
        ...     category_id=1,
        ...     segmentation=[[100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0]],
        ...     area=10000.0,
        ...     bbox=(100.0, 100.0, 100.0, 100.0),
        ...     iscrowd=0,
        ...     keypoints=[150.0, 120.0, 2, 150.0, 160.0, 2, 130.0, 180.0, 1],
        ...     num_keypoints=3
        ... )
        >>> ann.num_keypoints
        3
        >>> len(ann.keypoints)
        9
    """

    keypoints: List[float]
    num_keypoints: int


@dataclass_json
@dataclass
class KeypointDetectionCategory(ObjectDetectionCategory):
    """Category definition for keypoint detection.

    Extends object detection category with keypoint definitions and
    skeletal structure information.

    Attributes:
        keypoints (List[str]): Names of keypoints (e.g., ["nose", "left_eye", "right_eye"]).
        skeleton (List[Tuple[int, int]]): List of keypoint pairs defining skeletal connections,
            using 1-based indexing into the keypoints list.

    Example:
        >>> cat = KeypointDetectionCategory(
        ...     id=1,
        ...     name="person",
        ...     supercategory="human",
        ...     keypoints=["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        ...     skeleton=[(1, 2), (1, 3), (2, 4), (3, 5)]
        ... )
        >>> cat.keypoints[0]
        'nose'
        >>> len(cat.skeleton)
        4
    """

    keypoints: List[str]
    skeleton: List[Tuple[int, int]]


@dataclass_json
@dataclass
class KeypointDetectionDataset(Dataset):
    """Complete keypoint detection dataset.

    Contains all data for a COCO keypoint detection dataset including
    keypoint annotations and category definitions with skeletal structure.

    Attributes:
        annotations (List[KeypointDetectionAnnotation]): List of keypoint annotations.
        categories (List[KeypointDetectionCategory]): List of categories with keypoint definitions.

    Example:
        >>> from coco_lib.common import Info, Image, License
        >>> info = Info(year=2023, version="1.0")
        >>> images = [Image(id=1, width=640, height=480, file_name="test.jpg")]
        >>> licenses = [License(id=1, name="CC BY 4.0")]
        >>> annotations = [KeypointDetectionAnnotation(
        ...     id=1, image_id=1, category_id=1,
        ...     segmentation=[[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]],
        ...     area=10000.0, bbox=(0.0, 0.0, 100.0, 100.0), iscrowd=0,
        ...     keypoints=[50.0, 30.0, 2, 50.0, 70.0, 2], num_keypoints=2
        ... )]
        >>> categories = [KeypointDetectionCategory(
        ...     id=1, name="person", supercategory="human",
        ...     keypoints=["head", "body"], skeleton=[(1, 2)]
        ... )]
        >>> dataset = KeypointDetectionDataset(
        ...     info=info, images=images, licenses=licenses,
        ...     annotations=annotations, categories=categories
        ... )
        >>> dataset.annotations[0].num_keypoints
        2
    """

    annotations: List[KeypointDetectionAnnotation]
    categories: List[KeypointDetectionCategory]
