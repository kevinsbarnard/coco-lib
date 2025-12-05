"""Common dataclasses for COCO datasets.

This module defines common structures used across all COCO dataset types,
including dataset metadata, image information, and license details.
"""

import warnings
from os import PathLike
from datetime import datetime
from typing import List, Optional, Type, TypeVar, Union

from dateparser import parse as parse_datetime_string
from pydantic import field_validator, field_serializer

from coco_lib.bases import Serializable


def parse_datetime(date_string: Optional[str]) -> Optional[datetime]:
    """Parse a datetime string flexibly using dateparser.

    Args:
        date_string (Optional[str]): A string representing a date/time, or None.

    Returns:
        Optional[datetime]: A datetime object if parsing succeeds, None otherwise.

    Warnings:
        Emits a warning if the date_string is provided but cannot be parsed.
    """
    if date_string is None:
        return None

    try:
        # Suppress deprecation warnings from dateparser's internal implementation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result = parse_datetime_string(date_string)

        if result is None:
            warnings.warn(
                f"Failed to parse datetime string: {date_string!r}. Returning None.",
                UserWarning,
                stacklevel=3,
            )
        return result
    except Exception as e:
        warnings.warn(
            f"Error parsing datetime string {date_string!r}: {e}. Returning None.",
            UserWarning,
            stacklevel=3,
        )
        return None


class Info(Serializable):
    """Dataset metadata information.

    Contains high-level information about the dataset including versioning,
    description, and creation details.

    Attributes:
        year (Optional[int]): Year the dataset was created.
        version (Optional[str]): Dataset version string.
        description (Optional[str]): Textual description of the dataset.
        contributor (Optional[str]): Name of the dataset contributor.
        url (Optional[str]): URL where the dataset can be accessed.
        date_created (Optional[datetime]): Date when the dataset was created.

    Example:
        >>> from datetime import datetime
        >>> info = Info(
        ...     year=2023,
        ...     version="1.0",
        ...     description="Example COCO dataset",
        ...     contributor="Research Team",
        ...     url="https://example.com/dataset",
        ...     date_created=datetime(2023, 1, 15)
        ... )
        >>> info.year
        2023
        >>> info.version
        '1.0'
    """

    year: Optional[int] = None
    version: Optional[str] = None
    description: Optional[str] = None
    contributor: Optional[str] = None
    url: Optional[str] = None
    date_created: Optional[datetime] = None

    @field_validator("date_created", mode="before")
    @classmethod
    def parse_date_created(
        cls, v: Optional[Union[datetime, str]]
    ) -> Optional[datetime]:
        """Validator to parse date_created from string to datetime.

        Args:
            v (Optional[Union[datetime, str]]): The input datetime or string.

        Returns:
            Optional[datetime]: The parsed datetime object or None.
        """
        if isinstance(v, datetime):
            return v
        return parse_datetime(v)

    @field_serializer("date_created", mode="plain")
    def serialize_date_created(self, value: Optional[datetime]) -> Optional[str]:
        """Serializer to convert date_created datetime to string.

        Args:
            value (Optional[datetime]): The datetime object.

        Returns:
            Optional[str]: The formatted date string or None.
        """
        if value is None:
            return None
        return value.strftime("%Y/%m/%d")


class Image(Serializable):
    """Image metadata in a COCO dataset.

    Contains information about a single image including dimensions,
    filename, and optional URLs and capture time.

    Attributes:
        id (int): Unique image identifier.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        file_name (str): Image filename.
        license (Optional[int]): License ID associated with the image.
        flickr_url (Optional[str]): Flickr URL if the image is from Flickr.
        coco_url (Optional[str]): COCO dataset URL for the image.
        date_captured (Optional[datetime]): Date and time when the image was captured.

    Example:
        >>> from datetime import datetime
        >>> img = Image(
        ...     id=1,
        ...     width=1920,
        ...     height=1080,
        ...     file_name="example.jpg",
        ...     license=1,
        ...     date_captured=datetime(2023, 6, 15, 14, 30, 0)
        ... )
        >>> img.width
        1920
        >>> img.file_name
        'example.jpg'
    """

    id: int
    width: int
    height: int
    file_name: str
    license: Optional[int] = None
    flickr_url: Optional[str] = None
    coco_url: Optional[str] = None
    date_captured: Optional[datetime] = None

    @field_validator("date_captured", mode="before")
    @classmethod
    def parse_date_captured(
        cls, v: Optional[Union[datetime, str]]
    ) -> Optional[datetime]:
        """Validator to parse date_captured from string to datetime.

        Args:
            v (Optional[Union[datetime, str]]): The input datetime or string.

        Returns:
            Optional[datetime]: The parsed datetime object or None.
        """
        if isinstance(v, datetime):
            return v
        return parse_datetime(v)

    @field_serializer("date_captured", mode="plain")
    def serialize_date_captured(self, value: Optional[datetime]) -> Optional[str]:
        """Serializer to convert date_captured datetime to string.

        Args:
            value (Optional[datetime]): The datetime object.

        Returns:
            Optional[str]: The formatted date string or None.
        """
        if value is None:
            return None
        return value.strftime("%Y-%m-%d %H:%M:%S")


class License(Serializable):
    """Image license information.

    Represents a license under which an image is distributed.

    Attributes:
        id (int): Unique license identifier.
        name (str): Name of the license (e.g., "CC BY 4.0").
        url (Optional[str]): URL with full license text.

    Example:
        >>> license = License(
        ...     id=1,
        ...     name="Creative Commons Attribution 4.0",
        ...     url="https://creativecommons.org/licenses/by/4.0/"
        ... )
        >>> license.name
        'Creative Commons Attribution 4.0'
    """

    id: int
    name: str
    url: Optional[str] = None


DatasetT = TypeVar("DatasetT", bound="Dataset")


class Dataset(Serializable):
    """Base class for COCO datasets.

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

    info: Info = Info()
    images: List[Image] = []
    licenses: List[License] = []

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
            >>> dataset = ObjectDetectionDataset(info=info, images=images, licenses=licenses,
            ...                                   annotations=annotations, categories=categories)
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
