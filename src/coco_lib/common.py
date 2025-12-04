"""Common dataclasses for COCO datasets.

This module defines common structures used across all COCO dataset types,
including dataset metadata, image information, and license details.
"""

from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import dataclass_json, config
from marshmallow import fields


@dataclass_json
@dataclass
class Info:
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
    date_created: Optional[datetime] = field(
        default=None,
        metadata=config(
            encoder=lambda d: d.strftime("%Y/%m/%d") if d is not None else None,
            decoder=lambda s: datetime.strptime(s, "%Y/%m/%d")
            if s is not None
            else None,
            mm_field=fields.DateTime(format="%Y/%m/%d"),
        ),
    )


@dataclass_json
@dataclass
class Image:
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
    date_captured: Optional[datetime] = field(
        default=None,
        metadata=config(
            encoder=lambda d: d.strftime("%Y-%m-%d %H:%M:%S")
            if d is not None
            else None,
            decoder=lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            if s is not None
            else None,
            mm_field=fields.DateTime(format="iso"),
        ),
    )


@dataclass_json
@dataclass
class License:
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
