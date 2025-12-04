"""Image captioning COCO format structures.

This module provides dataclasses for working with COCO image captioning datasets,
where each annotation contains a textual description of an image.
"""

from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json

from coco_lib.bases import Annotation, Dataset


@dataclass_json
@dataclass
class ImageCaptioningAnnotation(Annotation):
    """Annotation for image captioning task.

    Contains a textual caption describing the content of an image.

    Attributes:
        id (int): Unique annotation identifier.
        image_id (int): ID of the image this caption describes.
        caption (str): Textual description of the image.

    Example:
        >>> ann = ImageCaptioningAnnotation(
        ...     id=1,
        ...     image_id=1,
        ...     caption="A person riding a bicycle on a sunny day"
        ... )
        >>> ann.caption
        'A person riding a bicycle on a sunny day'
        >>> ann.image_id
        1
    """

    id: int
    image_id: int
    caption: str


@dataclass_json
@dataclass
class ImageCaptioningDataset(Dataset):
    """Complete image captioning dataset.

    Contains all data for a COCO image captioning dataset including
    caption annotations for each image.

    Attributes:
        annotations (List[ImageCaptioningAnnotation]): List of image captions.

    Example:
        >>> from coco_lib.common import Info, Image, License
        >>> info = Info(year=2023, version="1.0")
        >>> images = [Image(id=1, width=640, height=480, file_name="test.jpg")]
        >>> licenses = [License(id=1, name="CC BY 4.0")]
        >>> annotations = [ImageCaptioningAnnotation(
        ...     id=1,
        ...     image_id=1,
        ...     caption="A test image"
        ... )]
        >>> dataset = ImageCaptioningDataset(
        ...     info=info, images=images, licenses=licenses, annotations=annotations
        ... )
        >>> len(dataset.annotations)
        1
        >>> dataset.annotations[0].caption
        'A test image'
    """

    annotations: List[ImageCaptioningAnnotation]
