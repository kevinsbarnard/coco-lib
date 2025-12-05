"""Base classes for COCO dataset structures.

This module provides abstract base classes for annotations, categories, and datasets
that are extended by specific COCO format implementations.
"""

from typing import Type, TypeVar

from pydantic import BaseModel


SerializableT = TypeVar("SerializableT", bound="Serializable")


class Serializable(BaseModel):
    """Base class for serializable COCO objects.

    This class provides methods for serializing and deserializing objects to and from JSON format.
    """

    def to_json(self, **kwargs) -> str:
        """Serialize the object to a JSON string.

        Args:
            **kwargs: Additional keyword arguments passed to the JSON encoder.
        """
        return self.model_dump_json(exclude_none=True, **kwargs)

    @classmethod
    def from_json(cls: Type[SerializableT], json_str: str, **kwargs) -> SerializableT:
        """Deserialize the object from a JSON string.

        Args:
            json_str (str): The JSON string to deserialize.
            **kwargs: Additional keyword arguments passed to the JSON decoder.

        Returns:
            SerializableT: An instance of the class deserialized from JSON.
        """
        return cls.model_validate_json(json_str, **kwargs)


class Annotation(Serializable):
    """Base class for COCO annotations.

    This class serves as a base for all annotation types in COCO datasets,
    including object detection, keypoint detection, and panoptic segmentation.
    """

    pass


class Category(Serializable):
    """Base class for COCO categories.

    This class serves as a base for all category types in COCO datasets,
    defining the structure for object classes and their properties.
    """

    pass
