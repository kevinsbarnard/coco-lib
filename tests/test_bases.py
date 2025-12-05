"""Tests for base classes."""

from coco_lib.objectdetection import (
    ObjectDetectionAnnotation,
    ObjectDetectionCategory,
)


class TestSerializable:
    """Test suite for Serializable base class."""

    def test_serializable_to_json(self) -> None:
        """Test that Serializable can serialize to JSON."""
        annotation = ObjectDetectionAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            segmentation=[[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]],
            area=10000.0,
            bbox=(0.0, 0.0, 100.0, 100.0),
            iscrowd=0,
        )
        json_str = annotation.to_json()
        assert isinstance(json_str, str)
        assert '"id":1' in json_str

    def test_serializable_from_json(self) -> None:
        """Test that Serializable can deserialize from JSON."""
        json_str = '{"id": 1, "name": "person", "supercategory": "human"}'
        category = ObjectDetectionCategory.from_json(json_str)
        assert category.id == 1
        assert category.name == "person"
