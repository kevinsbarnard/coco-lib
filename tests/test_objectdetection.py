"""Tests for object detection module."""

import json
from pathlib import Path

from coco_lib.common import Image, Info, License
from coco_lib.objectdetection import (
    ObjectDetectionAnnotation,
    ObjectDetectionCategory,
    ObjectDetectionDataset,
)


class TestObjectDetectionAnnotation:
    """Test suite for ObjectDetectionAnnotation."""

    def test_annotation_initialization(self) -> None:
        """Test creating an ObjectDetectionAnnotation."""
        ann = ObjectDetectionAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            segmentation=[[100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0]],
            area=10000.0,
            bbox=(100.0, 100.0, 100.0, 100.0),
            iscrowd=0,
        )

        assert ann.id == 1
        assert ann.image_id == 1
        assert ann.category_id == 1
        assert len(ann.segmentation) == 1
        assert len(ann.segmentation[0]) == 8
        assert ann.area == 10000.0
        assert ann.bbox == (100.0, 100.0, 100.0, 100.0)
        assert ann.iscrowd == 0

    def test_annotation_bbox_format(self) -> None:
        """Test bounding box is in (x, y, width, height) format."""
        ann = ObjectDetectionAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            segmentation=[[0.0, 0.0, 50.0, 0.0, 50.0, 50.0, 0.0, 50.0]],
            area=2500.0,
            bbox=(10.0, 20.0, 30.0, 40.0),
            iscrowd=0,
        )

        x, y, width, height = ann.bbox
        assert x == 10.0
        assert y == 20.0
        assert width == 30.0
        assert height == 40.0

    def test_annotation_multiple_segmentations(self) -> None:
        """Test annotation with multiple segmentation polygons."""
        ann = ObjectDetectionAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            segmentation=[
                [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
                [20.0, 20.0, 30.0, 20.0, 30.0, 30.0, 20.0, 30.0],
            ],
            area=200.0,
            bbox=(0.0, 0.0, 30.0, 30.0),
            iscrowd=0,
        )

        assert len(ann.segmentation) == 2
        assert len(ann.segmentation[0]) == 8
        assert len(ann.segmentation[1]) == 8

    def test_annotation_iscrowd_flag(self) -> None:
        """Test iscrowd flag values."""
        ann1 = ObjectDetectionAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            segmentation=[[0.0, 0.0, 10.0, 10.0]],
            area=100.0,
            bbox=(0.0, 0.0, 10.0, 10.0),
            iscrowd=0,
        )
        ann2 = ObjectDetectionAnnotation(
            id=2,
            image_id=1,
            category_id=1,
            segmentation=[[0.0, 0.0, 10.0, 10.0]],
            area=100.0,
            bbox=(0.0, 0.0, 10.0, 10.0),
            iscrowd=1,
        )

        assert ann1.iscrowd == 0
        assert ann2.iscrowd == 1

    def test_annotation_json_serialization(self) -> None:
        """Test JSON serialization of annotation."""
        ann = ObjectDetectionAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            segmentation=[[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]],
            area=100.0,
            bbox=(0.0, 0.0, 10.0, 10.0),
            iscrowd=0,
        )

        json_str = ann.to_json()
        assert '"id": 1' in json_str
        assert '"image_id": 1' in json_str
        assert '"category_id": 1' in json_str

    def test_annotation_json_deserialization(self) -> None:
        """Test JSON deserialization of annotation."""
        json_str = """{
            "id": 1, "image_id": 1, "category_id": 1,
            "segmentation": [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]],
            "area": 100.0, "bbox": [0.0, 0.0, 10.0, 10.0], "iscrowd": 0
        }"""
        ann = ObjectDetectionAnnotation.from_json(json_str)

        assert ann.id == 1
        assert ann.area == 100.0


class TestObjectDetectionCategory:
    """Test suite for ObjectDetectionCategory."""

    def test_category_initialization(self) -> None:
        """Test creating an ObjectDetectionCategory."""
        cat = ObjectDetectionCategory(
            id=1,
            name="person",
            supercategory="human",
        )

        assert cat.id == 1
        assert cat.name == "person"
        assert cat.supercategory == "human"

    def test_category_json_serialization(self) -> None:
        """Test JSON serialization of category."""
        cat = ObjectDetectionCategory(id=1, name="car", supercategory="vehicle")
        json_str = cat.to_json()

        assert '"id": 1' in json_str
        assert '"name": "car"' in json_str
        assert '"supercategory": "vehicle"' in json_str

    def test_category_json_deserialization(self) -> None:
        """Test JSON deserialization of category."""
        json_str = '{"id": 1, "name": "dog", "supercategory": "animal"}'
        cat = ObjectDetectionCategory.from_json(json_str)

        assert cat.id == 1
        assert cat.name == "dog"
        assert cat.supercategory == "animal"

    def test_multiple_categories(self) -> None:
        """Test working with multiple categories."""
        categories = [
            ObjectDetectionCategory(id=1, name="person", supercategory="human"),
            ObjectDetectionCategory(id=2, name="car", supercategory="vehicle"),
            ObjectDetectionCategory(id=3, name="dog", supercategory="animal"),
        ]

        assert len(categories) == 3
        assert categories[0].name == "person"
        assert categories[1].name == "car"
        assert categories[2].name == "dog"


class TestObjectDetectionDataset:
    """Test suite for ObjectDetectionDataset."""

    def test_dataset_initialization(
        self,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test creating an ObjectDetectionDataset."""
        annotations = [
            ObjectDetectionAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                segmentation=[[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]],
                area=10000.0,
                bbox=(0.0, 0.0, 100.0, 100.0),
                iscrowd=0,
            )
        ]
        categories = [
            ObjectDetectionCategory(id=1, name="person", supercategory="human")
        ]

        dataset = ObjectDetectionDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
            categories=categories,
        )

        assert dataset.info == sample_info
        assert len(dataset.images) == 3
        assert len(dataset.annotations) == 1
        assert len(dataset.categories) == 1

    def test_dataset_multiple_annotations(
        self,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test dataset with multiple annotations."""
        annotations = [
            ObjectDetectionAnnotation(
                id=i,
                image_id=1,
                category_id=1,
                segmentation=[[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]],
                area=100.0,
                bbox=(0.0, 0.0, 10.0, 10.0),
                iscrowd=0,
            )
            for i in range(1, 6)
        ]
        categories = [
            ObjectDetectionCategory(id=1, name="person", supercategory="human")
        ]

        dataset = ObjectDetectionDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
            categories=categories,
        )

        assert len(dataset.annotations) == 5
        assert all(ann.category_id == 1 for ann in dataset.annotations)

    def test_dataset_save_load(
        self,
        temp_dir: Path,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test saving and loading a dataset."""
        annotations = [
            ObjectDetectionAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                segmentation=[[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]],
                area=10000.0,
                bbox=(0.0, 0.0, 100.0, 100.0),
                iscrowd=0,
            )
        ]
        categories = [
            ObjectDetectionCategory(id=1, name="person", supercategory="human")
        ]

        dataset = ObjectDetectionDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
            categories=categories,
        )

        # Save and load
        file_path = temp_dir / "object_detection.json"
        dataset.save(file_path)
        loaded = ObjectDetectionDataset.load(file_path)

        assert len(loaded.annotations) == len(dataset.annotations)
        assert len(loaded.categories) == len(dataset.categories)
        assert loaded.annotations[0].id == dataset.annotations[0].id

    def test_dataset_json_structure(
        self,
        temp_dir: Path,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test the JSON structure of saved dataset."""
        annotations = [
            ObjectDetectionAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                segmentation=[[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]],
                area=10000.0,
                bbox=(0.0, 0.0, 100.0, 100.0),
                iscrowd=0,
            )
        ]
        categories = [
            ObjectDetectionCategory(id=1, name="person", supercategory="human")
        ]

        dataset = ObjectDetectionDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
            categories=categories,
        )

        file_path = temp_dir / "structure_test.json"
        dataset.save(file_path)

        with open(file_path, "r") as f:
            data = json.load(f)

        assert "info" in data
        assert "images" in data
        assert "licenses" in data
        assert "annotations" in data
        assert "categories" in data
        assert isinstance(data["annotations"], list)
        assert isinstance(data["categories"], list)
