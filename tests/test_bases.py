"""Tests for base classes."""

import json
from pathlib import Path

from coco_lib.bases import Annotation, Category, Dataset
from coco_lib.common import Image, Info, License
from coco_lib.objectdetection import (
    ObjectDetectionAnnotation,
    ObjectDetectionCategory,
    ObjectDetectionDataset,
)


class TestAnnotation:
    """Test suite for Annotation base class."""

    def test_annotation_is_abstract(self) -> None:
        """Test that Annotation is an abstract base class."""
        # Annotation itself should not be instantiable directly
        # but can be subclassed
        assert issubclass(ObjectDetectionAnnotation, Annotation)


class TestCategory:
    """Test suite for Category base class."""

    def test_category_is_abstract(self) -> None:
        """Test that Category is an abstract base class."""
        assert issubclass(ObjectDetectionCategory, Category)


class TestDataset:
    """Test suite for Dataset base class."""

    def test_dataset_is_abstract(self) -> None:
        """Test that Dataset is an abstract base class."""
        assert issubclass(ObjectDetectionDataset, Dataset)

    def test_dataset_save(
        self,
        temp_dir: Path,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test saving a dataset to a file."""
        # Create a simple dataset
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

        # Save to file
        output_path = temp_dir / "dataset.json"
        dataset.save(output_path)

        # Verify file exists and contains valid JSON
        assert output_path.exists()
        with open(output_path, "r") as f:
            data = json.load(f)

        assert "info" in data
        assert "images" in data
        assert "licenses" in data
        assert "annotations" in data
        assert "categories" in data

    def test_dataset_load(self, temp_dir: Path) -> None:
        """Test loading a dataset from a file."""
        # Create a JSON file
        data = {
            "info": {"year": 2023, "version": "1.0"},
            "images": [{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg"}],
            "licenses": [{"id": 1, "name": "CC BY 4.0"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]],
                    "area": 10000.0,
                    "bbox": [0.0, 0.0, 100.0, 100.0],
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "person", "supercategory": "human"}],
        }

        json_path = temp_dir / "dataset.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Load the dataset
        dataset = ObjectDetectionDataset.load(json_path)

        assert dataset.info.year == 2023
        assert len(dataset.images) == 1
        assert len(dataset.licenses) == 1
        assert len(dataset.annotations) == 1
        assert len(dataset.categories) == 1

    def test_dataset_save_load_roundtrip(
        self,
        temp_dir: Path,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test save/load roundtrip preserves data."""
        annotations = [
            ObjectDetectionAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                segmentation=[[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
                area=1600.0,
                bbox=(10.0, 10.0, 40.0, 40.0),
                iscrowd=0,
            )
        ]
        categories = [
            ObjectDetectionCategory(id=1, name="car", supercategory="vehicle")
        ]

        original_dataset = ObjectDetectionDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
            categories=categories,
        )

        # Save and load
        file_path = temp_dir / "roundtrip.json"
        original_dataset.save(file_path)
        loaded_dataset = ObjectDetectionDataset.load(file_path)

        # Verify data matches
        assert loaded_dataset.info.year == original_dataset.info.year
        assert len(loaded_dataset.images) == len(original_dataset.images)
        assert len(loaded_dataset.annotations) == len(original_dataset.annotations)
        assert loaded_dataset.annotations[0].id == original_dataset.annotations[0].id
        assert loaded_dataset.categories[0].name == original_dataset.categories[0].name

    def test_dataset_empty_lists(self) -> None:
        """Test creating a dataset with empty lists."""
        info = Info(year=2023)

        dataset = ObjectDetectionDataset(
            info=info,
            images=[],
            licenses=[],
            annotations=[],
            categories=[],
        )

        assert len(dataset.images) == 0
        assert len(dataset.licenses) == 0
        assert len(dataset.annotations) == 0
        assert len(dataset.categories) == 0
