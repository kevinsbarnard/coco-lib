"""Tests for common COCO dataclasses."""

import json
from datetime import datetime
from pathlib import Path
from typing import List

from coco_lib.common import Dataset, Image, Info, License
from coco_lib.objectdetection import (
    ObjectDetectionAnnotation,
    ObjectDetectionCategory,
    ObjectDetectionDataset,
)


class TestInfo:
    """Test suite for Info dataclass."""

    def test_info_initialization(self) -> None:
        """Test creating an Info object with all fields."""
        info = Info(
            year=2023,
            version="1.0.0",
            description="Test dataset",
            contributor="Test Team",
            url="https://example.com",
            date_created=datetime(2023, 1, 15),
        )

        assert info.year == 2023
        assert info.version == "1.0.0"
        assert info.description == "Test dataset"
        assert info.contributor == "Test Team"
        assert info.url == "https://example.com"
        assert info.date_created == datetime(2023, 1, 15)

    def test_info_defaults(self) -> None:
        """Test Info with default values."""
        info = Info()

        assert info.year is None
        assert info.version is None
        assert info.description is None
        assert info.contributor is None
        assert info.url is None
        assert info.date_created is None

    def test_info_json_deserialization(self) -> None:
        """Test JSON deserialization of Info."""
        json_str = '{"year": 2023, "version": "1.0", "description": "Test"}'
        info = Info.from_json(json_str)

        assert info.year == 2023
        assert info.version == "1.0"
        assert info.description == "Test"

    def test_info_date_deserialization(self) -> None:
        """Test date deserialization format."""
        json_str = '{"date_created": "2023/01/15"}'
        info = Info.from_json(json_str)

        assert info.date_created == datetime(2023, 1, 15)

    def test_info_date_created_serializer_called(self) -> None:
        """Test that serialize_date_created is properly called with non-None value."""
        import json as json_module

        info = Info(
            year=2024,
            version="2.0",
            date_created=datetime(2024, 12, 5),
        )
        json_str = info.to_json()
        data = json_module.loads(json_str)

        # Verify the serializer was called and returned the formatted string
        assert data["date_created"] == "2024/12/05"
        assert "2024/12/05" in json_str

    def test_info_date_created_none_serialization(self) -> None:
        """Test that serialize_date_created handles None value correctly."""
        import json as json_module

        info = Info(
            year=2024,
            version="2.0",
            date_created=None,
        )
        # Don't use exclude_none to ensure the serializer is called with None
        json_str = info.model_dump_json()
        data = json_module.loads(json_str)

        # Verify that date_created is None in the output
        assert data["date_created"] is None


class TestImage:
    """Test suite for Image dataclass."""

    def test_image_initialization(self) -> None:
        """Test creating an Image object with required fields."""
        image = Image(
            id=1,
            width=1920,
            height=1080,
            file_name="test.jpg",
        )

        assert image.id == 1
        assert image.width == 1920
        assert image.height == 1080
        assert image.file_name == "test.jpg"

    def test_image_with_optional_fields(self) -> None:
        """Test Image with optional fields."""
        image = Image(
            id=1,
            width=640,
            height=480,
            file_name="test.jpg",
            license=1,
            flickr_url="https://flickr.com/photo/123",
            coco_url="https://example.com/image.jpg",
            date_captured=datetime(2023, 6, 15, 14, 30, 0),
        )

        assert image.license == 1
        assert image.flickr_url == "https://flickr.com/photo/123"
        assert image.coco_url == "https://example.com/image.jpg"
        assert image.date_captured == datetime(2023, 6, 15, 14, 30, 0)

    def test_image_json_deserialization(self) -> None:
        """Test JSON deserialization of Image."""
        json_str = '{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg"}'
        image = Image.from_json(json_str)

        assert image.id == 1
        assert image.width == 640
        assert image.height == 480
        assert image.file_name == "test.jpg"

    def test_image_date_captured_serializer_called(self) -> None:
        """Test that serialize_date_captured is properly called with non-None value."""
        import json as json_module

        image = Image(
            id=2,
            width=800,
            height=600,
            file_name="captured.jpg",
            date_captured=datetime(2024, 12, 5, 16, 45, 30),
        )
        json_str = image.to_json()
        data = json_module.loads(json_str)

        # Verify the serializer was called and returned the formatted string
        assert data["date_captured"] == "2024-12-05 16:45:30"
        assert "2024-12-05 16:45:30" in json_str

    def test_image_date_captured_none_serialization(self) -> None:
        """Test that serialize_date_captured handles None value correctly."""
        import json as json_module

        image = Image(
            id=3,
            width=1024,
            height=768,
            file_name="no_date.jpg",
            date_captured=None,
        )
        # Don't use exclude_none to ensure the serializer is called with None
        json_str = image.model_dump_json()
        data = json_module.loads(json_str)

        # Verify that date_captured is None in the output
        assert data["date_captured"] is None


class TestLicense:
    """Test suite for License dataclass."""

    def test_license_initialization(self) -> None:
        """Test creating a License object."""
        license_obj = License(
            id=1,
            name="CC BY 4.0",
            url="https://creativecommons.org/licenses/by/4.0/",
        )

        assert license_obj.id == 1
        assert license_obj.name == "CC BY 4.0"
        assert license_obj.url == "https://creativecommons.org/licenses/by/4.0/"

    def test_license_json_deserialization(self) -> None:
        """Test JSON deserialization of License."""
        json_str = (
            '{"id": 1, "name": "MIT", "url": "https://opensource.org/licenses/MIT"}'
        )
        license_obj = License.from_json(json_str)

        assert license_obj.id == 1
        assert license_obj.name == "MIT"
        assert license_obj.url == "https://opensource.org/licenses/MIT"


class TestDataset:
    """Test suite for Dataset base class."""

    def test_dataset_is_abstract(self) -> None:
        """Test that Dataset is an abstract base class."""
        assert issubclass(ObjectDetectionDataset, Dataset)

    def test_dataset_save(
        self,
        temp_dir: Path,
        sample_info: Info,
        sample_images: List[Image],
        sample_licenses: List[License],
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
        sample_images: List[Image],
        sample_licenses: List[License],
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
