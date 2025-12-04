"""Tests for keypoint detection module."""

import json
from pathlib import Path

from coco_lib.common import Image, Info, License
from coco_lib.keypointdetection import (
    KeypointDetectionAnnotation,
    KeypointDetectionCategory,
    KeypointDetectionDataset,
)


class TestKeypointDetectionAnnotation:
    """Test suite for KeypointDetectionAnnotation."""

    def test_annotation_initialization(self) -> None:
        """Test creating a KeypointDetectionAnnotation."""
        ann = KeypointDetectionAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            segmentation=[[100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0]],
            area=10000.0,
            bbox=(100.0, 100.0, 100.0, 100.0),
            iscrowd=0,
            keypoints=[150.0, 120.0, 2, 150.0, 160.0, 2, 130.0, 180.0, 1],
            num_keypoints=3,
        )

        assert ann.id == 1
        assert ann.keypoints == [150.0, 120.0, 2, 150.0, 160.0, 2, 130.0, 180.0, 1]
        assert ann.num_keypoints == 3
        assert len(ann.keypoints) == 9  # 3 keypoints * 3 values (x, y, v)

    def test_annotation_keypoint_format(self) -> None:
        """Test keypoint format with x, y, visibility triplets."""
        ann = KeypointDetectionAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            segmentation=[[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]],
            area=10000.0,
            bbox=(0.0, 0.0, 100.0, 100.0),
            iscrowd=0,
            keypoints=[50.0, 30.0, 2, 50.0, 70.0, 1, 45.0, 90.0, 0],
            num_keypoints=2,
        )

        # Extract first keypoint
        x1, y1, v1 = ann.keypoints[0:3]
        assert x1 == 50.0
        assert y1 == 30.0
        assert v1 == 2  # Visible

        # Extract second keypoint
        x2, y2, v2 = ann.keypoints[3:6]
        assert x2 == 50.0
        assert y2 == 70.0
        assert v2 == 1  # Labeled but not visible

        # Extract third keypoint
        x3, y3, v3 = ann.keypoints[6:9]
        assert v3 == 0  # Not labeled

    def test_annotation_visibility_flags(self) -> None:
        """Test different visibility flag values."""
        # v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible
        ann = KeypointDetectionAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            segmentation=[[0.0, 0.0, 10.0, 10.0]],
            area=100.0,
            bbox=(0.0, 0.0, 10.0, 10.0),
            iscrowd=0,
            keypoints=[10.0, 10.0, 0, 20.0, 20.0, 1, 30.0, 30.0, 2],
            num_keypoints=2,
        )

        assert ann.keypoints[2] == 0  # Not labeled
        assert ann.keypoints[5] == 1  # Labeled but not visible
        assert ann.keypoints[8] == 2  # Labeled and visible
        assert ann.num_keypoints == 2  # Only count v > 0

    def test_annotation_json_serialization(self) -> None:
        """Test JSON serialization of annotation."""
        ann = KeypointDetectionAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            segmentation=[[0.0, 0.0, 10.0, 10.0]],
            area=100.0,
            bbox=(0.0, 0.0, 10.0, 10.0),
            iscrowd=0,
            keypoints=[5.0, 5.0, 2],
            num_keypoints=1,
        )

        json_str = ann.to_json()
        assert '"keypoints"' in json_str
        assert '"num_keypoints": 1' in json_str

    def test_annotation_json_deserialization(self) -> None:
        """Test JSON deserialization of annotation."""
        json_str = """{
            "id": 1, "image_id": 1, "category_id": 1,
            "segmentation": [[0.0, 0.0, 10.0, 10.0]], 
            "area": 100.0, "bbox": [0.0, 0.0, 10.0, 10.0], 
            "iscrowd": 0, "keypoints": [5.0, 5.0, 2], 
            "num_keypoints": 1
        }"""
        ann = KeypointDetectionAnnotation.from_json(json_str)

        assert ann.num_keypoints == 1
        assert ann.keypoints == [5.0, 5.0, 2]


class TestKeypointDetectionCategory:
    """Test suite for KeypointDetectionCategory."""

    def test_category_initialization(self) -> None:
        """Test creating a KeypointDetectionCategory."""
        cat = KeypointDetectionCategory(
            id=1,
            name="person",
            supercategory="human",
            keypoints=["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
            skeleton=[(1, 2), (1, 3), (2, 4), (3, 5)],
        )

        assert cat.id == 1
        assert cat.name == "person"
        assert len(cat.keypoints) == 5
        assert len(cat.skeleton) == 4

    def test_category_skeleton_structure(self) -> None:
        """Test skeleton connections are 1-based pairs."""
        cat = KeypointDetectionCategory(
            id=1,
            name="person",
            supercategory="human",
            keypoints=["head", "neck", "left_shoulder", "right_shoulder"],
            skeleton=[(1, 2), (2, 3), (2, 4)],  # 1-based indexing
        )

        # Skeleton connects head to neck, neck to left shoulder, neck to right shoulder
        assert cat.skeleton[0] == (1, 2)
        assert cat.skeleton[1] == (2, 3)
        assert cat.skeleton[2] == (2, 4)

    def test_category_json_serialization(self) -> None:
        """Test JSON serialization of category."""
        cat = KeypointDetectionCategory(
            id=1,
            name="person",
            supercategory="human",
            keypoints=["nose", "left_eye"],
            skeleton=[(1, 2)],
        )

        json_str = cat.to_json()
        assert '"keypoints"' in json_str
        assert '"skeleton"' in json_str

    def test_category_json_deserialization(self) -> None:
        """Test JSON deserialization of category."""
        json_str = """{
            "id": 1, "name": "person", "supercategory": "human",
            "keypoints": ["nose", "left_eye", "right_eye"],
            "skeleton": [[1, 2], [1, 3]]
        }"""
        cat = KeypointDetectionCategory.from_json(json_str)

        assert len(cat.keypoints) == 3
        assert len(cat.skeleton) == 2
        assert cat.keypoints[0] == "nose"


class TestKeypointDetectionDataset:
    """Test suite for KeypointDetectionDataset."""

    def test_dataset_initialization(
        self,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test creating a KeypointDetectionDataset."""
        annotations = [
            KeypointDetectionAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                segmentation=[[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]],
                area=10000.0,
                bbox=(0.0, 0.0, 100.0, 100.0),
                iscrowd=0,
                keypoints=[50.0, 30.0, 2, 50.0, 70.0, 2],
                num_keypoints=2,
            )
        ]
        categories = [
            KeypointDetectionCategory(
                id=1,
                name="person",
                supercategory="human",
                keypoints=["head", "body"],
                skeleton=[(1, 2)],
            )
        ]

        dataset = KeypointDetectionDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
            categories=categories,
        )

        assert len(dataset.annotations) == 1
        assert len(dataset.categories) == 1
        assert dataset.categories[0].keypoints == ["head", "body"]

    def test_dataset_multiple_annotations(
        self,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test dataset with multiple person annotations."""
        annotations = [
            KeypointDetectionAnnotation(
                id=i,
                image_id=1,
                category_id=1,
                segmentation=[[0.0, 0.0, 10.0, 10.0]],
                area=100.0,
                bbox=(float(i * 10), 0.0, 10.0, 10.0),
                iscrowd=0,
                keypoints=[float(i * 10 + 5), 5.0, 2],
                num_keypoints=1,
            )
            for i in range(1, 4)
        ]
        categories = [
            KeypointDetectionCategory(
                id=1,
                name="person",
                supercategory="human",
                keypoints=["center"],
                skeleton=[],
            )
        ]

        dataset = KeypointDetectionDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
            categories=categories,
        )

        assert len(dataset.annotations) == 3
        assert all(ann.num_keypoints == 1 for ann in dataset.annotations)

    def test_dataset_save_load(
        self,
        temp_dir: Path,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test saving and loading a dataset."""
        annotations = [
            KeypointDetectionAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                segmentation=[[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]],
                area=10000.0,
                bbox=(0.0, 0.0, 100.0, 100.0),
                iscrowd=0,
                keypoints=[50.0, 30.0, 2, 50.0, 70.0, 2],
                num_keypoints=2,
            )
        ]
        categories = [
            KeypointDetectionCategory(
                id=1,
                name="person",
                supercategory="human",
                keypoints=["head", "body"],
                skeleton=[(1, 2)],
            )
        ]

        dataset = KeypointDetectionDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
            categories=categories,
        )

        # Save and load
        file_path = temp_dir / "keypoint_detection.json"
        dataset.save(file_path)
        loaded = KeypointDetectionDataset.load(file_path)

        assert len(loaded.annotations) == len(dataset.annotations)
        assert loaded.annotations[0].num_keypoints == 2
        assert loaded.categories[0].keypoints == ["head", "body"]

    def test_dataset_json_structure(
        self,
        temp_dir: Path,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test the JSON structure of saved dataset."""
        annotations = [
            KeypointDetectionAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                segmentation=[[0.0, 0.0, 10.0, 10.0]],
                area=100.0,
                bbox=(0.0, 0.0, 10.0, 10.0),
                iscrowd=0,
                keypoints=[5.0, 5.0, 2],
                num_keypoints=1,
            )
        ]
        categories = [
            KeypointDetectionCategory(
                id=1,
                name="person",
                supercategory="human",
                keypoints=["center"],
                skeleton=[],
            )
        ]

        dataset = KeypointDetectionDataset(
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

        assert "annotations" in data
        assert "categories" in data
        assert "keypoints" in data["annotations"][0]
        assert "skeleton" in data["categories"][0]
