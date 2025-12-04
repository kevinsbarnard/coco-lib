"""Tests for panoptic segmentation module."""

import json
from pathlib import Path

from coco_lib.common import Image, Info, License
from coco_lib.panopticsegmentation import (
    PanopticSegmentationAnnotation,
    PanopticSegmentationCategory,
    PanopticSegmentationDataset,
    SegmentInfo,
)


class TestSegmentInfo:
    """Test suite for SegmentInfo."""

    def test_segment_initialization(self) -> None:
        """Test creating a SegmentInfo object."""
        segment = SegmentInfo(
            id=1,
            category_id=1,
            area=5000.0,
            bbox=(50.0, 50.0, 100.0, 50.0),
            iscrowd=0,
        )

        assert segment.id == 1
        assert segment.category_id == 1
        assert segment.area == 5000.0
        assert segment.bbox == (50.0, 50.0, 100.0, 50.0)
        assert segment.iscrowd == 0

    def test_segment_json_serialization(self) -> None:
        """Test JSON serialization of SegmentInfo."""
        segment = SegmentInfo(
            id=1, category_id=1, area=1000.0, bbox=(0.0, 0.0, 10.0, 10.0), iscrowd=0
        )

        json_str = segment.to_json()
        assert '"id": 1' in json_str
        assert '"category_id": 1' in json_str

    def test_segment_json_deserialization(self) -> None:
        """Test JSON deserialization of SegmentInfo."""
        json_str = """{
            "id": 1, "category_id": 2, "area": 1500.0,
            "bbox": [10.0, 20.0, 30.0, 40.0], "iscrowd": 0
        }"""
        segment = SegmentInfo.from_json(json_str)

        assert segment.id == 1
        assert segment.category_id == 2
        assert segment.area == 1500.0


class TestPanopticSegmentationAnnotation:
    """Test suite for PanopticSegmentationAnnotation."""

    def test_annotation_initialization(self) -> None:
        """Test creating a PanopticSegmentationAnnotation."""
        segments = [
            SegmentInfo(
                id=1,
                category_id=1,
                area=5000.0,
                bbox=(50.0, 50.0, 100.0, 50.0),
                iscrowd=0,
            ),
            SegmentInfo(
                id=2,
                category_id=2,
                area=3000.0,
                bbox=(200.0, 200.0, 50.0, 60.0),
                iscrowd=0,
            ),
        ]

        ann = PanopticSegmentationAnnotation(
            image_id=1,
            file_name="mask_001.png",
            segments_info=segments,
        )

        assert ann.image_id == 1
        assert ann.file_name == "mask_001.png"
        assert len(ann.segments_info) == 2

    def test_annotation_single_segment(self) -> None:
        """Test annotation with a single segment."""
        ann = PanopticSegmentationAnnotation(
            image_id=1,
            file_name="mask_001.png",
            segments_info=[
                SegmentInfo(
                    id=1,
                    category_id=1,
                    area=10000.0,
                    bbox=(0.0, 0.0, 100.0, 100.0),
                    iscrowd=0,
                )
            ],
        )

        assert len(ann.segments_info) == 1
        assert ann.segments_info[0].area == 10000.0

    def test_annotation_multiple_segments(self) -> None:
        """Test annotation with many segments."""
        segments = [
            SegmentInfo(
                id=i,
                category_id=(i % 3) + 1,
                area=float(i * 100),
                bbox=(float(i * 10), 0.0, 10.0, 10.0),
                iscrowd=0,
            )
            for i in range(1, 11)
        ]

        ann = PanopticSegmentationAnnotation(
            image_id=1,
            file_name="mask_complex.png",
            segments_info=segments,
        )

        assert len(ann.segments_info) == 10
        assert all(seg.iscrowd == 0 for seg in ann.segments_info)

    def test_annotation_json_serialization(self) -> None:
        """Test JSON serialization of annotation."""
        ann = PanopticSegmentationAnnotation(
            image_id=1,
            file_name="mask_001.png",
            segments_info=[
                SegmentInfo(
                    id=1,
                    category_id=1,
                    area=100.0,
                    bbox=(0.0, 0.0, 10.0, 10.0),
                    iscrowd=0,
                )
            ],
        )

        json_str = ann.to_json()
        assert '"image_id": 1' in json_str
        assert '"file_name": "mask_001.png"' in json_str
        assert '"segments_info"' in json_str

    def test_annotation_json_deserialization(self) -> None:
        """Test JSON deserialization of annotation."""
        json_str = """{
            "image_id": 1,
            "file_name": "mask.png",
            "segments_info": [
                {"id": 1, "category_id": 1, "area": 100.0, 
                 "bbox": [0.0, 0.0, 10.0, 10.0], "iscrowd": 0}
            ]
        }"""
        ann = PanopticSegmentationAnnotation.from_json(json_str)

        assert ann.image_id == 1
        assert ann.file_name == "mask.png"
        assert len(ann.segments_info) == 1


class TestPanopticSegmentationCategory:
    """Test suite for PanopticSegmentationCategory."""

    def test_category_initialization_thing(self) -> None:
        """Test creating a 'thing' category."""
        cat = PanopticSegmentationCategory(
            id=1,
            name="person",
            supercategory="human",
            isthing=1,
            color=(220, 20, 60),
        )

        assert cat.id == 1
        assert cat.name == "person"
        assert cat.isthing == 1
        assert cat.color == (220, 20, 60)

    def test_category_initialization_stuff(self) -> None:
        """Test creating a 'stuff' category."""
        cat = PanopticSegmentationCategory(
            id=2,
            name="sky",
            supercategory="background",
            isthing=0,
            color=(70, 130, 180),
        )

        assert cat.name == "sky"
        assert cat.isthing == 0

    def test_category_color_values(self) -> None:
        """Test color RGB values."""
        cat = PanopticSegmentationCategory(
            id=1, name="car", supercategory="vehicle", isthing=1, color=(255, 0, 0)
        )

        r, g, b = cat.color
        assert r == 255
        assert g == 0
        assert b == 0

    def test_category_json_serialization(self) -> None:
        """Test JSON serialization of category."""
        cat = PanopticSegmentationCategory(
            id=1, name="person", supercategory="human", isthing=1, color=(220, 20, 60)
        )

        json_str = cat.to_json()
        assert '"isthing": 1' in json_str
        assert '"color"' in json_str

    def test_category_json_deserialization(self) -> None:
        """Test JSON deserialization of category."""
        json_str = """{
            "id": 1, "name": "grass", "supercategory": "nature",
            "isthing": 0, "color": [0, 255, 0]
        }"""
        cat = PanopticSegmentationCategory.from_json(json_str)

        assert cat.name == "grass"
        assert cat.isthing == 0
        assert cat.color == (0, 255, 0)


class TestPanopticSegmentationDataset:
    """Test suite for PanopticSegmentationDataset."""

    def test_dataset_initialization(
        self,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test creating a PanopticSegmentationDataset."""
        annotations = [
            PanopticSegmentationAnnotation(
                image_id=1,
                file_name="mask_001.png",
                segments_info=[
                    SegmentInfo(
                        id=1,
                        category_id=1,
                        area=5000.0,
                        bbox=(50.0, 50.0, 100.0, 50.0),
                        iscrowd=0,
                    )
                ],
            )
        ]
        categories = [
            PanopticSegmentationCategory(
                id=1,
                name="person",
                supercategory="human",
                isthing=1,
                color=(220, 20, 60),
            )
        ]

        dataset = PanopticSegmentationDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
            categories=categories,
        )

        assert len(dataset.annotations) == 1
        assert len(dataset.categories) == 1

    def test_dataset_mixed_categories(
        self,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test dataset with both things and stuff categories."""
        annotations = [
            PanopticSegmentationAnnotation(
                image_id=1,
                file_name="mask_001.png",
                segments_info=[
                    SegmentInfo(
                        id=1,
                        category_id=1,
                        area=5000.0,
                        bbox=(0.0, 0.0, 100.0, 50.0),
                        iscrowd=0,
                    ),
                    SegmentInfo(
                        id=2,
                        category_id=2,
                        area=20000.0,
                        bbox=(0.0, 0.0, 640.0, 480.0),
                        iscrowd=0,
                    ),
                ],
            )
        ]
        categories = [
            PanopticSegmentationCategory(
                id=1,
                name="person",
                supercategory="human",
                isthing=1,
                color=(220, 20, 60),
            ),
            PanopticSegmentationCategory(
                id=2,
                name="sky",
                supercategory="background",
                isthing=0,
                color=(70, 130, 180),
            ),
        ]

        dataset = PanopticSegmentationDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
            categories=categories,
        )

        things = [cat for cat in dataset.categories if cat.isthing == 1]
        stuff = [cat for cat in dataset.categories if cat.isthing == 0]

        assert len(things) == 1
        assert len(stuff) == 1

    def test_dataset_save_load(
        self,
        temp_dir: Path,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test saving and loading a dataset."""
        annotations = [
            PanopticSegmentationAnnotation(
                image_id=1,
                file_name="mask_001.png",
                segments_info=[
                    SegmentInfo(
                        id=1,
                        category_id=1,
                        area=5000.0,
                        bbox=(50.0, 50.0, 100.0, 50.0),
                        iscrowd=0,
                    )
                ],
            )
        ]
        categories = [
            PanopticSegmentationCategory(
                id=1,
                name="person",
                supercategory="human",
                isthing=1,
                color=(220, 20, 60),
            )
        ]

        dataset = PanopticSegmentationDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
            categories=categories,
        )

        # Save and load
        file_path = temp_dir / "panoptic_segmentation.json"
        dataset.save(file_path)
        loaded = PanopticSegmentationDataset.load(file_path)

        assert len(loaded.annotations) == len(dataset.annotations)
        assert loaded.categories[0].isthing == 1
        assert loaded.categories[0].color == (220, 20, 60)

    def test_dataset_json_structure(
        self,
        temp_dir: Path,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test the JSON structure of saved dataset."""
        annotations = [
            PanopticSegmentationAnnotation(
                image_id=1,
                file_name="mask_001.png",
                segments_info=[
                    SegmentInfo(
                        id=1,
                        category_id=1,
                        area=100.0,
                        bbox=(0.0, 0.0, 10.0, 10.0),
                        iscrowd=0,
                    )
                ],
            )
        ]
        categories = [
            PanopticSegmentationCategory(
                id=1, name="object", supercategory="thing", isthing=1, color=(255, 0, 0)
            )
        ]

        dataset = PanopticSegmentationDataset(
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
        assert "segments_info" in data["annotations"][0]
        assert "isthing" in data["categories"][0]
        assert "color" in data["categories"][0]
