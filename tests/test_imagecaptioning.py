"""Tests for image captioning module."""

import json
from pathlib import Path

from coco_lib.common import Image, Info, License
from coco_lib.imagecaptioning import (
    ImageCaptioningAnnotation,
    ImageCaptioningDataset,
)


class TestImageCaptioningAnnotation:
    """Test suite for ImageCaptioningAnnotation."""

    def test_annotation_initialization(self) -> None:
        """Test creating an ImageCaptioningAnnotation."""
        ann = ImageCaptioningAnnotation(
            id=1,
            image_id=1,
            caption="A person riding a bicycle on a sunny day",
        )

        assert ann.id == 1
        assert ann.image_id == 1
        assert ann.caption == "A person riding a bicycle on a sunny day"

    def test_annotation_long_caption(self) -> None:
        """Test annotation with a long caption."""
        long_caption = (
            "A detailed scene showing multiple people walking through "
            "a busy city street with tall buildings in the background, "
            "cars passing by, and trees lining the sidewalks."
        )
        ann = ImageCaptioningAnnotation(
            id=1,
            image_id=1,
            caption=long_caption,
        )

        assert ann.caption == long_caption
        assert len(ann.caption) > 100

    def test_annotation_empty_caption(self) -> None:
        """Test annotation with empty caption."""
        ann = ImageCaptioningAnnotation(
            id=1,
            image_id=1,
            caption="",
        )

        assert ann.caption == ""

    def test_annotation_json_serialization(self) -> None:
        """Test JSON serialization of annotation."""
        ann = ImageCaptioningAnnotation(
            id=1,
            image_id=1,
            caption="Test caption",
        )

        json_str = ann.to_json()
        assert '"id": 1' in json_str
        assert '"image_id": 1' in json_str
        assert '"caption": "Test caption"' in json_str

    def test_annotation_json_deserialization(self) -> None:
        """Test JSON deserialization of annotation."""
        json_str = '{"id": 1, "image_id": 1, "caption": "A test image"}'
        ann = ImageCaptioningAnnotation.from_json(json_str)

        assert ann.id == 1
        assert ann.image_id == 1
        assert ann.caption == "A test image"

    def test_multiple_captions_per_image(self) -> None:
        """Test multiple captions for the same image."""
        captions = [
            ImageCaptioningAnnotation(
                id=1,
                image_id=1,
                caption="A cat sitting on a couch",
            ),
            ImageCaptioningAnnotation(
                id=2,
                image_id=1,
                caption="A feline resting on furniture",
            ),
            ImageCaptioningAnnotation(
                id=3,
                image_id=1,
                caption="A pet cat relaxing indoors",
            ),
        ]

        assert all(ann.image_id == 1 for ann in captions)
        assert len(captions) == 3
        assert len(set(ann.caption for ann in captions)) == 3


class TestImageCaptioningDataset:
    """Test suite for ImageCaptioningDataset."""

    def test_dataset_initialization(
        self,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test creating an ImageCaptioningDataset."""
        annotations = [
            ImageCaptioningAnnotation(
                id=1,
                image_id=1,
                caption="Test caption",
            )
        ]

        dataset = ImageCaptioningDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
        )

        assert dataset.info == sample_info
        assert len(dataset.images) == 3
        assert len(dataset.annotations) == 1

    def test_dataset_multiple_captions(
        self,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test dataset with multiple captions per image."""
        annotations = []
        caption_id = 1
        for img in sample_images[:2]:  # First 2 images
            for i in range(5):  # 5 captions per image
                annotations.append(
                    ImageCaptioningAnnotation(
                        id=caption_id,
                        image_id=img.id,
                        caption=f"Caption {i + 1} for image {img.id}",
                    )
                )
                caption_id += 1

        dataset = ImageCaptioningDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
        )

        assert len(dataset.annotations) == 10  # 2 images * 5 captions

    def test_dataset_save_load(
        self,
        temp_dir: Path,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test saving and loading a dataset."""
        annotations = [
            ImageCaptioningAnnotation(
                id=1,
                image_id=1,
                caption="A beautiful landscape",
            ),
            ImageCaptioningAnnotation(
                id=2,
                image_id=1,
                caption="Scenic nature view",
            ),
        ]

        dataset = ImageCaptioningDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
        )

        # Save and load
        file_path = temp_dir / "image_captioning.json"
        dataset.save(file_path)
        loaded = ImageCaptioningDataset.load(file_path)

        assert len(loaded.annotations) == len(dataset.annotations)
        assert loaded.annotations[0].caption == dataset.annotations[0].caption
        assert loaded.annotations[1].caption == dataset.annotations[1].caption

    def test_dataset_json_structure(
        self,
        temp_dir: Path,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test the JSON structure of saved dataset."""
        annotations = [
            ImageCaptioningAnnotation(
                id=1,
                image_id=1,
                caption="Test caption",
            )
        ]

        dataset = ImageCaptioningDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=annotations,
        )

        file_path = temp_dir / "structure_test.json"
        dataset.save(file_path)

        with open(file_path, "r") as f:
            data = json.load(f)

        assert "info" in data
        assert "images" in data
        assert "licenses" in data
        assert "annotations" in data
        assert isinstance(data["annotations"], list)
        assert "caption" in data["annotations"][0]

    def test_dataset_empty_annotations(
        self,
        sample_info: Info,
        sample_images: list[Image],
        sample_licenses: list[License],
    ) -> None:
        """Test dataset with no annotations."""
        dataset = ImageCaptioningDataset(
            info=sample_info,
            images=sample_images,
            licenses=sample_licenses,
            annotations=[],
        )

        assert len(dataset.annotations) == 0
        assert len(dataset.images) == 3
