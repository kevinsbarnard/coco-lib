"""Integration tests for complete workflows."""

from datetime import datetime
from pathlib import Path

from coco_lib.common import Image, Info, License
from coco_lib.imagecaptioning import (
    ImageCaptioningAnnotation,
    ImageCaptioningDataset,
)
from coco_lib.keypointdetection import (
    KeypointDetectionAnnotation,
    KeypointDetectionCategory,
    KeypointDetectionDataset,
)
from coco_lib.objectdetection import (
    ObjectDetectionAnnotation,
    ObjectDetectionCategory,
    ObjectDetectionDataset,
)
from coco_lib.panopticsegmentation import (
    PanopticSegmentationAnnotation,
    PanopticSegmentationCategory,
    PanopticSegmentationDataset,
    SegmentInfo,
)


class TestObjectDetectionWorkflow:
    """Integration test for object detection workflow."""

    def test_complete_workflow(self, temp_dir: Path) -> None:
        """Test creating, saving, loading, and modifying an object detection dataset."""
        # Create dataset
        info = Info(
            year=2023,
            version="1.0.0",
            description="Object Detection Test Dataset",
            contributor="Test Team",
            date_created=datetime(2023, 1, 15),
        )

        images = [
            Image(id=1, width=1920, height=1080, file_name="img_001.jpg", license=1),
            Image(id=2, width=1920, height=1080, file_name="img_002.jpg", license=1),
            Image(id=3, width=1920, height=1080, file_name="img_003.jpg", license=2),
        ]

        licenses = [
            License(
                id=1,
                name="CC BY 4.0",
                url="https://creativecommons.org/licenses/by/4.0/",
            ),
            License(
                id=2,
                name="CC0 1.0",
                url="https://creativecommons.org/publicdomain/zero/1.0/",
            ),
        ]

        categories = [
            ObjectDetectionCategory(id=1, name="person", supercategory="human"),
            ObjectDetectionCategory(id=2, name="car", supercategory="vehicle"),
            ObjectDetectionCategory(id=3, name="dog", supercategory="animal"),
        ]

        annotations = [
            ObjectDetectionAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                segmentation=[[100.0, 100.0, 300.0, 100.0, 300.0, 400.0, 100.0, 400.0]],
                area=60000.0,
                bbox=(100.0, 100.0, 200.0, 300.0),
                iscrowd=0,
            ),
            ObjectDetectionAnnotation(
                id=2,
                image_id=1,
                category_id=2,
                segmentation=[[500.0, 300.0, 800.0, 300.0, 800.0, 500.0, 500.0, 500.0]],
                area=60000.0,
                bbox=(500.0, 300.0, 300.0, 200.0),
                iscrowd=0,
            ),
            ObjectDetectionAnnotation(
                id=3,
                image_id=2,
                category_id=3,
                segmentation=[[200.0, 400.0, 400.0, 400.0, 400.0, 600.0, 200.0, 600.0]],
                area=40000.0,
                bbox=(200.0, 400.0, 200.0, 200.0),
                iscrowd=0,
            ),
        ]

        dataset = ObjectDetectionDataset(
            info=info,
            images=images,
            licenses=licenses,
            annotations=annotations,
            categories=categories,
        )

        # Save to file
        output_path = temp_dir / "object_detection_full.json"
        dataset.save(output_path)

        # Verify file exists
        assert output_path.exists()

        # Load from file
        loaded_dataset = ObjectDetectionDataset.load(output_path)

        # Verify data integrity
        assert loaded_dataset.info.year == 2023
        assert len(loaded_dataset.images) == 3
        assert len(loaded_dataset.licenses) == 2
        assert len(loaded_dataset.annotations) == 3
        assert len(loaded_dataset.categories) == 3

        # Verify specific annotation
        person_ann = next(a for a in loaded_dataset.annotations if a.category_id == 1)
        assert person_ann.area == 60000.0
        assert person_ann.bbox == (100.0, 100.0, 200.0, 300.0)


class TestImageCaptioningWorkflow:
    """Integration test for image captioning workflow."""

    def test_complete_workflow(self, temp_dir: Path) -> None:
        """Test creating and managing an image captioning dataset."""
        info = Info(year=2023, version="1.0", description="Captioning Dataset")

        images = [
            Image(id=1, width=640, height=480, file_name="img_001.jpg", license=1),
            Image(id=2, width=640, height=480, file_name="img_002.jpg", license=1),
        ]

        licenses = [License(id=1, name="CC BY 4.0")]

        # Multiple captions per image
        annotations = []
        ann_id = 1
        for img in images:
            for i in range(5):
                annotations.append(
                    ImageCaptioningAnnotation(
                        id=ann_id,
                        image_id=img.id,
                        caption=f"Caption {i + 1} for image {img.id}: Description here.",
                    )
                )
                ann_id += 1

        dataset = ImageCaptioningDataset(
            info=info,
            images=images,
            licenses=licenses,
            annotations=annotations,
        )

        # Save and reload
        output_path = temp_dir / "captioning_full.json"
        dataset.save(output_path)
        loaded_dataset = ImageCaptioningDataset.load(output_path)

        # Verify
        assert len(loaded_dataset.annotations) == 10  # 2 images * 5 captions
        img1_captions = [a for a in loaded_dataset.annotations if a.image_id == 1]
        assert len(img1_captions) == 5


class TestKeypointDetectionWorkflow:
    """Integration test for keypoint detection workflow."""

    def test_complete_workflow(self, temp_dir: Path) -> None:
        """Test creating and managing a keypoint detection dataset."""
        info = Info(year=2023, version="1.0", description="Keypoint Dataset")

        images = [
            Image(id=1, width=1920, height=1080, file_name="person_001.jpg", license=1),
        ]

        licenses = [License(id=1, name="CC BY 4.0")]

        # Define person keypoints
        categories = [
            KeypointDetectionCategory(
                id=1,
                name="person",
                supercategory="human",
                keypoints=[
                    "nose",
                    "left_eye",
                    "right_eye",
                    "left_ear",
                    "right_ear",
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                ],
                skeleton=[
                    (1, 2),
                    (1, 3),
                    (2, 4),
                    (3, 5),  # Head connections
                    (1, 6),
                    (1, 7),  # Nose to shoulders
                    (6, 8),
                    (8, 10),  # Left arm
                    (7, 9),
                    (9, 11),  # Right arm
                    (6, 12),
                    (7, 13),  # Shoulders to hips
                    (12, 14),
                    (14, 16),  # Left leg
                    (13, 15),
                    (15, 17),  # Right leg
                ],
            )
        ]

        # Create annotation with keypoints
        annotations = [
            KeypointDetectionAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                segmentation=[[300.0, 200.0, 500.0, 200.0, 500.0, 800.0, 300.0, 800.0]],
                area=120000.0,
                bbox=(300.0, 200.0, 200.0, 600.0),
                iscrowd=0,
                keypoints=[
                    400.0,
                    250.0,
                    2,  # nose
                    380.0,
                    240.0,
                    2,  # left_eye
                    420.0,
                    240.0,
                    2,  # right_eye
                    360.0,
                    245.0,
                    2,  # left_ear
                    440.0,
                    245.0,
                    2,  # right_ear
                    350.0,
                    320.0,
                    2,  # left_shoulder
                    450.0,
                    320.0,
                    2,  # right_shoulder
                    320.0,
                    420.0,
                    2,  # left_elbow
                    480.0,
                    420.0,
                    2,  # right_elbow
                    310.0,
                    520.0,
                    2,  # left_wrist
                    490.0,
                    520.0,
                    2,  # right_wrist
                    360.0,
                    500.0,
                    2,  # left_hip
                    440.0,
                    500.0,
                    2,  # right_hip
                    350.0,
                    650.0,
                    2,  # left_knee
                    450.0,
                    650.0,
                    2,  # right_knee
                    340.0,
                    780.0,
                    2,  # left_ankle
                    460.0,
                    780.0,
                    2,  # right_ankle
                ],
                num_keypoints=17,
            )
        ]

        dataset = KeypointDetectionDataset(
            info=info,
            images=images,
            licenses=licenses,
            annotations=annotations,
            categories=categories,
        )

        # Save and reload
        output_path = temp_dir / "keypoint_full.json"
        dataset.save(output_path)
        loaded_dataset = KeypointDetectionDataset.load(output_path)

        # Verify
        assert len(loaded_dataset.categories[0].keypoints) == 17
        assert len(loaded_dataset.categories[0].skeleton) == 16
        assert loaded_dataset.annotations[0].num_keypoints == 17


class TestPanopticSegmentationWorkflow:
    """Integration test for panoptic segmentation workflow."""

    def test_complete_workflow(self, temp_dir: Path) -> None:
        """Test creating and managing a panoptic segmentation dataset."""
        info = Info(year=2023, version="1.0", description="Panoptic Dataset")

        images = [
            Image(id=1, width=1920, height=1080, file_name="scene_001.jpg", license=1),
        ]

        licenses = [License(id=1, name="CC BY 4.0")]

        # Mix of things and stuff
        categories = [
            PanopticSegmentationCategory(
                id=1,
                name="person",
                supercategory="human",
                isthing=1,
                color=(220, 20, 60),
            ),
            PanopticSegmentationCategory(
                id=2, name="car", supercategory="vehicle", isthing=1, color=(0, 0, 142)
            ),
            PanopticSegmentationCategory(
                id=3,
                name="sky",
                supercategory="background",
                isthing=0,
                color=(70, 130, 180),
            ),
            PanopticSegmentationCategory(
                id=4, name="grass", supercategory="nature", isthing=0, color=(0, 255, 0)
            ),
        ]

        # Complex scene with multiple segments
        annotations = [
            PanopticSegmentationAnnotation(
                image_id=1,
                file_name="scene_001_mask.png",
                segments_info=[
                    SegmentInfo(
                        id=1,
                        category_id=1,
                        area=50000.0,
                        bbox=(300.0, 200.0, 200.0, 500.0),
                        iscrowd=0,
                    ),
                    SegmentInfo(
                        id=2,
                        category_id=1,
                        area=48000.0,
                        bbox=(700.0, 250.0, 180.0, 480.0),
                        iscrowd=0,
                    ),
                    SegmentInfo(
                        id=3,
                        category_id=2,
                        area=80000.0,
                        bbox=(1000.0, 400.0, 400.0, 200.0),
                        iscrowd=0,
                    ),
                    SegmentInfo(
                        id=4,
                        category_id=3,
                        area=500000.0,
                        bbox=(0.0, 0.0, 1920.0, 400.0),
                        iscrowd=0,
                    ),
                    SegmentInfo(
                        id=5,
                        category_id=4,
                        area=400000.0,
                        bbox=(0.0, 600.0, 1920.0, 480.0),
                        iscrowd=0,
                    ),
                ],
            )
        ]

        dataset = PanopticSegmentationDataset(
            info=info,
            images=images,
            licenses=licenses,
            annotations=annotations,
            categories=categories,
        )

        # Save and reload
        output_path = temp_dir / "panoptic_full.json"
        dataset.save(output_path)
        loaded_dataset = PanopticSegmentationDataset.load(output_path)

        # Verify
        assert len(loaded_dataset.annotations[0].segments_info) == 5
        things = [c for c in loaded_dataset.categories if c.isthing == 1]
        stuff = [c for c in loaded_dataset.categories if c.isthing == 0]
        assert len(things) == 2
        assert len(stuff) == 2


class TestCrossDatasetCompatibility:
    """Test compatibility and conversions between dataset types."""

    def test_shared_metadata(self, temp_dir: Path) -> None:
        """Test that all dataset types share common metadata structures."""
        # Create common metadata
        info = Info(year=2023, version="1.0", description="Test")
        images = [Image(id=1, width=640, height=480, file_name="test.jpg", license=1)]
        licenses = [License(id=1, name="CC BY 4.0")]

        # Create different dataset types with same metadata
        od_dataset = ObjectDetectionDataset(
            info=info,
            images=images,
            licenses=licenses,
            annotations=[
                ObjectDetectionAnnotation(
                    id=1,
                    image_id=1,
                    category_id=1,
                    segmentation=[[0.0, 0.0, 10.0, 10.0]],
                    area=100.0,
                    bbox=(0.0, 0.0, 10.0, 10.0),
                    iscrowd=0,
                )
            ],
            categories=[
                ObjectDetectionCategory(id=1, name="obj", supercategory="thing")
            ],
        )

        ic_dataset = ImageCaptioningDataset(
            info=info,
            images=images,
            licenses=licenses,
            annotations=[ImageCaptioningAnnotation(id=1, image_id=1, caption="Test")],
        )

        # Save both
        od_path = temp_dir / "od.json"
        ic_path = temp_dir / "ic.json"
        od_dataset.save(od_path)
        ic_dataset.save(ic_path)

        # Load and verify info is consistent
        loaded_od = ObjectDetectionDataset.load(od_path)
        loaded_ic = ImageCaptioningDataset.load(ic_path)

        assert loaded_od.info.year == loaded_ic.info.year == 2023
        assert loaded_od.images[0].file_name == loaded_ic.images[0].file_name
