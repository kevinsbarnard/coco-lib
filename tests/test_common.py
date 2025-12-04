"""Tests for common COCO dataclasses."""

from datetime import datetime

from coco_lib.common import Image, Info, License


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

    def test_info_partial_fields(self) -> None:
        """Test Info with only some fields set."""
        info = Info(year=2024, description="Partial info")

        assert info.year == 2024
        assert info.description == "Partial info"
        assert info.version is None

    def test_info_json_serialization(self) -> None:
        """Test JSON serialization of Info."""
        info = Info(year=2023, version="1.0")
        json_str = info.to_json()

        assert "2023" in json_str
        assert "1.0" in json_str

    def test_info_json_deserialization(self) -> None:
        """Test JSON deserialization of Info."""
        json_str = '{"year": 2023, "version": "1.0", "description": "Test"}'
        info = Info.from_json(json_str)

        assert info.year == 2023
        assert info.version == "1.0"
        assert info.description == "Test"

    def test_info_date_serialization(self) -> None:
        """Test date serialization format."""
        info = Info(date_created=datetime(2023, 1, 15))
        json_str = info.to_json()

        assert "2023/01/15" in json_str

    def test_info_date_deserialization(self) -> None:
        """Test date deserialization format."""
        json_str = '{"date_created": "2023/01/15"}'
        info = Info.from_json(json_str)

        assert info.date_created == datetime(2023, 1, 15)


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

    def test_image_defaults(self) -> None:
        """Test Image optional field defaults."""
        image = Image(id=1, width=100, height=100, file_name="test.jpg")

        assert image.license is None
        assert image.flickr_url is None
        assert image.coco_url is None
        assert image.date_captured is None

    def test_image_json_serialization(self) -> None:
        """Test JSON serialization of Image."""
        image = Image(id=1, width=640, height=480, file_name="test.jpg")
        json_str = image.to_json()

        assert '"id": 1' in json_str
        assert '"width": 640' in json_str
        assert '"height": 480' in json_str
        assert '"file_name": "test.jpg"' in json_str

    def test_image_json_deserialization(self) -> None:
        """Test JSON deserialization of Image."""
        json_str = '{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg"}'
        image = Image.from_json(json_str)

        assert image.id == 1
        assert image.width == 640
        assert image.height == 480
        assert image.file_name == "test.jpg"

    def test_image_date_captured_serialization(self) -> None:
        """Test date_captured serialization format."""
        image = Image(
            id=1,
            width=640,
            height=480,
            file_name="test.jpg",
            date_captured=datetime(2023, 6, 15, 14, 30, 0),
        )
        json_str = image.to_json()

        assert "2023-06-15 14:30:00" in json_str

    def test_image_different_dimensions(self) -> None:
        """Test images with various dimensions."""
        images = [
            Image(id=1, width=640, height=480, file_name="vga.jpg"),
            Image(id=2, width=1920, height=1080, file_name="fullhd.jpg"),
            Image(id=3, width=3840, height=2160, file_name="4k.jpg"),
        ]

        assert images[0].width == 640
        assert images[1].width == 1920
        assert images[2].width == 3840


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

    def test_license_without_url(self) -> None:
        """Test License without URL."""
        license_obj = License(id=1, name="Custom License")

        assert license_obj.id == 1
        assert license_obj.name == "Custom License"
        assert license_obj.url is None

    def test_license_json_serialization(self) -> None:
        """Test JSON serialization of License."""
        license_obj = License(id=1, name="Test License")
        json_str = license_obj.to_json()

        assert '"id": 1' in json_str
        assert '"name": "Test License"' in json_str

    def test_license_json_deserialization(self) -> None:
        """Test JSON deserialization of License."""
        json_str = (
            '{"id": 1, "name": "MIT", "url": "https://opensource.org/licenses/MIT"}'
        )
        license_obj = License.from_json(json_str)

        assert license_obj.id == 1
        assert license_obj.name == "MIT"
        assert license_obj.url == "https://opensource.org/licenses/MIT"

    def test_multiple_licenses(self) -> None:
        """Test working with multiple licenses."""
        licenses = [
            License(id=1, name="CC BY 4.0"),
            License(id=2, name="CC BY-SA 4.0"),
            License(id=3, name="CC0 1.0"),
        ]

        assert len(licenses) == 3
        assert licenses[0].name == "CC BY 4.0"
        assert licenses[1].name == "CC BY-SA 4.0"
        assert licenses[2].name == "CC0 1.0"
