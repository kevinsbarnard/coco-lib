"""Pytest configuration and shared fixtures."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator

import pytest

from coco_lib.common import Image, Info, License


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files.

    Yields:
        Path: Temporary directory path that is cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_info() -> Info:
    """Provide a sample Info object for testing.

    Returns:
        Info: Sample dataset info with standard test values.
    """
    return Info(
        year=2023,
        version="1.0.0",
        description="Test COCO dataset",
        contributor="Test Team",
        url="https://example.com/dataset",
        date_created=datetime(2023, 1, 15),
    )


@pytest.fixture
def sample_image() -> Image:
    """Provide a sample Image object for testing.

    Returns:
        Image: Sample image metadata with standard test values.
    """
    return Image(
        id=1,
        width=640,
        height=480,
        file_name="test_image.jpg",
        license=1,
        date_captured=datetime(2023, 6, 15, 14, 30, 0),
    )


@pytest.fixture
def sample_images() -> list[Image]:
    """Provide a list of sample Image objects for testing.

    Returns:
        list[Image]: List of sample images with varied dimensions.
    """
    return [
        Image(id=1, width=640, height=480, file_name="image_001.jpg", license=1),
        Image(id=2, width=1920, height=1080, file_name="image_002.jpg", license=1),
        Image(id=3, width=800, height=600, file_name="image_003.jpg", license=2),
    ]


@pytest.fixture
def sample_license() -> License:
    """Provide a sample License object for testing.

    Returns:
        License: Sample license with standard test values.
    """
    return License(
        id=1,
        name="Creative Commons Attribution 4.0",
        url="https://creativecommons.org/licenses/by/4.0/",
    )


@pytest.fixture
def sample_licenses() -> list[License]:
    """Provide a list of sample License objects for testing.

    Returns:
        list[License]: List of sample licenses.
    """
    return [
        License(
            id=1, name="CC BY 4.0", url="https://creativecommons.org/licenses/by/4.0/"
        ),
        License(
            id=2,
            name="CC BY-SA 4.0",
            url="https://creativecommons.org/licenses/by-sa/4.0/",
        ),
    ]
