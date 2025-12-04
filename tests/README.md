# Tests for coco-lib

This directory contains comprehensive tests for the coco-lib package.

## Test Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_common.py` - Tests for common dataclasses (Info, Image, License)
- `test_bases.py` - Tests for base classes (Annotation, Category, Dataset)
- `test_objectdetection.py` - Tests for object detection module
- `test_imagecaptioning.py` - Tests for image captioning module
- `test_keypointdetection.py` - Tests for keypoint detection module
- `test_panopticsegmentation.py` - Tests for panoptic segmentation module
- `test_integration.py` - Integration tests for complete workflows

## Running Tests

### Run all tests
```bash
just test
```

### Run tests with coverage (generates HTML report)
```bash
just test-cov
```

### Run doctests
```bash
just test-doctest
```

### Run all tests including doctests with coverage
```bash
just test-all
```

## Test Coverage

The test suite aims for comprehensive coverage including:

- **Unit tests** - Test individual classes and methods
- **Integration tests** - Test complete workflows
- **Serialization tests** - Test JSON encoding/decoding
- **Doctests** - Test examples in docstrings

## Writing Tests

Tests follow pytest conventions:
- Test files are named `test_*.py`
- Test classes are named `Test*`
- Test methods are named `test_*`
- Use type annotations for better clarity
- Use descriptive test names that explain what is being tested

## Fixtures

Common fixtures are defined in `conftest.py`:
- `temp_dir` - Temporary directory for test files
- `sample_info` - Sample Info object
- `sample_image` - Sample Image object
- `sample_images` - List of sample Image objects
- `sample_license` - Sample License object
- `sample_licenses` - List of sample License objects
