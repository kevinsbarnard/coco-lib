"""Tests for flexible datetime parsing with dateparser."""

import warnings
from datetime import datetime

from coco_lib.common import Image, Info


class TestFlexibleDatetimeParsing:
    """Test suite for flexible datetime parsing using dateparser."""

    def test_info_date_created_various_formats(self) -> None:
        """Test that date_created can parse various datetime formats."""
        test_cases = [
            ('{"date_created": "2023/01/15"}', datetime(2023, 1, 15)),
            ('{"date_created": "2023-01-15"}', datetime(2023, 1, 15)),
            ('{"date_created": "January 15, 2023"}', datetime(2023, 1, 15)),
            ('{"date_created": "15 Jan 2023"}', datetime(2023, 1, 15)),
            ('{"date_created": "Jan 15, 2023"}', datetime(2023, 1, 15)),
            ('{"date_created": "2023.01.15"}', datetime(2023, 1, 15)),
        ]

        for json_str, expected_date in test_cases:
            info = Info.from_json(json_str)
            assert info.date_created is not None
            assert info.date_created.date() == expected_date.date()

    def test_image_date_captured_various_formats(self) -> None:
        """Test that date_captured can parse various datetime formats."""
        base_json = '{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg", "date_captured": "%s"}'

        test_cases = [
            ("2023-06-15 14:30:00", datetime(2023, 6, 15, 14, 30, 0)),
            ("2023-06-15T14:30:00", datetime(2023, 6, 15, 14, 30, 0)),
            ("June 15, 2023 2:30 PM", datetime(2023, 6, 15, 14, 30, 0)),
            ("15/06/2023 14:30", datetime(2023, 6, 15, 14, 30, 0)),
            ("2023-06-15", datetime(2023, 6, 15, 0, 0, 0)),
        ]

        for date_str, expected_date in test_cases:
            json_str = base_json % date_str
            image = Image.from_json(json_str)
            assert image.date_captured is not None
            # Compare date and hour/minute, not necessarily seconds due to parsing variations
            assert image.date_captured.date() == expected_date.date()
            if ":" in date_str:  # If time was specified, check it
                assert image.date_captured.hour == expected_date.hour
                assert image.date_captured.minute == expected_date.minute

    def test_info_invalid_date_returns_none_with_warning(self) -> None:
        """Test that invalid date strings return None and emit warnings."""
        json_str = '{"date_created": "not a valid date"}'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = Info.from_json(json_str)

            assert info.date_created is None
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Failed to parse datetime string" in str(w[0].message)
            assert "not a valid date" in str(w[0].message)

    def test_image_invalid_date_returns_none_with_warning(self) -> None:
        """Test that invalid date strings return None and emit warnings."""
        json_str = '{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg", "date_captured": "invalid"}'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            image = Image.from_json(json_str)

            assert image.date_captured is None
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Failed to parse datetime string" in str(w[0].message)
            assert "invalid" in str(w[0].message)

    def test_info_null_date_no_warning(self) -> None:
        """Test that null dates don't produce warnings."""
        json_str = '{"date_created": null}'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = Info.from_json(json_str)

            assert info.date_created is None
            # Filter out DeprecationWarnings from dateparser
            user_warnings = [
                warning for warning in w if issubclass(warning.category, UserWarning)
            ]
            assert len(user_warnings) == 0

    def test_image_null_date_no_warning(self) -> None:
        """Test that null dates don't produce warnings."""
        json_str = '{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg", "date_captured": null}'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            image = Image.from_json(json_str)

            assert image.date_captured is None
            # Filter out DeprecationWarnings from dateparser
            user_warnings = [
                warning for warning in w if issubclass(warning.category, UserWarning)
            ]
            assert len(user_warnings) == 0

    def test_info_iso_format(self) -> None:
        """Test parsing ISO 8601 format dates."""
        test_cases = [
            '{"date_created": "2023-01-15T00:00:00"}',
            '{"date_created": "2023-01-15T00:00:00Z"}',
            '{"date_created": "2023-01-15T00:00:00+00:00"}',
        ]

        for json_str in test_cases:
            info = Info.from_json(json_str)
            assert info.date_created is not None
            assert info.date_created.date() == datetime(2023, 1, 15).date()

    def test_image_iso_format(self) -> None:
        """Test parsing ISO 8601 format dates for images."""
        test_cases = [
            '"2023-06-15T14:30:00"',
            '"2023-06-15T14:30:00Z"',
            '"2023-06-15T14:30:00+00:00"',
        ]

        base_json = '{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg", "date_captured": %s}'

        for date_str in test_cases:
            json_str = base_json % date_str
            image = Image.from_json(json_str)
            assert image.date_captured is not None
            assert image.date_captured.date() == datetime(2023, 6, 15).date()
            assert image.date_captured.hour == 14
            assert image.date_captured.minute == 30

    def test_relative_date_parsing(self) -> None:
        """Test that relative dates are parsed (though may not be stable for assertions)."""
        json_str = '{"date_created": "today"}'

        info = Info.from_json(json_str)
        assert info.date_created is not None
        # Just check that it's a valid datetime, don't check the exact value
        assert isinstance(info.date_created, datetime)

    def test_backward_compatibility_with_original_formats(self) -> None:
        """Test that the original datetime formats still work correctly."""
        # Original Info format: %Y/%m/%d
        info_json = '{"date_created": "2023/01/15"}'
        info = Info.from_json(info_json)
        assert info.date_created == datetime(2023, 1, 15)

        # Original Image format: %Y-%m-%d %H:%M:%S
        image_json = '{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg", "date_captured": "2023-06-15 14:30:00"}'
        image = Image.from_json(image_json)
        assert image.date_captured == datetime(2023, 6, 15, 14, 30, 0)

    def test_serialization_format_unchanged(self) -> None:
        """Test that serialization still uses the original formats."""
        # Test Info serialization
        info = Info(date_created=datetime(2023, 1, 15))
        json_str = info.to_json()
        assert "2023/01/15" in json_str

        # Test Image serialization
        image = Image(
            id=1,
            width=640,
            height=480,
            file_name="test.jpg",
            date_captured=datetime(2023, 6, 15, 14, 30, 0),
        )
        json_str = image.to_json()
        assert "2023-06-15 14:30:00" in json_str

    def test_info_non_string_type_triggers_error_warning(self) -> None:
        """Test that non-string types trigger an error and emit warning."""
        # Test with integer
        json_str = '{"date_created": 12345}'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = Info.from_json(json_str)

            assert info.date_created is None
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Error parsing datetime string" in str(w[0].message)
            assert "12345" in str(w[0].message)

    def test_image_non_string_type_triggers_error_warning(self) -> None:
        """Test that non-string types trigger an error and emit warning."""
        # Test with boolean
        json_str = '{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg", "date_captured": true}'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            image = Image.from_json(json_str)

            assert image.date_captured is None
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Error parsing datetime string" in str(w[0].message)
            assert "True" in str(w[0].message)

    def test_info_none_input_returns_none_without_warning(self) -> None:
        """Test that None input returns None without emitting warnings."""
        from coco_lib.common import parse_datetime

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = parse_datetime(None)

            assert result is None
            user_warnings = [
                warning for warning in w if issubclass(warning.category, UserWarning)
            ]
            assert len(user_warnings) == 0

    def test_info_missing_date_field_no_warning(self) -> None:
        """Test that missing date field doesn't produce warnings."""
        json_str = '{"year": 2023, "version": "1.0"}'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = Info.from_json(json_str)

            assert info.date_created is None
            user_warnings = [
                warning for warning in w if issubclass(warning.category, UserWarning)
            ]
            assert len(user_warnings) == 0
