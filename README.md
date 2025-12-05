# coco-lib
COCO dataset library. Provides serializable native Python bindings for several COCO dataset formats.

Supported bindings and their corresponding modules:

- Object Detection: `objectdetection`
- Keypoint Detection: `keypointdetection`
- Panoptic Segmentation: `panopticsegmentation`
- Image Captioning: `imagecaptioning`

## Installation

`coco-lib` is available on PyPI:

``` bash
pip install coco-lib
```

## Usage

### Creating a dataset (Object Detection)

```python
>>> from coco_lib.common import Info, Image, License
>>> from coco_lib.objectdetection import ObjectDetectionAnnotation, \
...                                      ObjectDetectionCategory, \
...                                      ObjectDetectionDataset
>>> from datetime import datetime
>>> info = Info(  # Describe the dataset
...    year=datetime.now().year, 
...    version='1.0', 
...    description='This is a test dataset', 
...    contributor='Test', 
...    url='https://test', 
...    date_created=datetime.now()
... )
>>> mit_license = License(  # Set the license
...     id=0, 
...     name='MIT', 
...     url='https://opensource.org/licenses/MIT'
... )
>>> images = [  # Describe the images
...     Image(
...         id=0, 
...         width=640, height=480, 
...         file_name='test.jpg', 
...         license=mit_license.id,
...         flickr_url='',
...         coco_url='',
...         date_captured=datetime.now()
...     ),
...     ...
... ]
>>> categories = [  # Describe the categories
...     ObjectDetectionCategory(
...         id=0,
...         name='pedestrian',
...         supercategory=''
...     ),
...     ...
... ]
>>> annotations = [  # Describe the annotations
...     ObjectDetectionAnnotation(
...         id=0,
...         image_id=0,
...         category_id=0,
...         segmentation=[],
...         area=800.0,
...         bbox=[300.0, 100.0, 20.0, 40.0],
...         is_crowd=0
...     ),
...     ...
... ]
>>> dataset = ObjectDetectionDataset(  # Create the dataset
...     info=info,
...     images=images,
...     licenses=[mit_license],
...     categories=categories,
...     annotations=annotations
... )
>>> dataset.save('test_dataset.json', indent=2)  # Save the dataset
```

### Loading a dataset

```python
>>> from coco_lib.objectdetection import ObjectDetectionDataset
>>> dataset = ObjectDetectionDataset.load('test_dataset.json')  # Load the dataset
```

### Flexible Datetime Parsing

The library now supports flexible datetime parsing using [dateparser](https://dateparser.readthedocs.io/). Date fields (`Info.date_created` and `Image.date_captured`) can now accept various datetime formats:

```python
>>> from coco_lib.common import Info, Image
>>> # Various date formats are automatically parsed
>>> info1 = Info.from_json('{"date_created": "2023/01/15"}')  # Original format
>>> info2 = Info.from_json('{"date_created": "2023-01-15"}')  # ISO format
>>> info3 = Info.from_json('{"date_created": "January 15, 2023"}')  # Natural language
>>> info4 = Info.from_json('{"date_created": "15 Jan 2023"}')  # Short format
>>> # All produce the same date
>>> assert info1.date_created.date() == info2.date_created.date() == info3.date_created.date() == info4.date_created.date()
>>> # Invalid dates return None and emit a warning
>>> import warnings
>>> with warnings.catch_warnings(record=True) as w:
...     warnings.simplefilter("always")
...     info = Info.from_json('{"date_created": "invalid date"}')
...     assert info.date_created is None  # Returns None
...     assert len([warning for warning in w if issubclass(warning.category, UserWarning)]) > 0  # Warning emitted
>>> # Serialization maintains original formats for backward compatibility
>>> from datetime import datetime
>>> info = Info(date_created=datetime(2023, 1, 15))
>>> json_str = info.to_json()
>>> assert "2023/01/15" in json_str  # Uses YYYY/MM/DD format
```