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