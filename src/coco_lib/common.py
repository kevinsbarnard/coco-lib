from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import dataclass_json, config
from marshmallow import fields


@dataclass_json
@dataclass
class Info:
    year: Optional[int] = None
    version: Optional[str] = None
    description: Optional[str] = None
    contributor: Optional[str] = None
    url: Optional[str] = None
    date_created: Optional[datetime] = field(
        metadata=config(
            encoder=lambda d: d.strftime("%Y/%m/%d") if d is not None else None,
            decoder=lambda s: datetime.strptime(s, "%Y/%m/%d")
            if s is not None
            else None,
            mm_field=fields.DateTime(format="%Y/%m/%d"),
        )
    )


@dataclass_json
@dataclass
class Image:
    id: int
    width: int
    height: int
    file_name: str
    license: Optional[int] = None
    flickr_url: Optional[str] = None
    coco_url: Optional[str] = None
    date_captured: Optional[datetime] = field(
        metadata=config(
            encoder=lambda d: d.strftime("%Y-%m-%d %H:%M:%S")
            if d is not None
            else None,
            decoder=lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            if s is not None
            else None,
            mm_field=fields.DateTime(format="iso"),
        )
    )


@dataclass_json
@dataclass
class License:
    id: int
    name: str
    url: Optional[str] = None
