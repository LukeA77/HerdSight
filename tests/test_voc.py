import numpy as np
import pytest

from herdsight.data.voc import parse_voc

VOC = """<annotation>
  <filename>DJI_0001.JPG</filename>
  <size><width>5280</width><height>3956</height><depth>3</depth></size>
  <object><name>cow</name><bndbox><xmin>11</xmin><ymin>21</ymin><xmax>50</xmax><ymax>40</ymax></bndbox></object>
  <object><name>Cow</name><bndbox><xmin>101.5</xmin><ymin>201</ymin><xmax>150</xmax><ymax>260</ymax></bndbox></object>
  <object><name>person</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
</annotation>"""


def test_parse_voc_converts_to_zero_based_and_filters(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text(VOC)
    a = parse_voc(p, keep=["cow"])
    assert (a.filename, a.width, a.height) == ("DJI_0001.JPG", 5280, 3956)
    np.testing.assert_allclose(a.boxes_xyxy, [[10, 20, 50, 40], [100.5, 200, 150, 260]])
    assert a.names == ("cow", "Cow")
    assert parse_voc(p).boxes_xyxy.shape == (3, 4)


def test_empty_annotation(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text("<annotation><size><width>10</width><height>10</height></size></annotation>")
    assert parse_voc(p).boxes_xyxy.shape == (0, 4)


def test_entity_expansion_attack_is_refused(tmp_path):
    p = tmp_path / "bad.xml"
    p.write_text(
        '<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "xxxxxxxx">'
        '<!ENTITY y "&x;&x;&x;&x;">]><annotation>&y;</annotation>'
    )
    with pytest.raises(Exception, match="(?i)entit"):
        parse_voc(p)
