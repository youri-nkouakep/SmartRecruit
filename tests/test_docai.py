import random

import numpy as np
from faker import Faker

from smartrecruit.docai.evaluate import f1_table, field_counts, same
from smartrecruit.docai.layout import find_column_split, heading_type, segment
from smartrecruit.docai.ocr import Line, Page
from smartrecruit.docai.preprocess import deskew, estimate_skew
from smartrecruit.docai.synth import make_resume


def test_heading_type():
    assert heading_type("PROFESSIONAL EXPERIENCE") == "experience"
    assert heading_type("Technical Skils") == "skills"
    assert heading_type("Built a data pipeline in Python") is None


def test_column_split_and_segments():
    left = [Line(f"left {i}", 20, 50 + 20 * i, 150, 60 + 20 * i, 10) for i in range(6)]
    right = [Line(f"right {i}", 220, 50 + 20 * i, 550, 60 + 20 * i, 10) for i in range(6)]
    split = find_column_split(Page(left + right, 600, 800))
    assert split is not None and 150 <= split <= 220
    sections = segment([Line("Jane Doe", 0, 0, 1, 1, 20), Line("SKILLS", 0, 1, 1, 2, 12),
                        Line("Python, SQL", 0, 2, 1, 3, 10)])
    assert [l.text for l in sections["skills"]] == ["Python, SQL"]


def test_deskew_recovers_angle():
    img = np.full((400, 400), 255, np.uint8)
    img[100:110:2, 50:350] = 0
    img[200:210:2, 50:350] = 0
    import cv2
    M = cv2.getRotationMatrix2D((200, 200), 2.0, 1.0)
    rotated = cv2.warpAffine(img, M, (400, 400), borderValue=255)
    assert abs(estimate_skew(rotated) + 2.0) <= 0.4
    assert deskew(rotated, 0).shape == rotated.shape


def test_field_scores():
    resume, _ = make_resume(random.Random(0), Faker())
    assert f1_table([field_counts(resume, resume)])["overall"] == 1.0
    assert same("Pyth0n", "Python", 0.8)
