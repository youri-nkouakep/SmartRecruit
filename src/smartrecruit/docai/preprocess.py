"""Image cleaning before OCR: illumination correction, deskew, denoising."""
import cv2
import numpy as np


def correct_illumination(gray: np.ndarray) -> np.ndarray:
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 31)
    return cv2.normalize(255 - cv2.absdiff(gray, background), None, 0, 255, cv2.NORM_MINMAX)


def estimate_skew(gray: np.ndarray, max_angle: float = 5.0, step: float = 0.2) -> float:
    """Angle (degrees) maximizing the variance of the horizontal projection profile."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = ink.shape
    best, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best, best_score = float(angle), score
    return best


def deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def preprocess(image: np.ndarray, upscale: float = 2.0) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = correct_illumination(gray)
    gray = deskew(gray, estimate_skew(gray))
    gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    if upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return gray
