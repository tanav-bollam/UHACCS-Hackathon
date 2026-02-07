"""
Attention module: Face/head gaze detection for focus tracking.

Determines if the user is attentive (looking at screen) based on frame data.
Uses OpenCV Haar cascade for face detection - no paid API calls.
"""

from typing import Optional

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# Module-level detector instance for is_attentive() function
_detector: Optional["AttentionDetector"] = None


def is_attentive(frame: Optional[np.ndarray]) -> bool:
    """
    Determine if the user is attentive based on the frame.

    Module-level function that delegates to AttentionDetector.
    Use this for direct calls: is_attentive(camera.capture_frame())

    Args:
        frame: Webcam frame as numpy array (H, W, C) BGR. May be None.

    Returns:
        True if user appears attentive (face in center region), False otherwise.
    """
    global _detector
    if _detector is None:
        _detector = AttentionDetector()
    return _detector.is_attentive(frame)


class AttentionDetector:
    """
    Detects user attention from webcam frames using OpenCV face detection.

    Placeholder logic: face in center 60% of frame = attentive.
    TODO: Integrate head pose or gaze estimation for better accuracy.
    TODO: Add temporal smoothing to reduce false negatives.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize the attention detector.

        Args:
            threshold: Fraction of frame center region for "attentive" (0.6 = center 60%).
        """
        self.threshold = threshold
        self._cascade = None
        if _CV2_AVAILABLE:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            try:
                self._cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self._cascade = None

    def is_attentive(self, frame: Optional[np.ndarray]) -> bool:
        """
        Determine if the user is attentive based on the frame.

        Uses Haar face detection. If a face is in the center 60% of the frame,
        considers the user attentive.

        Args:
            frame: Webcam frame as numpy array (H, W, C). May be None.

        Returns:
            True if user appears attentive, False otherwise.
        """
        if frame is None:
            return False
        if not _CV2_AVAILABLE or self._cascade is None:
            return True
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        except Exception:
            return False
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return False
        h, w = frame.shape[:2]
        center_x_min = w * 0.2
        center_x_max = w * 0.8
        center_y_min = h * 0.2
        center_y_max = h * 0.8
        for (x, y, fw, fh) in faces:
            face_cx = x + fw / 2
            face_cy = y + fh / 2
            if center_x_min <= face_cx <= center_x_max and center_y_min <= face_cy <= center_y_max:
                return True
        return False
