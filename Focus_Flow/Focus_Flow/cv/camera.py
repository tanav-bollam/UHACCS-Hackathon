"""
Camera module: Webcam frame capture for attention detection.

Provides frames to the attention detector. Uses OpenCV with optional stub mode
for headless/no-camera environments.
"""

from typing import Optional

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class CameraCapture:
    """
    Captures frames from the webcam for processing.

    When use_stub=True, returns dummy frames without opening a camera.
    TODO: Add frame rate limiting and resolution configuration.
    TODO: Handle camera initialization failures gracefully.
    """

    def __init__(self, device_id: int = 0, use_stub: bool = False):
        """
        Initialize camera capture.

        Args:
            device_id: Camera device index. 0 is usually the default webcam.
            use_stub: If True, skip OpenCV and return dummy frames (for headless).
        """
        self.device_id = device_id
        self.use_stub = use_stub
        self._cap = None
        if not use_stub and _CV2_AVAILABLE:
            self._cap = cv2.VideoCapture(device_id)

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the webcam.

        Returns:
            Frame as numpy array (H, W, C) BGR or None if capture failed.
            Stub returns a dummy 480x640x3 array for runnability.
        """
        if self.use_stub:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        if self._cap is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
