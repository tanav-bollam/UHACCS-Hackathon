"""
CV module: Computer vision components for attention detection.

Exports:
- CameraCapture: Webcam frame capture
- AttentionDetector: Face/head gaze detection for is_attentive boolean
- is_attentive: Module-level function(frame) -> bool
"""

from cv.camera import CameraCapture
from cv.attention import AttentionDetector, is_attentive

__all__ = ["CameraCapture", "AttentionDetector", "is_attentive"]
