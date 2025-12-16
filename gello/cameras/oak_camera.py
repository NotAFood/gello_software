"""
DepthAI OAK color camera driver for calibration data collection.

Minimal wrapper around DepthAI SDK v3.x for capturing RGB images from OAK cameras.
Supports device selection by MxId for multi-camera setups.

Based on DepthAI v3.x API which simplified device connections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

try:
    import depthai as dai
except ImportError:
    dai = None


@dataclass
class OakColorCamera:
    """Minimal OAK color camera driver for calibration capture using DepthAI v3.x API."""

    # Camera identification
    name: str = "oak_camera"
    device_mxid: Optional[str] = None  # Specific device MxId; None = first available

    # Output configuration
    output_size: Tuple[int, int] = (1280, 800)  # (width, height)
    fps: int = 30

    # Runtime state
    _device: Optional[Any] = field(default=None, init=False, repr=False)
    _pipeline: Optional[Any] = field(default=None, init=False, repr=False)
    _queue: Optional[Any] = field(default=None, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if dai is None:
            raise RuntimeError(
                "depthai package not available. Install with: pip install depthai"
            )
        self._build_and_start()

    def _build_and_start(self):
        """Connect to device and start camera using v3.x API."""
        # In v3, we connect to device directly by MxId or first available
        if self.device_mxid:
            self._device = dai.Device(self.device_mxid)
        else:
            # Connect to first available device
            devices = dai.Device.getAllAvailableDevices()
            if not devices:
                raise RuntimeError("No OAK devices found")
            self._device = dai.Device(devices[0])

        # Create pipeline with device context
        self._pipeline = dai.Pipeline(self._device)

        # Create camera node using simplified v3 API
        cam = self._pipeline.create(dai.node.Camera).build()

        # Request output with specific resolution
        w, h = int(self.output_size[0]), int(self.output_size[1])
        self._queue = cam.requestOutput((w, h)).createOutputQueue()

        # Start pipeline
        self._pipeline.start()
        self._started = True

    def read(self) -> Tuple[np.ndarray, float]:
        """Read the latest RGB frame from camera.

        Returns:
            Tuple of:
                - RGB image as np.ndarray (H, W, 3) uint8
                - Timestamp in milliseconds
        """
        if not self._started or not self._queue:
            raise RuntimeError("Camera not started")

        # Get latest frame from queue
        frame_data = self._queue.get()

        # Verify frame type and extract image
        if not isinstance(frame_data, dai.ImgFrame):
            raise RuntimeError(f"Unexpected frame type: {type(frame_data)}")

        # Convert to numpy array (OpenCV format)
        frame_rgb = frame_data.getCvFrame()

        # Get timestamp (in milliseconds)
        timestamp = frame_data.getTimestamp()
        timestamp_ms = timestamp.total_seconds() * 1000.0

        return frame_rgb, timestamp_ms

    def get_device_id(self) -> str:
        """Get the MxId of the connected device."""
        if not self._device:
            raise RuntimeError("Device not initialized")
        return self._device.getMxId()

    def stop(self) -> None:
        """Stop camera and release resources."""
        try:
            if self._pipeline is not None:
                # Pipeline cleanup is handled automatically in v3
                pass
        finally:
            self._device = None
            self._pipeline = None
            self._queue = None
            self._started = False

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
