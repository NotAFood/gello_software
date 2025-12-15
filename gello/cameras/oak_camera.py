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
    output_size: Tuple[int, int] = (640, 400)  # (width, height)
    fps: int = 30

    # Runtime state
    _device: Optional[Any] = field(default=None, init=False, repr=False)
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
        # In v3, we connect to device with device info (like the get_oak_intrinsics example)
        if self.device_mxid:
            device_info = dai.DeviceInfo(self.device_mxid)
            self._device = dai.Device(device_info)
        else:
            # Connect to first available device
            devices = dai.Device.getAllAvailableDevices()
            if not devices:
                raise RuntimeError("No OAK devices found")
            self._device = dai.Device(devices[0])

        # In v3, we need to build a pipeline for camera streaming
        # The pipeline creation happens after device connection
        pipeline = dai.Pipeline()
        
        # Create ColorCamera node
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        
        # Find RGB socket (same approach as get_oak_intrinsics)
        rgb_socket = (
            dai.CameraBoardSocket.CAM_A
            if hasattr(dai.CameraBoardSocket, "CAM_A")
            else getattr(dai.CameraBoardSocket, "RGB", dai.CameraBoardSocket.AUTO)
        )
        cam_rgb.setBoardSocket(rgb_socket)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setFps(self.fps)
        
        # Set output size
        w, h = int(self.output_size[0]), int(self.output_size[1])
        cam_rgb.setPreviewSize(w, h)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

            # In v3, we don't need XLinkOut - we create queues directly from outputs!
            # Start pipeline on the device first
        self._device.startPipeline(pipeline)
        
            # Create output queue directly from the camera output
            self._queue = cam_rgb.preview.createOutputQueue(maxSize=4, blocking=False)
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

        # Get latest frame (drain queue)
        frame_data = self._queue.get()
        while True:
            latest = self._queue.tryGet()
            if latest is None:
                break
            frame_data = latest

        # Convert to numpy array (already RGB due to setColorOrder)
        frame_rgb = frame_data.getCvFrame()

        # Get timestamp
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
            if self._device is not None:
                self._device.close()
        finally:
            self._device = None
            self._queue = None
            self._started = False

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
