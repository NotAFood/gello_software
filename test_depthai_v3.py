#!/usr/bin/env python3
import depthai as dai

print(f"DepthAI version: {dai.__version__}")

# Create pipeline
pipeline = dai.Pipeline()

# Try Camera node (v3 API)
cam = pipeline.create(dai.node.Camera)
print("\nCamera methods (set*):")
for m in sorted([x for x in dir(cam) if x.startswith('set')]):
    print(f"  {m}")

# Check outputs
print("\nCamera outputs:")
for attr in ['video', 'preview', 'still', 'isp', 'raw']:
    if hasattr(cam, attr):
        print(f"  {attr}: {type(getattr(cam, attr))}")

# Check device connection methods
print("\nPipeline create methods:")
for m in sorted([x for x in dir(pipeline) if 'create' in x.lower()]):
    print(f"  {m}")
