# Frames: the currency of every video

## 1. Frames with a timestamp

Every source gives the same object, a `Frame` with the image in `.array`
(values between 0 and 1) and the time in seconds in `.t`.

```python
import numpy as np
from dense_armor.utility.vision import ArrayStream

frames = np.zeros((5, 120, 160), dtype=np.float32)
s = ArrayStream(frames, fps=30.0)
for f in s:
    print(f.array.shape, round(f.t, 4))
```

```
(120, 160) 0.0
(120, 160) 0.0333
(120, 160) 0.0667
(120, 160) 0.1
(120, 160) 0.1333
```

Five frames, 120 rows by 160 columns, one every 1/30 of a second: the
timestamps grow by 0.0333 s at a time. The image never leaves the array,
it is just a numpy view with a timestamp attached.

The same object comes from three sources. `ArrayStream` reads frames
already in memory, which is what tests use. `CameraStream(device=0, fps=30,
size=(160, 120))` reads a real camera through OpenCV; OpenCV is installed
with `pip install dense-armor[vision]`, and if it is missing the constructor
raises a clear `ImportError` with that exact message. A real camera
delivers frames in OpenCV's native BGR order, which is turned into RGB
before the grey conversion, so a red square stays a red square in the
features. `ImageFolderStream(path, fps=...)` reads a folder of images in
name order, using Pillow, which is already a dependency of matplotlib;
if no image file is found the constructor raises a `ValueError`.

The `Frame` object itself is deliberately thin: a numpy array and a
timestamp, nothing more. Every source — a USB camera, a folder of PNGs, a
numpy array built on the fly — produces frames in different ways and at
different rates, and the rest of the library should not care which one it
is reading from. The stream is the adapter, the frame is the common
currency. When a robot gets a new camera, only the stream changes.

![Five frames of a synthetic scene, one every 1/30 s, with the timestamp under each.](../assets/vision/frames/five_frames.png)

## Reading any video

A video is a stream of frames in time, and it comes from a file, an
URL, a glob of files, a camera, or a cube of multi-channel samples.
Whatever the source, every stream yields the same object, a
`Frame`, with the image and
the timestamp of the frame. The time of each frame is the container's
own clock, not ``index / fps``: a recording whose frame rate changes
with time keeps the true time of every frame.

    >>> import tempfile, pathlib
    >>> import numpy as np
    >>> import cv2
    >>> d = pathlib.Path(tempfile.mkdtemp()) / "clip.mp4"
    >>> w = cv2.VideoWriter(str(d), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (8, 8))
    >>> for i in range(5):
    ...     _ = w.write(np.full((8, 8, 3), i * 50, dtype=np.uint8))
    >>> w.release()
    >>> from dense_armor.utility.vision.streams import VideoFileStream
    >>> for fr in VideoFileStream(str(d)):
    ...     print(round(fr.t, 2))
    0.0
    0.1
    0.2
    0.3
    0.4

A cube of multi-channel samples (a thermal sequence, a multispectral
stack, a depth map) reads with
`CubeStream`. A cube with a
16-bit depth keeps every value of the sensor without loss.

![Four bands of a synthetic thermal cube; the hot square sits at 350 in every band.](../assets/vision/frames/thermal_bands.png)
