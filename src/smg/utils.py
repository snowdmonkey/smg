from typing import Iterator

import cv2
import numpy as np


def iter_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        else:
            yield frame
