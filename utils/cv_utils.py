import cv2 as cv
import numpy as np
from numpy.typing import NDArray


def cv_to_tf(frame: NDArray, tf_size_xy: tuple[int, int]) -> NDArray:
    frame = cv.resize(frame, tf_size_xy)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame
