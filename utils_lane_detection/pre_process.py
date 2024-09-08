import cv2 as cv
import numpy as np
from numpy.typing import NDArray


class Preprocess:
    @staticmethod
    def smooth_contour(contour: NDArray, sz: int = 3) -> NDArray:
        if not (sz % 2):
            sz += 1
        N = contour.shape[0]
        sz_p = (sz - 1) // 2
        y = contour[:, :, 0]
        x = contour[:, :, 1]
        y = cv.copyMakeBorder(y, sz_p, sz_p, 0, 0, cv.BORDER_WRAP)
        x = cv.copyMakeBorder(x, sz_p, sz_p, 0, 0, cv.BORDER_WRAP)
        y = cv.blur(y, (1, sz))
        x = cv.blur(x, (1, sz))
        contour[:, :, 0] = y[sz_p:sz_p + N]
        contour[:, :, 1] = x[sz_p:sz_p + N]
        return contour

    def __init__(self):
        self._morph_element_open = cv.getStructuringElement(cv.MORPH_CROSS, [3, 3])
        self._morph_element_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, [1, 9])

    def cleanup(self, m: NDArray) -> NDArray:
        m = cv.morphologyEx(m, cv.MORPH_OPEN, self._morph_element_open)
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, self._morph_element_close)
        contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours = [contour for contour in contours if len(contour) > 24]
        contours = [Preprocess.smooth_contour(contour, 11) for contour in contours]
        m = np.zeros_like(m)
        m = cv.drawContours(m, contours, -1, (255,), -1)
        m = cv.ximgproc.thinning(m)
        return m

