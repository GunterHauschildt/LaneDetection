import cv2 as cv
import numpy as np
from collections import defaultdict


class Postprocess:
    def __init__(self, num, sequential_count=4, Q=0.025, R=32.):
        self._num = num
        self._filters = {}
        self._enabled_count = defaultdict(lambda: 0)
        self._didnt_find_count = defaultdict(lambda: 0)
        self._sequential_count = sequential_count

        for lane in range(0, num):
            self._filters[lane] = cv.KalmanFilter(4, 2)
            self._filters[lane].measurementMatrix = (
                np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            )
            self._filters[lane].transitionMatrix = (
                np.array([[1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], np.float32)
            )
            self._filters[lane].measurementNoiseCov = (
                    np.array([[1, 0], [0, 1]], np.float32) * R
            )
            self._filters[lane].processNoiseCov = (
                    np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], np.float32) * Q)

    def correct(self, lane: int, value: tuple[float, float]):
        m = value[0]
        b = value[1]
        if self._enabled_count[lane] == 0:
            self._filters[lane].statePre = np.array([[m], [b], [0], [0]], np.float32)
        if self._enabled_count[lane] <= self._sequential_count:
            self._enabled_count[lane] += 1
        self._filters[lane].correct(np.array([m, b]).astype(np.float32))

    def predict(self, lane: int) -> int or None:
        if self._enabled_count[lane] < self._sequential_count:
            return None
        else:
            return self._filters[lane].predict()

    def disable(self, lane):
        self._enabled_count[lane] = 0

    def filter(self,
               all_lane_descriptors: dict[int, list[tuple[float, float]]]
               ) -> dict[int, tuple[float, float]]:

        # lanes have a 'life-time'.
        # - if we miss a lane, for a short time, we predict it instead
        # - when we find a missing lane, it needs a couple of cycles to apear

        # run the 'didn't find' life-time counter
        didnt_find = []
        for i in range(self._num):
            if i not in all_lane_descriptors:
                self._didnt_find_count[i] += 1
            else:
                self._didnt_find_count[i] = 0

        for i in range(self._num):
            if self._didnt_find_count[i] >= self._sequential_count:
                didnt_find.append(i)
                self.disable(i)

        # if there's more than one, average
        predicted_lane_descriptors = {}
        for i, lane_descriptors in all_lane_descriptors.items():
            avgs = np.average(np.array(lane_descriptors), axis=0)
            avgs_m = float(avgs[0])
            avgs_b = float(avgs[1])
            predicted_lane_descriptors[i] = (avgs_m, avgs_b)

        # insert a prediction if we didn't find the lane
        for lane in range(self._num):
            if lane not in didnt_find and lane not in all_lane_descriptors:
                m_b_predicted = self.predict(lane)
                if m_b_predicted is not None:
                    m_b_predicted = np.squeeze(m_b_predicted, axis=-1)[0:2]
                    m = float(m_b_predicted[0])
                    b = float(m_b_predicted[1])
                    predicted_lane_descriptors[lane] = (m, b)

        # kalman filter. the 'just found' lifetime is handled in the
        # Kalman filter itself
        for lane in range(self._num):
            if lane in predicted_lane_descriptors:
                m = predicted_lane_descriptors[lane][0]
                b = predicted_lane_descriptors[lane][1]
                self.correct(lane, (m, b))
                m_b_filtered = self.predict(lane)
                if m_b_filtered is not None:
                    m_b_filtered = np.squeeze(m_b_filtered, axis=-1)[0:2]
                    m = float(m_b_filtered[0])
                    b = float(m_b_filtered[1])
                    predicted_lane_descriptors[lane] = (m, b)
                else:
                    del predicted_lane_descriptors[lane]

        return predicted_lane_descriptors

