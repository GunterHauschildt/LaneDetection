from numpy.typing import NDArray
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class Calibrate:

    def __init__(self, num_lane_markers, enable, calibration_frames):
        self._file_name = 'lines_calibration.npy'
        self._num_lane_markers = num_lane_markers

        self._kmeans = None
        self._cluster_centers = None

        self._calibrate_frame_nums = [f for row in calibration_frames for f in
                                      range(row[0], row[1])]

        self._lane_markers = None  # []
        self._is_calibrated = False
        self._is_calibrating = enable
        self._avgs = None
        self._limits = None
        self._kmeans_to_left_right_lut = None
        if os.path.isfile(self._file_name):
            self._lane_descriptors = np.load(self._file_name)
            self._kmeans_train(self._lane_descriptors)
            self._is_calibrated = True

    def is_calibrated(self):
        return self._is_calibrated

    def limits(self):
        return self._limits

    def is_calibrating(self):
        return self._is_calibrating

    def cluster_centers(self):
        return self._cluster_centers

    def _kmeans_train(self, lane_markers_raw: NDArray, draw=True):

        # TODO, it'd be smart to weight based on the length of the sample

        def find_outliers(data, md=2.5):
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            s = d / (mdev if mdev else 1.)
            return list(np.where([s > md])[1])

        outliers_m = find_outliers(lane_markers_raw[:, 0])
        outliers_b = find_outliers(lane_markers_raw[:, 1])
        outliers = outliers_m + [outlier_b for outlier_b in outliers_b if
                                 outlier_b not in outliers_m]

        lane_markers_c = np.array(
            [lane_markers_raw[i] for i in range(lane_markers_raw.shape[0]) if i not in outliers])

        if draw_histogram := False:
            plt.hist(lane_markers_raw[:, 0], bins=21)
            plt.show(block=True)

        self._maxs, self._mins = {}, {}
        lane_markers_n = np.empty_like(lane_markers_c)
        for f in [0, 1]:
            self._maxs[f] = np.max(lane_markers_c[:, f])
            self._mins[f] = np.min(lane_markers_c[:, f])
            lane_markers_n[:, f] = \
                ((lane_markers_c[:, f] - self._mins[f]) / (self._maxs[f] - self._mins[f]))

        self._kmeans = KMeans(n_clusters=self._num_lane_markers)
        self._kmeans.fit(lane_markers_n.astype(np.float32))

        self._cluster_centers = self._kmeans.cluster_centers_.copy()
        for f in [0, 1]:
            self._cluster_centers[:, f] = (
                    ((self._maxs[f] - self._mins[f]) * self._cluster_centers[:, f]) + self._mins[f]
            )

        # sort by m (reminder our descriptors are y=mx+b) so we can go from left to right
        cluster_centers_not_sorted = \
            {i: (m := self._kmeans.cluster_centers_[i][0]) for i in range(self._num_lane_markers)}
        cluster_centers_sorted = (
            # m =  value[1]
            sorted(cluster_centers_not_sorted.items(), key=lambda value: value[1])
        )
        self._kmeans_to_left_right_lut = \
            {cluster_centers_sorted[i][0]: i for i in range(self._num_lane_markers)}

    def predict_lane_marker(self,
                            line_segment_descriptor_raw: tuple[float, float]
                            ) -> tuple[int, float] or tuple[None, None]:

        line_segment_descriptor_r = np.array(line_segment_descriptor_raw).astype(np.float32)
        line_segment_descriptor_n = np.empty_like(line_segment_descriptor_r).astype(np.float32)

        for f in [0, 1]:
            line_segment_descriptor_n[f] = \
                ((line_segment_descriptor_r[f] - self._mins[f]) / (self._maxs[f] - self._mins[f]))

        try:
            score = self._kmeans.score(np.array([line_segment_descriptor_n]))
            kmeans_classification = self._kmeans.predict(np.array([line_segment_descriptor_n]))[0]
            return self._kmeans_to_left_right_lut[kmeans_classification], abs(score)
        except ValueError:
            return None, None

    def calibrate(self, frame_num: int, lane_markers: NDArray):

        if not self._is_calibrating:
            return
        if frame_num not in self._calibrate_frame_nums:
            return

        # calibrate.
        if self._is_calibrating:
            if self._lane_markers is None:
                self._lane_markers = np.array(lane_markers)
            else:
                self._lane_markers = np.vstack((self._lane_markers, np.array(lane_markers)))

            # save and overwrite (each time we don't know when we're stopping)
            np.save(os.path.splitext(self._file_name)[0], self._lane_markers)

    def left_to_right_order(self, unordered) -> int:
        return self._kmeans_to_left_right_lut[unordered]
