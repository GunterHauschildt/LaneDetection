import cv2 as cv
import numpy as np
from numpy.typing import NDArray


class Display:

    @staticmethod
    def draw_descriptors_as_lines(descriptor: tuple[float, float],
                                  draw: NDArray, color: tuple[int, int, int] = (255, 255, 255),
                                  thickness: int = 2) -> NDArray:
        x0 = 100
        x1 = 240
        m = descriptor[0]
        b = descriptor[1]
        y0 = (m * x0) + b
        y1 = (m * x1) + b
        pt0 = (round(y0), round(x0))
        pt1 = (round(y1), round(x1))
        return cv.line(draw, pt0, pt1, color, thickness, cv.LINE_AA)

    def __init__(self,
                 nn_size_xy: tuple[int, int],
                 markup_size_xy: tuple[int, int],
                 full_size_xy: tuple[int, int],
                 record_name: str or None = None):
        self._markup_size_xy = markup_size_xy
        self._full_size_xy = full_size_xy
        self._nn_size_xy = nn_size_xy
        self._display_size_xy = 2 + self._markup_size_xy[0] * 2 + self._full_size_xy[0], \
            self._full_size_xy[1]

        self._nn_draw = None
        self._preprocess_draw = None
        self._line_segments_draw = None
        self._lane_markers_draw = None
        self._frame = None
        self._frame_markup_dark = None
        self._frame_markup = None
        self._alarm_draw = None
        self._alarms = []
        self._video_writer = None
        self._final_draw = None

        if record_name is not None:
            self._video_writer = cv.VideoWriter(record_name,
                                                cv.VideoWriter.fourcc(*'mp4v'),
                                                10,
                                                self._display_size_xy)

    def __del__(self):
        if self._video_writer is not None and self._video_writer.isOpened():
            self._video_writer.release()

    def _record_if(self, m: NDArray):
        if self._video_writer is not None and self._video_writer.isOpened():
            self._video_writer.write(m)

    def set_frame(self, frame: NDArray):
        self._frame = cv.resize(frame, self._full_size_xy)
        self._frame_markup = cv.resize(frame, self._markup_size_xy)
        self._frame_markup_dark = (cv.resize(frame, self._markup_size_xy) * .66).astype(np.uint8)

    def format_markup(self, m: NDArray) -> NDArray:
        m = cv.resize(m, self._markup_size_xy).astype(np.uint8)
        if len(m.shape) == 2 or (len(m.shape) == 3 and m.shape[2] == 1):
            m = cv.cvtColor(m, cv.COLOR_GRAY2BGR)
        return m

    def set_nn_draw(self, nn: NDArray):
        self._nn_draw = self.format_markup(nn)

    def set_skeleton_draw(self, skeleton: NDArray):
        self._preprocess_draw = self.format_markup(skeleton)

    def set_line_segments_draw(self, line_segments: list[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]]]):

        # TODO: for efficiency i should really resize the segments not the image
        line_segments_draw = np.zeros((self._nn_size_xy[1], self._nn_size_xy[0], 3)).astype(
            np.uint8)
        for line_segment in line_segments:
            pt1 = round(line_segment[0][0]), round(line_segment[0][1])
            pt2 = round(line_segment[1][0]), round(line_segment[1][1])
            line_segments_draw = cv.line(
                line_segments_draw, pt1, pt2, (255, 255, 255), 3, cv.LINE_AA
            )
        self._line_segments_draw = cv.resize(
            line_segments_draw, self._markup_size_xy, interpolation=cv.INTER_NEAREST
        )

    def set_lane_markers_draw(self, lane_descriptors: dict[int, tuple[float, float]]):
        # all white for now
        colors = {
            0: (255, 255, 255),
            1: (255, 255, 255),
            2: (255, 255, 255),
            3: (255, 255, 255)
        }
        lane_markers_draw = np.zeros((self._nn_size_xy[1], self._nn_size_xy[0], 3)).astype(np.uint8)
        for lane, descriptor in lane_descriptors.items():
            lane_markers_draw = Display.draw_descriptors_as_lines(descriptor, lane_markers_draw,
                                                                  colors[lane])
        self._lane_markers_draw = cv.resize(lane_markers_draw, self._markup_size_xy,
                                            interpolation=cv.INTER_NEAREST)

    def set_final_draw(self, lane_descriptors):
        draw = np.zeros((self._nn_size_xy[1], self._nn_size_xy[0], 3)).astype(np.uint8)
        for lane in [0, 1, 2, 3]:
            if lane in self._alarms:
                color, thickness = (0, 0, 255), 3
            else:
                color, thickness = (0, 255, 255), 2
            if lane in lane_descriptors:
                draw = Display.draw_descriptors_as_lines(
                    lane_descriptors[lane],
                    draw, color, thickness
                )
        self._final_draw = cv.resize(draw, self._full_size_xy, interpolation=cv.INTER_NEAREST)

    def append_alarm(self, lane: int):
        self._alarms.append(lane)

    def clear_all_alarms(self):
        self._alarms = []

    def display_and_record_if(self, frame_num, lane_descriptors):

        nn_draw = cv.addWeighted(self._nn_draw, 1,
                                 self._frame_markup_dark, 1.0, 0.0)
        nn_draw = cv.putText(nn_draw,
                             "(1) Neural Network", (10, 15),
                             cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))

        preprocess_draw = cv.addWeighted(self._preprocess_draw, 1.0,
                                         self._frame_markup_dark, 1.0, 0.0)
        preprocess_draw = cv.putText(preprocess_draw,
                                     "(2) PreProcess", (10, 15),
                                     cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))

        line_segments_draw = cv.addWeighted(self._line_segments_draw, 1.0,
                                            self._frame_markup_dark, 1.0, 0.0)
        line_segments_draw = cv.putText(line_segments_draw,
                                        "(3) Line Segments", (10, 15),
                                        cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))

        # save me, we can put lane_markers on one of the displays as well
        # lane_markers_draw = cv.addWeighted(self._lane_markers_draw, 1.0,
        #                                    self._frame_markup, 1.0, 0.0)
        # lane_markers_draw = cv.putText(lane_markers_draw,
        #                                "(4) Post: Ordered Markers", (10, 15),
        #                                cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))

        self.set_final_draw(lane_descriptors)
        if self._final_draw is not None:
            if len(self._alarms):
                self._final_draw = cv.putText(
                    self._final_draw,
                    "LANE DEPARTURE",
                    (self._final_draw.shape[1] // 3, self._final_draw.shape[0] // 2),
                    cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255)
                )
                alpha, beta = .75, .66
            else:
                alpha, beta = .25, 1.0
            frame_full_draw = cv.addWeighted(self._final_draw, alpha, self._frame, beta, 0.0)
        else:
            frame_full_draw = self._frame.copy()
        frame_full_draw = cv.putText(frame_full_draw,
                                     f"(4) Final Ordered Lane Descriptors",
                                     (10, 15), cv.FONT_HERSHEY_PLAIN, 1.0,
                                     (255, 255, 255))

        def markup_position(r, c):
            r0 = r + self._markup_size_xy[1] * r
            r1 = r0 + self._markup_size_xy[1]
            c0 = c + self._markup_size_xy[0] * c
            c1 = c0 + self._markup_size_xy[0]
            return r0, r1, c0, c1

        def frame_position(r, c):
            r0 = r + self._markup_size_xy[1] * r
            r1 = r0 + self._full_size_xy[1]
            c0 = c + self._markup_size_xy[0] * c
            c1 = c0 + self._full_size_xy[0]
            return r0, r1, c0, c1

        display = np.zeros((self._display_size_xy[1], self._display_size_xy[0], 3)).astype(np.uint8)

        r0, r1, c0, c1 = markup_position(0, 0)
        display[r0:r1, c0:c1] = self._frame_markup

        r0, r1, c0, c1 = markup_position(0, 1)
        display[r0:r1, c0:c1] = nn_draw  # skeleton_draw

        r0, r1, c0, c1 = markup_position(1, 0)
        display[r0:r1, c0:c1] = preprocess_draw  # line_segments_draw

        r0, r1, c0, c1 = markup_position(1, 1)
        display[r0:r1, c0:c1] = line_segments_draw

        r0, r1, c0, c1 = frame_position(0, 2)
        display[r0:r1, c0:c1] = frame_full_draw

        self._record_if(display)
        cv.imshow(f"Lane Detection", display)
        cv.waitKey(1)
