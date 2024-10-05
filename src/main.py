import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from model.unet_model_mobilenetv2_pix2pix import unet_model
import os
import argparse
from collections import defaultdict

from utils.video_stream import VideoStream
from utils_lane_detection.calibrate import Calibrate
from utils_lane_detection.post_process import Postprocess
from utils_lane_detection.pre_process import Preprocess
from utils_lane_detection.display import Display
from utils_lane_detection.alarm_handler import AlarmHandler
from utils.cv_utils import cv_to_tf


def contour_to_line_segments(skeleton: NDArray) -> \
        (list[tuple[tuple[float, float], tuple[float, float]],
         tuple[tuple[float, float], tuple[float, float]]],
         list[tuple[float, float]]):
    contours, _ = cv.findContours(skeleton,
                                  cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if len(contour) > 16]

    line_segments = []
    line_segment_descriptors = []

    for contour in contours:
        rect = cv.minAreaRect(contour)
        rect = cv.boxPoints(rect)

        # swap such that x=y and y=x (so vertical lines aren't possible)
        # (as such the try catch below should occur infrequently if at all,
        # and it likely means the data is best ignored)
        y2 = rect[2][0]
        y1 = rect[0][0]
        x2 = rect[2][1]
        x1 = rect[0][1]
        line_seg = (y1, x1), (y2, x2)
        try:
            m = (y2 - y1) / (x2 - x1)
        except ZeroDivisionError:
            continue
        b = y1 - m * x1
        line_segments.append(line_seg)
        line_segment_descriptors.append(np.array([m, b]))

    return line_segments, line_segment_descriptors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-source', type=str, default=None)
    parser.add_argument('--record-name', type=str, default=None)
    parser.add_argument('--kmeans-score', type=float, default=.005)
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--calibrate', type=bool, default=None)
    # these frames work well with TuSimple-0313-2.mp4. TODO, fix so can read
    #  with argparse
    parser.add_argument('--calibrate-frames', type=list,
                        default=[[0, 75], [250, 500]])
    args = parser.parse_args()

    root_dir = os.getcwd() if args.root_dir is None else args.root_dir

    if not os.path.isdir(root_dir):
        print(f"Unable to find {root_dir}.")
        exit(-1)

    # ####################################
    # the NN has been trained to find three lanes (left, center, right)
    # and thus 4 lane lines
    MODEL_LANE_MARKERS = 4
    MODEL_SIZE_XY = (224, 224)

    #####################################
    # open the video stream
    video_stream = VideoStream(args.video_source)
    if not video_stream.is_open():
        print(f"Check {args.video_source}. Exiting.")
        return -1

    #####################################
    # get the calibration running
    calibrate = Calibrate(MODEL_LANE_MARKERS,
                          args.calibrate,
                          args.calibrate_frames)

    #####################################
    # get the model running
    model_dir = os.path.join(root_dir, 'trained_models', 'lanedetection')
    os.makedirs(model_dir, exist_ok=True)
    model = unet_model(2)
    model.build(input_shape=(None, MODEL_SIZE_XY[1], MODEL_SIZE_XY[0], 3))
    model.summary()

    weights_file_name = os.path.join(model_dir,
                                     'checkpoints',
                                     'epoch_20-val_loss_0.06.tf')
    if (weights_file_name is not None and
            os.path.isfile(weights_file_name + '.index')):
        model.load_weights(weights_file_name).expect_partial()
        print(f"Loaded weights file: {weights_file_name}")

    #####################################
    # get the alarm handler running
    alarm_handler = AlarmHandler()

    #####################################
    # set the preprocess (which cleans up the mask from the NN)
    preprocess = Preprocess()

    # ####################################
    # set the post process (which adds
    # life-times and kalman filters the lane descriptors)
    postprocess = Postprocess(MODEL_LANE_MARKERS)

    #####################################
    # initialize the display
    frame_size_xy = video_stream.size_xy()
    nn_size_xy = (MODEL_SIZE_XY[0], MODEL_SIZE_XY[1])
    markup_draw_size_xy = round(frame_size_xy[0] * .20), round(
        frame_size_xy[1] * .20)
    frame_draw_size_xy = round(frame_size_xy[0] * .41), round(
        frame_size_xy[1] * .41)
    display = Display(nn_size_xy, markup_draw_size_xy, frame_draw_size_xy,
                      args.record_name)

    ######################
    # OK time to get on with it
    ######################

    while (frame := video_stream.next_frame()) is not None:

        # basic frame handling
        frame_num = frame[1]
        frame = frame[0]
        display.set_frame(frame)

        # handle any alarms
        alarm_handler.reset_if(frame_num)
        if not alarm_handler.alarm(frame_num):
            display.clear_all_alarms()

        # predict the lane markers with the NN
        frame = cv_to_tf(frame, MODEL_SIZE_XY)
        prediction = model.predict(frame)[0]
        lane_marker_mask = np.argmax(prediction, axis=-1).astype(np.uint8)
        lane_marker_mask = np.expand_dims(lane_marker_mask, axis=-1) * 255
        display.set_nn_draw(lane_marker_mask)

        # cleanup
        lane_marker_mask = preprocess.cleanup(lane_marker_mask)
        display.set_skeleton_draw(lane_marker_mask)

        # fine line segments and their descriptors (just m, b)
        line_segments, line_segment_descriptors = \
            contour_to_line_segments(lane_marker_mask)
        display.set_line_segments_draw(line_segments)

        # cluster line segments into lane markers
        line_segment_descriptors = \
            np.array(line_segment_descriptors).astype(np.float32)

        # calibrate ...
        valid_lane_markers = defaultdict(lambda: list())
        if calibrate.is_calibrating():
            calibrate.calibrate(frame_num, line_segment_descriptors)

        # ... or predict the lane markers
        else:
            for line_segment_descriptor in line_segment_descriptors:
                lane_marker, score = calibrate.predict_lane_marker(
                    line_segment_descriptor)
                if score is not None and score < args.kmeans_score:
                    valid_lane_markers[lane_marker].append(
                        line_segment_descriptor)

        # filter (and kalman filter predict if anything missing)
        if not calibrate.is_calibrating():
            ordered_lanes = postprocess.filter(valid_lane_markers)
        else:
            ordered_lanes = {}
        display.set_lane_markers_draw(ordered_lanes)

        # now finally check to see if we're crossing a lane
        for lane, descriptor in ordered_lanes.items():
            m = descriptor[0]
            b = descriptor[1]
            if alarm_handler.set_if(lane, m, frame_num):
                display.append_alarm(lane)

        # and draw it all
        display.display_and_record_if(frame_num, ordered_lanes)


if __name__ == '__main__':
    main()
