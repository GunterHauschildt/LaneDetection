import math


class AlarmHandler:
    def __init__(self):
        self._frame_on = -1
        self._frame_off = -math.inf
        self._frame_reset = -math.inf
        self._off_frames = 75
        self._reset_frames = 500

        # hard coded for now. I only have data for one crossing (going left across lane 1).
        self._alarms = {
            1: (-0.5, -math.inf)
        }

    def set_if(self, lane: int, m: float, frame: int) -> bool:
        if lane in self.alarms():
            if m > self._alarms[lane][0] and frame > self._frame_reset:
                self._frame_on = frame
                self._frame_off = frame + self._off_frames
                self._frame_reset = frame + self._reset_frames
                return True
        return False

    def reset_if(self, frame: int):
        if frame > self._frame_reset:
            self._frame_on = -1
            self._frame_off = -math.inf
            self._frame_reset = -math.inf

    def alarm(self, frame: int) -> bool:
        return frame < self._frame_off

    def alarms(self) -> dict[int, tuple[float, float]]:
        return self._alarms

