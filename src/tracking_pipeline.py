from torch import device
from torch import device
from ultralytics import YOLO
import supervision as sv
from trackers import ByteTrackTracker


class TrackingPipeline:

    def __init__(
        self,
        model_path,
        conf=0.2,
        lost_track_buffer=75,
        frame_rate=25,
        track_activation_threshold=0.4,
        minimum_consecutive_frames=2,
        minimum_iou_threshold=0.1,
        high_conf_det_threshold=0.572,
        device="cpu",
        verbose=False
    ):

        self.verbose = verbose
        # YOLO detector
        self.model = YOLO(model_path, task="detect", verbose=verbose)

        # ByteTrack
        self.tracker = ByteTrackTracker(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
            high_conf_det_threshold=high_conf_det_threshold
        )

        self.conf = conf

        # save MOT results
        self.results = []

    # ==================================
    # DETECT
    # ==================================
    def detect(self, frame):

        result = self.model(
            frame,
            conf=self.conf,
            device="cpu",
            verbose=self.verbose
        )[0]

        detections = sv.Detections.from_ultralytics(result)

        return detections

    # ==================================
    # TRACK
    # ==================================
    def track(self, detections):

        tracked = self.tracker.update(
            detections
        )

        return tracked
