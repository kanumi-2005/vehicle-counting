from ultralytics import YOLO
import supervision as sv
from trackers import ByteTrackTracker


class TrackingPipeline:

    def __init__(
        self,
        model_path,
        conf
    ):

        # YOLO detector
        self.model = YOLO(model_path)

        # ByteTrack
        self.tracker = ByteTrackTracker()

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
            verbose=False
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
