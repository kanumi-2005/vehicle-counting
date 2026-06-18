from ultralytics import YOLO
import supervision as sv
import torch
from trackers import ByteTrackTracker, SORTTracker


TRACKER_CLASSES = {
    "byte": ByteTrackTracker,
    "sort": SORTTracker,
}


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class TrackingPipeline:

    def __init__(
        self,
        model_path,
        confidence_threshold=0.2,
        device="cpu",
        verbose=False,
        tracker_type="byte",
        lost_track_buffer=75,
        frame_rate=25,
        track_activation_threshold=0.572,
        minimum_consecutive_frames=2,
        minimum_iou_threshold=0.1,
        high_conf_detection_threshold=0.572,
    ):

        self.verbose = verbose
        self.device = resolve_device(device)
        # YOLO detector
        self.model = YOLO(model_path, task="detect", verbose=verbose)

        tracker_type = tracker_type.lower()
        if tracker_type not in TRACKER_CLASSES:
            supported_trackers = ", ".join(sorted(TRACKER_CLASSES))
            raise ValueError(
                f"Unsupported tracker_type '{tracker_type}'. "
                f"Supported values: {supported_trackers}"
            )

        tracker_kwargs = {
            "lost_track_buffer": lost_track_buffer,
            "frame_rate": frame_rate,
            "track_activation_threshold": track_activation_threshold,
            "minimum_consecutive_frames": minimum_consecutive_frames,
            "minimum_iou_threshold": minimum_iou_threshold,
        }
        if tracker_type == "byte":
            tracker_kwargs[
                "high_conf_det_threshold"
            ] = high_conf_detection_threshold

        self.tracker = TRACKER_CLASSES[tracker_type](**tracker_kwargs)

        self.confidence_threshold = confidence_threshold

        # save MOT results
        self.results = []

    # ==================================
    # DETECT
    # ==================================
    def detect(self, frame):

        result = self.model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
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
