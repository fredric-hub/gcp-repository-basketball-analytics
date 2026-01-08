import os
import torch
import numpy as np
import supervision as sv
from inference import get_model
from sam2.build_sam import build_sam2_camera_predictor

# --- Configuration from Notebook ---
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
PLAYER_CONFIDENCE = 0.4
PLAYER_IOU = 0.9
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7] # Player, possession, jump-shot, etc.

class VideoTrackerService:
    def __init__(self):
        # Load RF-DETR Model
        print("Loading RF-DETR Model...")
        self.detection_model = get_model(model_id=PLAYER_DETECTION_MODEL_ID)
        
        # Load SAM2 Model
        print("Loading SAM2 Model...")
        checkpoint = os.environ.get("SAM2_CHECKPOINT_PATH")
        config = os.environ.get("SAM2_CONFIG_PATH")
        self.predictor = build_sam2_camera_predictor(config, checkpoint)
        self.tracker = SAM2Tracker(self.predictor)

    def process_frame_logic(self, frame_generator):
        """
        Iterates through frames, initializes tracking on the first frame,
        and propagates tracks. Returns a generator yielding (frame_idx, frame, detections).
        """
        # Get first frame
        try:
            first_frame = next(frame_generator)
        except StopIteration:
            return

        # 1. Detect on First Frame (RF-DETR)
        result = self.detection_model.infer(
            first_frame, 
            confidence=PLAYER_CONFIDENCE, 
            iou_threshold=PLAYER_IOU
        )[0]
        detections = sv.Detections.from_inference(result)
        
        # Filter for players
        detections = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]
        detections.tracker_id = np.arange(1, len(detections.class_id) + 1)

        # 2. Prompt SAM2
        self.tracker.prompt_first_frame(first_frame, detections)

        # Yield first frame data
        yield 0, first_frame, detections

        # 3. Propagate Tracking (SAM2)
        for i, frame in enumerate(frame_generator, start=1):
            detections = self.tracker.propagate(frame)
            yield i, frame, detections

# --- The SAM2Tracker Class from Notebook ---
class SAM2Tracker:
    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self._prompted = False

    def prompt_first_frame(self, frame: np.ndarray, detections: sv.Detections) -> None:
        if len(detections) == 0:
            return # Handle empty safely

        if detections.tracker_id is None:
            detections.tracker_id = list(range(1, len(detections) + 1))

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.load_first_frame(frame)
            for xyxy, obj_id in zip(detections.xyxy, detections.tracker_id):
                bbox = np.asarray([xyxy], dtype=np.float32)
                self.predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=int(obj_id),
                    bbox=bbox,
                )
        self._prompted = True

    def propagate(self, frame: np.ndarray) -> sv.Detections:
        if not self._prompted:
            raise RuntimeError("Call prompt_first_frame before propagate")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tracker_ids, mask_logits = self.predictor.track(frame)

        tracker_ids = np.asarray(tracker_ids, dtype=np.int32)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)

        if masks.ndim == 2:
            masks = masks[None, ...]
            
        #
        masks = np.array([
            sv.filter_segments_by_distance(mask, relative_distance=0.03, mode="edge")
            for mask in masks
        ])

        xyxy = sv.mask_to_xyxy(masks=masks)
        detections = sv.Detections(xyxy=xyxy, mask=masks, tracker_id=tracker_ids)
        return detections