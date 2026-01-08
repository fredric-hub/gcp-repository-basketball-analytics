import os
import json
import base64
import cv2
import supervision as sv
from flask import Flask, request
from google.cloud import storage, pubsub_v1
from core import VideoTrackerService

app = Flask(__name__)

# Initialize Cloud Clients
storage_client = storage.Client()
publisher = pubsub_v1.PublisherClient()
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
NEXT_TOPIC_ID = "process-identities" # Topic for the next service
TOPIC_PATH = publisher.topic_path(PROJECT_ID, NEXT_TOPIC_ID)
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")

# Initialize Models Once (Global State)
tracker_service = VideoTrackerService()

@app.route("/", methods=["POST"])
def index():
    envelope = request.get_json()
    if not envelope:
        return "No Pub/Sub message received", 400

    # Parse Pub/Sub message
    if 'message' in envelope:
        msg_data = base64.b64decode(envelope['message']['data']).decode('utf-8')
        data = json.loads(msg_data)
        video_path_gcs = data.get("video_path") # e.g., "source/game1.mp4"
        video_id = data.get("video_id", "video_001")
    else:
        return "Invalid message format", 400

    print(f"Processing video: {video_path_gcs}")

    # 1. Download Video (or use /mnt/gcs if using Cloud Run volume mounting)
    local_video_path = "/tmp/input_video.mp4"
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(video_path_gcs)
    blob.download_to_filename(local_video_path)

    # 2. Process Video
    frame_generator = sv.get_video_frames_generator(local_video_path)
    
    # Batch data for the next service
    crop_batch = []
    
    # Run the core logic from src/core.py
    for frame_idx, frame, detections in tracker_service.process_frame_logic(frame_generator):
        
        # LOGIC: Extract crops for identification every 30 frames (approx 1 sec)
        # This prevents overloading the ID service with redundant data
        if frame_idx % 30 == 0:
            for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
                # Crop image
                crop = sv.crop_image(frame, xyxy)
                
                # Save crop to GCS
                crop_filename = f"crops/{video_id}/{track_id}_{frame_idx}.jpg"
                # (Ideally, write to /tmp and upload async, or use GCS fuse mount)
                temp_crop_path = f"/tmp/{track_id}_{frame_idx}.jpg"
                cv2.imwrite(temp_crop_path, crop)
                
                output_blob = bucket.blob(crop_filename)
                output_blob.upload_from_filename(temp_crop_path)
                
                crop_batch.append({
                    "gcs_path": crop_filename,
                    "track_id": int(track_id),
                    "frame_idx": frame_idx
                })

            # 3. Trigger Next Service (Identifier)
            if crop_batch:
                message_json = json.dumps({"video_id": video_id, "crops": crop_batch})
                publisher.publish(TOPIC_PATH, message_json.encode("utf-8"))
                crop_batch = [] # Reset batch

    return "Video Processed Successfully", 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))