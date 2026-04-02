import os
import cv2
import torch
import numpy as np
import supervision as sv
import argparse

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.supervision_utils import CUSTOM_COLOR_MAP
from supervision.draw.color import ColorPalette

"""
Custom args for make debugging fast
"""
parser = argparse.ArgumentParser()
parser.add_argument("--video-path", default="/content/drive/MyDrive/warehouse_videos/safe_trolley.mp4")
parser.add_argument("--text-prompt", default="person.")
parser.add_argument("--frame-start", type=int, default=0)
parser.add_argument("--output-dir", default="./isolated_dino_output")
args = parser.parse_args()

"""
Hyperparam for Ground and Tracking
"""
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
VIDEO_PATH = args.video_path
TEXT_PROMPT = args.text_prompt
FRAME_START = args.frame_start
SOURCE_VIDEO_FRAME_DIR = "./custom_video_frames"
OUTPUT_PATH = args.output_dir

"""
Step 1: Environment settings and model initialization for SAM 2
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# build grounding dino from huggingface
model_id = MODEL_ID
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

"""
Custom video input directly using video files
"""
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
print(video_info)
# frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)
# using more stride to extract fewer slices, and starting at slice 25 to get clearer picture of the object
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=3, start=FRAME_START, end=None)  

# saving video to frames
source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames, 
    overwrite=True, 
    image_name_pattern="{:05d}.jpg"
) as sink:
    # for frame in tqdm(frame_generator, desc="Saving Video Frames"):
    #     sink.save_image(frame)
    for frame in tqdm(frame_generator, desc="Saving Video Frames with Higher Stride and Lower Res"):
        frame = cv2.resize(frame, (640, 360))  # to 640x360
        sink.save_image(frame)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# NOTE: even though we intend to not start at the first frame, this is already the cropped version 
# of the video, if we want to adjust where should the video start, head to the frame_generator 
# variable above
ann_frame_idx = 0  # the frame index we interact with (start at the first frame)
"""
Step 2: Prompt Grounding DINO 1.0 with HF model for box coordinates
"""

# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
image = Image.open(img_path)
inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

input_boxes = results[0]["boxes"].cpu().numpy()
confidences = results[0]["scores"].cpu().numpy().tolist()
class_names = results[0]["labels"]

print(input_boxes)
print(confidences)
print(class_names)


"""
Step 3: Visualize the detection result along with the original image
"""
original_img = cv2.imread(img_path)

# construct ids: [0, 1, ... N] as long as the number of detected classes
class_ids = np.arange(list(class_names))

# build labels alongside that, we map the class name with the confidence score for better visualization
# example item: "car 0.80"
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence in zip(class_names, confidences)
]

# build the detections model from HF supervision
detections = sv.Detections(
    xyxy=input_boxes,
    class_id=class_ids
)

# combine the original image with the detection results, and visualize the bounding boxes with labels
box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))

# build the annotated frame (in this case it's just a slice)
annotated_frame = box_annotator.annotate(scene=original_img.copy(), detections=detections)

# build the label annotator to visualize the labels
label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

# save the result as image
os.makedirs(OUTPUT_PATH, exist_ok=True)
cv2.imwrite(os.path.join(OUTPUT_PATH, "groundingdino_annotated_image.jpg"), annotated_frame)