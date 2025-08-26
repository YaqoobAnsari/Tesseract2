
import sys 

import numpy as np
import cv2
import os
from datetime import datetime
sys.path.append("/data1/yansari/cad2map/Yaqoob_CAD2MAP/utils")  # Add the utils folder to the Python path
 
from PIL import Image

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
 
device = torch.device("cuda" 
    if torch.cuda.is_available() else "mps" 
    if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else "cpu"
)

"""
applies transformations for image to be passed into detection model

parameters
- img: image to pass to model, cv2 image
    (required)

returns 
    transformed image, numpy array
"""
def img_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img).permute(2,0,1)
    return img

"""
Infernece of a single input image

parameters
- img: input-image as torch.tensor (shape: [C, H, W])
    (required)
- model: model for infernce (torch.nn.Module)
    (required)
- detection_threshold: Confidence-threshold for NMS 
    (default 0.7)

returns
    boxes: bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
    labels: class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
    scores: confidence-score (Format [N] => N times confidence-score between 0 and 1)
"""
def inference(img, model, detection_threshold=0.00):
    model.eval()

    img = img.to(device)
    outputs = model([img])
    
    boxes = outputs[0]['boxes'].data.cpu().numpy()
    scores = outputs[0]['scores'].data.cpu().numpy()
    labels = outputs[0]['labels'].data.cpu().numpy()

    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    labels = labels[scores >= detection_threshold]
    scores = scores[scores >= detection_threshold]

    return boxes, scores, labels

"""
Function that draws the BBoxes, scores, and labels on the image.

inputs:
  img: input-image as numpy.array (shape: [H, W, C])
  boxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
  scores: list of conf-scores (Format [N] => N times confidence-score between 0 and 1)
  labels: list of class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
  dataset: list of all classes e.g. ["background", "class1", "class2", ..., "classN"] => Format [N_classes]
"""
def plot_image(img, boxes, scores, labels, dataset, save_path=None):
    cmap = plt.get_cmap("tab20b")
    class_labels = np.array(dataset)

    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    height, width, _ = img.shape
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(16, 8))
    
    # Display the image
    ax.imshow(img)
    
    for i, box in enumerate(boxes):
      class_pred = labels[i]
      conf = scores[i]
      
      width = box[2] - box[0]
      height = box[3] - box[1]
      
      rect = patches.Rectangle(
          (box[0], box[1]),
          width,
          height,
          linewidth=2,
          edgecolor=colors[int(class_pred)],
          facecolor="none",
      )
      
      # Add the patch to the Axes
      ax.add_patch(rect)
      plt.text(
          box[0], box[1],
          s=class_labels[int(class_pred)] + " " + str(int(100*conf)) + "%",
          color="white",
          verticalalignment="top",
          bbox={"color": colors[int(class_pred)], "pad": 0},
      )

    # Used to save inference phase results
    if save_path is not None:
      plt.savefig(save_path)

    plt.show()

"""
sets up detection model to perform bounding box detection with

parameters
- weights_pth: location to pretrained model weights, str
    (required)

returns
    FasterRCNN model, from pytorch
"""
def detection_model(weights_pth):
    num_classes = 2  # 1 classes + background

    # set up FasterRCNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(min_size=300,
        max_size=480, weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # load pretrained weights
    checkpoint = torch.load(weights_pth, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

import cv2
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
 
def detect_doors(
    image_path,
    threshold=0.8,
    chunk_size=300,
    overlap=50,
    results_dir="/data1/yansari/cad2map/Yaqoob_CAD2MAP/Results/Plots", 
):
    """
    Detects doors in a large image by chunking it into smaller patches for Faster R-CNN inference,
    writes the results to a text file, and saves an image with plotted bounding boxes.

    Args:
        image_path (str): Path to the input image. 
        threshold (float): Confidence threshold for valid detections.
        chunk_size (int): Size of each chunk (both width and height).
        overlap (int): Overlap between adjacent chunks.
        results_dir (string): Path to save outputs

    Returns:
        None
    """
    # Step 1: Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    original_height, original_width = img.shape[:2]

    #print(f"       Original image dimensions: {original_width}x{original_height}")

    door_detect_dir = os.path.join(results_dir, "door_detect")
    os.makedirs(door_detect_dir, exist_ok=True)

    image_name_no_ext = os.path.splitext(os.path.basename(image_path))[0] 
    img_door_dir = os.path.join(door_detect_dir, f"{image_name_no_ext}")
    os.makedirs(img_door_dir, exist_ok=True)

    # Step 2: Divide the image into chunks
    chunks = []
    for y in range(0, original_height, chunk_size - overlap):
        for x in range(0, original_width, chunk_size - overlap):
            chunk = img[y : y + chunk_size, x : x + chunk_size]
            chunks.append((chunk, x, y))  # Save chunk and top-left corner coordinates

    #print(f"       Number of chunks created: {len(chunks)}")

    door_model_path = "/data1/yansari/cad2map/Yaqoob_CAD2MAP/Model_weights/door_mdl_32.pth"
    door_model = detection_model(door_model_path)

    # Step 3: Run inference on each chunk
    all_boxes, all_scores, all_labels = [], [], []
    for i, (chunk, x_offset, y_offset) in enumerate(chunks):
        #print(f"Processing chunk {i+1} at position ({x_offset}, {y_offset})")

        # Transform and run inference
        transformed_chunk = img_transform(chunk)
        boxes, scores, labels = inference(transformed_chunk, door_model, detection_threshold=threshold)

        # Adjust bounding boxes back to original image coordinates
        for box, score, label in zip(boxes, scores, labels):
            adjusted_box = [
                box[0] + x_offset,
                box[1] + y_offset,
                box[2] + x_offset,
                box[3] + y_offset,
            ]
            all_boxes.append(adjusted_box)
            all_scores.append(score)
            all_labels.append(label)

            # Debug: Print bounding box adjustment
            #print(f"Chunk {i+1} detection: Box {box} (adjusted to {adjusted_box}), Score: {score:.2f}, Label: {label}")

    # Step 4: Save results to a text file
    output_txt_path = os.path.join(img_door_dir, f"{image_name_no_ext}_door_bbox.txt")
    with open(output_txt_path, "w") as f:
        for i, (box, score, label) in enumerate(zip(all_boxes, all_scores, all_labels)):
            f.write(
                f"Detection {i+1}:\n"
                f"  Bounding Box: {box}\n"
                f"  Confidence Score: {score:.2f}\n"
                f"  Label: {label}\n\n"
            )
    print(f"Results saved to {output_txt_path}")

    # Step 5: Plot bounding boxes on the original image 
    plot_bounding_boxes(img, all_boxes, all_scores, all_labels, img_door_dir, image_name_no_ext) 
    
    print(f"{len(all_boxes)} door bbox returned.")
    return all_boxes

def plot_bounding_boxes(image, boxes, scores, labels, output_dir, image_name_no_ext):
    """
    Plots bounding boxes on the image and saves two versions: one with only bounding boxes,
    and another with bounding boxes and text. Bounding boxes are red, and labels are white
    text with a deep orange background.

    Args:
        image (np.ndarray): Original image.
        boxes (list): List of bounding boxes (each as [xmin, ymin, xmax, ymax]).
        scores (list): List of confidence scores for each bounding box.
        labels (list): List of labels for each bounding box.
        output_dir (str): Directory to save the images.

    Returns:
        None
    """
    # Convert BGR to RGB for plotting
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define red color for bounding boxes
    box_color = "#FF0000"  # Red
    label_bg_color = "#FF5722"  # Deep orange for label background
    label_text_color = "white"  # White text

    # Plot with only bounding boxes
    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(image_rgb)

    for box in boxes:
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor=box_color,
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.axis("off")
    plt.tight_layout()
    bbox_only_path = os.path.join(output_dir, f"{image_name_no_ext}_door_bbox_only.png")
    plt.savefig(bbox_only_path, dpi=300)
    plt.close(fig)
    print(f"Door bounding boxes plot saved to {bbox_only_path}")

    # Plot with bounding boxes and text
    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(image_rgb)

    for box, score, label in zip(boxes, scores, labels):
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor=box_color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            box[0],
            box[1] - 10,
            f"Label: {label}, Score: {score:.2f}",
            color=label_text_color,
            fontsize=12,
            bbox=dict(facecolor=label_bg_color, alpha=0.8),
        )

    plt.axis("off")
    plt.tight_layout()
    bbox_with_text_path = os.path.join(output_dir, f"{image_name_no_ext}_door_bbox_with_text.png")
    plt.savefig(bbox_with_text_path, dpi=300)
    plt.close(fig)
    print(f"Door bounding boxes plot with text saved to {bbox_with_text_path}")

def refine_door_bboxes(image_path, results_dir, door_threshold=20, door_bboxes=None):
    """
    Refines the list of door bounding boxes by merging overlapping ones or those whose centers
    are within a given threshold distance.

    Args:
        image_path (str): Path to the image where the bboxes will be drawn.
        results_dir (str): Directory to save the results.
        door_threshold (int): Threshold for detecting overlap and merging bounding boxes.
        door_bboxes (list): List of door bounding boxes in the form [x1, y1, x2, y2].
        
    Returns:
        list: List of refined bounding boxes after merging.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    door_detect_dir = os.path.join(results_dir, "door_detect")
    os.makedirs(door_detect_dir, exist_ok=True)

    image_name_no_ext = os.path.splitext(os.path.basename(image_path))[0] 
    img_door_dir = os.path.join(door_detect_dir, f"{image_name_no_ext}")
    os.makedirs(img_door_dir, exist_ok=True)

    print(f"Initial number of door bounding boxes: {len(door_bboxes)}")

    # Function to check if two bboxes overlap
    def is_overlap(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

    # Function to calculate the center of a bbox
    def get_center(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    # Function to calculate the Euclidean distance between two points
    def distance(center1, center2):
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    # Merge overlapping bounding boxes or those whose centers are close within the threshold
    merged_bboxes = []
    while door_bboxes:
        current_bbox = door_bboxes.pop(0)
        x1, y1, x2, y2 = current_bbox
        current_center = get_center(current_bbox)
        merged = False

        # Check if it overlaps with any of the existing merged bboxes or if the centers are close
        for i, (bx1, by1, bx2, by2) in enumerate(merged_bboxes):
            existing_center = get_center([bx1, by1, bx2, by2])
            if is_overlap([bx1, by1, bx2, by2], [x1, y1, x2, y2]) or distance(current_center, existing_center) < door_threshold:
                # Merge the bounding boxes
                new_bbox = [min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2)]
                merged_bboxes[i] = new_bbox
                merged = True
                break

        if not merged:
            merged_bboxes.append([x1, y1, x2, y2])

    print(f"Final number of door bounding boxes after merging: {len(merged_bboxes)}")

    # Draw the merged bounding boxes on the image
    for bbox in merged_bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red color for bounding boxes

    # Save the result using OpenCV
    save_path = os.path.join(img_door_dir, f"{image_name_no_ext}_refined_door_bboxes.png")
    cv2.imwrite(save_path, image)
    print(f"Refined doors saved at: {save_path}")

    # Return the merged bounding boxes
    return merged_bboxes

