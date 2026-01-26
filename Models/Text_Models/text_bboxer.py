import os
import torch
import cv2
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
import sys

sys.path.append("/data1/yansari/cad2map/Tesseract++/Models/Text_Models")
# Import utility modules (ensure these are available in your PYTHONPATH or adjust imports)
import craft_utils
import imgproc
import file_utils
from craft import CRAFT

# Helper function to load model weights
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def get_Textboxes(input_image_path, model_weights_dir, output_directory):
    """
    Perform text detection and return the path to the text file containing bounding box coordinates.

    Args:
        input_image_path (str): Path to the input image.
        model_checkpoint (str): Path to the model checkpoint file.
        output_directory (str): Path to the directory where results should be saved.

    Returns:
        str: Path to the text file containing bounding box coordinates.
    """
    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    model_checkpoint = os.path.join(model_weights_dir, "craft_mlt_25k.pth")
    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"Error: Model checkpoint not found at '{model_checkpoint}'.")

    #print(f"CUDA available: {cuda_available}")

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Derive image-specific folder
    image_name_no_ext = os.path.splitext(os.path.basename(input_image_path))[0]
    image_specific_folder = os.path.join(output_directory, image_name_no_ext)
    os.makedirs(image_specific_folder, exist_ok=True)
    
    # Load the input image
    image = imgproc.loadImage(input_image_path)

    # Load the CRAFT model 
    net = CRAFT()
    if cuda_available:
        net.load_state_dict(copyStateDict(torch.load(model_checkpoint, weights_only=False)))
        net = net.cuda()
        net = torch.nn.DataParallel(net)
    else:
        net.load_state_dict(copyStateDict(torch.load(model_checkpoint, map_location="cpu", weights_only=False)))
    net.eval()

    # Preprocess the image 
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
    )
    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if cuda_available:
        x = x.cuda()

    # Forward pass
    with torch.no_grad():
        y, _ = net(x)

    # Generate score maps
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, 0.7, 0.4, 0.4, False)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    # Handle None polygons
    for i in range(len(polys)):
        if polys[i] is None:
            polys[i] = boxes[i]

    # Save the mask image
    mask_output_path = os.path.join(image_specific_folder, f"{image_name_no_ext}_mask.jpg")
    render_img = np.hstack((score_text, score_link))
    cv2.imwrite(mask_output_path, imgproc.cvt2HeatmapImg(render_img))

    # Save the detected text regions and results text file
    file_utils.saveResult(input_image_path, image[:, :, ::-1], polys, dirname=image_specific_folder)

    # Return the path to the bounding boxes text file
    print("\nCreating Text bboxes...")
    text_file_path = os.path.join(image_specific_folder, f"res_{image_name_no_ext}.txt") 
    print(f"{len(boxes)} Text bounding boxes saved in: {text_file_path}")
    return text_file_path 

