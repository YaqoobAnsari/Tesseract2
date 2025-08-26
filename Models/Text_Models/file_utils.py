# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
    """
    Save text detection results.

    Args:
        img_file (str): image file name.
        img (array): raw image content.
        boxes (array): array of detected boxes.
        dirname (str): directory to save results.
        verticals (list): indicates vertical boxes.
        texts (list): additional text annotations.

    Returns:
        None
    """
    img = np.array(img)

    # Get the image filename without extension
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    # Create paths for the text and result image files
    res_file = os.path.join(dirname, f"res_{filename}.txt")
    res_img_file = os.path.join(dirname, f"res_{filename}.jpg")

    # Ensure the directory exists
    os.makedirs(dirname, exist_ok=True)

    # Save the text detection results to a file
    with open(res_file, 'w') as f:
        for i, box in enumerate(boxes):
            if box is None:
                continue  # Skip None boxes to avoid errors

            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + '\n'
            f.write(strResult)

            poly = poly.reshape(-1, 2)
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

            # Optional: Add vertical or textual annotations
            if verticals is not None and texts is not None:
                ptColor = (255, 0, 0) if verticals[i] else (0, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(img, f"{texts[i]}", (poly[0][0] + 1, poly[0][1] + 1), font, font_scale, (0, 0, 0), thickness=1)
                cv2.putText(img, f"{texts[i]}", tuple(poly[0]), font, font_scale, ptColor, thickness=1)

    # Save the result image
    cv2.imwrite(res_img_file, img)

