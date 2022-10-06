# def processig, border_analysis, check
# class ArmDefects, CriticalRegionDefects


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import warnings

from importlib import import_module

## from inference import infer
## currently calling infer and using it
## way forward -> do not
## use pcb_demo_env tools/infer.py commands directly
## use this https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/000_getting_started/001_getting_started.ipynb


from anomalib.models.components import AnomalyModule
from skimage.segmentation import mark_boundaries
from anomalib.post_processing import Visualizer, compute_mask, superimpose_anomaly_map
from anomalib.config import get_configurable_parameters
from anomalib.deploy import TorchInferencer
import time

from anomalib.data.utils import read_image


def processing(raw_image_path, manual_rotation=False, resize_fx=0.45, resize_fy=0.45, padding=100):
    # reads from path, resizes, removes background, removes rotation, and pads image
    
    raw_image = cv2.imread(raw_image_path, cv2.IMREAD_COLOR)
    # raw image shape 1655 x 2000 x 3    
    resized_image = cv2.resize(raw_image, (0, 0), fx=resize_fx, fy=resize_fy)
    # resized image shape must be 745 x 900 x 3
    
    if resized_image.shape[0] > 770 or resized_image.shape[0] < 720 or resized_image.shape[1] < 875 or resized_image.shape[1] > 925: 
        warnings.warn("Warning: Resized Image not within correct range.")
      
    if manual_rotation:
            height, width = resized_image.shape[:2]
            center = (width/2, height/2)
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=random.randint(15, 35), scale=1)
            resized_image = cv2.warpAffine(src=resized_image, M=rotate_matrix, dsize=(width, height))
    
    gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray_img, 7, 125, 125)
    blur = cv2.GaussianBlur(blur, (39, 39), 0)
    thresh_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Perform morpholgical operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    temp_img = cv2.cvtColor(close, cv2.COLOR_GRAY2BGR)
    bg_rmvd_image = np.array(temp_img/255*resized_image, dtype=np.uint8)
    
    # Find contours and rotated tect and its corners
    contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rotatedRect = cv2.minAreaRect(contours[0])
    (x, y), (width, height), angle = rotatedRect    
    box = cv2.boxPoints(rotatedRect) # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    
    if height < 490 or height > 510 or width < 490 or width > 510:
        warnings.warn(f'Warning: Image height = {height:.2f} or width = {width:.2f} not in range, likely to be defective.')

    # List the output points in the same order as input
    # Top-left, top-right, bottom-right, bottom-left
    dstPts = [[0, 0], [width, 0], [width, height], [0, height]]

    # Get the transform
    m = cv2.getPerspectiveTransform(np.float32(box), np.float32(dstPts))
    # Transform the image
    output_image = cv2.warpPerspective(bg_rmvd_image, m, (int(width), int(height)))
    # if angle < 5, image 90 degree rotated so correction below
    if angle < 5:
        output_image = cv2.rotate(output_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    if padding is not None:
        output_image = cv2.copyMakeBorder(output_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, 0)
    
#     output_image: BGR and 700 x 700 x 3
    return output_image, (height, width)


def border_analysis(processed_image, show_defects=True, defect_threshold=5, show_details=True, show_corners=True):
    
    defect_intensity = []
    corner_distance = []
    
    gray_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(gray_img, 7, 125, 125)
    blur = cv2.GaussianBlur(blur, (39, 39), 0)
    thresh_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    display_defect_img = processed_image.copy()
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    (x, y), (width, height), angle = rect

    hull = cv2.convexHull(cnt, clockwise=True, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        if d/256 > defect_threshold:
            defect_intensity.append(d/256)
            if show_details:
                print(f'Defect Intensity: {d/256}')
            if show_defects:
                cv2.line(display_defect_img,start,end,[0,255,0],2)
                cv2.circle(display_defect_img,far,5,[0,0,255],-1)


    x_min_max = [int(np.min(np.array(cnt).reshape([-1, 2])[:, 0])), int(np.max(np.array(cnt).reshape([-1, 2])[:, 0]))] 
    y_min_max = [int(np.min(np.array(cnt).reshape([-1, 2])[:, 1])), int(np.max(np.array(cnt).reshape([-1, 2])[:, 1]))]

    counter = 0
    for i in range(2):
        for j in range(2):
            counter += 1
            corner_dist = cv2.pointPolygonTest(cnt, [x_min_max[i], y_min_max[j]], True)
            corner_distance.append(corner_dist)
            if show_details:
                print(f'Corner {counter}: {corner_dist}')            
            if show_corners:
                cv2.circle(display_defect_img, [x_min_max[i], y_min_max[j]], 5, [255,0,255],-1)

    
    return display_defect_img, defect_intensity, corner_distance


class CriticalRegionDefects:
     
    def __init__(self, config_path, model_path, log_images=False):
        self.config = get_configurable_parameters(config_path=config_path)
        self.model_path = model_path
        
        if not log_images:
            self.config["project"]["log_images_to"] = []

        self.inferencer = TorchInferencer(config=self.config, model_source=self.model_path)

    def infer(self, image_path, threshold, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 255)):
        
        # returns image, boolean, defect intensity and number of defects
        is_defect_in_critical = False
        defect_intensity, number_of_defects = 0.0, 0

        predictions = self.inferencer.predict(image=read_image(image_path))
        pred_mask = predictions.pred_mask

        temp_img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        vis_img = mark_boundaries(temp_img, pred_mask, color=(1, 0, 0), mode="thick")
        
        vis_img_rect = vis_img.copy()
        vis_img_rect = np.array(vis_img_rect*255, dtype=np.uint8)
        cv2.rectangle(vis_img_rect, (50, 100), (225, 175), (0, 255,0), 3)
        
        if sum(sum(pred_mask[100:175, 50:225] > 1)) > 1:
            
            defect_intensity = sum(sum(pred_mask[100:175, 50:225] > 1))*100/(75*175)
            number_of_defects = len(cv2.findContours(pred_mask[100:175, 50:225], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0])
            text1 = f'Defect in Critical Area'
            text2 = f'Defect Intensity: {defect_intensity:.2f}%'
            text3 = f'Number of Defects: {number_of_defects}'

            vis_img_rect = cv2.putText(vis_img_rect, text1, (10, 50), font, fontScale, color, 1)
            vis_img_rect = cv2.putText(vis_img_rect, text2, (10, 65), font, fontScale, color, 1)
            vis_img_rect = cv2.putText(vis_img_rect, text3, (10, 80), font, fontScale, color, 1)
            
            is_defect_in_critical = True
            
        return vis_img_rect, is_defect_in_critical, (defect_intensity, number_of_defects)


class ArmDefects:
     
    def __init__(self, config_path, model_path, log_images=False):
        self.config = get_configurable_parameters(config_path=config_path)
        self.model_path = model_path
        
        if not log_images:
            self.config["project"]["log_images_to"] = []

        self.inferencer = TorchInferencer(config=self.config, model_source=self.model_path)
     
    def infer(self, image_path, threshold):

        predictions = self.inferencer.predict(image=read_image(image_path))
        pred_mask = predictions.pred_mask
        
        raw_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        temp_img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        vis_img = mark_boundaries(temp_img, pred_mask, color=(1, 0, 0), mode="thick")
        image_for_arms, is_defect_on_arms = self.detect_defect_on_arms(raw_image, pred_mask, 
                                                                       developer_mode_visualization=False)
            
        return image_for_arms, is_defect_on_arms
    
    
    def detect_defect_on_arms(self, image, segmented_img, is_image_processed=True, developer_mode_visualization=False):
    
        defect_on_arms = False

        # get avg pixel value of non-zero pixel for binary thrshold
        out_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_v = out_gray.reshape(-1)
        img_v = img_v[np.nonzero(img_v)]   
        thresh_gray = cv2.threshold(out_gray, int(sum(img_v)/len(img_v)) + 10, 255, cv2.THRESH_BINARY)[1]

        # get square shapes out of img
        ker_size = 10
        kernel = np.array(np.ones([ker_size, ker_size]), dtype=np.uint8)
        opening_sqr = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, kernel, iterations=1)

        # remove objects/noise by thresholding area
        opr_copy = np.zeros_like(opening_sqr)
        cnts = cv2.findContours(opening_sqr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 300:
                cv2.drawContours(opr_copy, [c], -1, 255, -1)

        # create antidiagonal kernel
        mat_size = 15
        antidiag = np.fliplr(np.diag(np.ones(mat_size)))
        kernel = np.array(antidiag, dtype=np.uint8)
        opening_adl = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, kernel, iterations=1)

        # create diagonal kernel
        mat_size = 15
        diag = np.diag(np.ones(mat_size))
        kernel = np.array(diag, dtype=np.uint8)
        opening_dnl = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, kernel, iterations=1)

        # combination of openings sqr, adl and dnl to get retain diagonal objects only
        arms = np.array((opening_adl + opening_dnl)/2, dtype=np.uint8)
        h = 1 - opr_copy/255
        arms = np.array(arms*h, dtype=np.uint8)

        # find contours and filter using contour area to remove noise
        arms_copy = np.zeros_like(arms)
        cnts = cv2.findContours(arms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 150:
                cv2.drawContours(arms_copy, [c], -1, 255, -1)

        # erosion with samller kernel followed by dilation to avoid arm breakage 
        erode = cv2.erode(arms_copy, np.ones([2, 2]),iterations = 1)  
        dilate = cv2.dilate(erode, np.ones([3, 3]),iterations = 1)

        mask_for_arm = np.zeros_like(segmented_img)
        if sum(sum(np.logical_and(segmented_img!=0, dilate!=0))) > 0:
            defect_on_arms = True
#             print('Defect detected on arms!')
            cnts, _ = cv2.findContours(segmented_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(cnts)):
                temp_img = np.zeros_like(segmented_img)
                cnt = cnts[i]
                cv2.drawContours(temp_img, [cnt], 0, 255, -1)
                if sum(sum(np.logical_and(dilate!=0, temp_img!=0))) > 0:
                    cv2.drawContours(mask_for_arm, [cnt], 0, 255, -1)

        vis_img = mark_boundaries(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask_for_arm, color=(1, 0, 0), mode="thick")

        if  developer_mode_visualization:
            fig, axs = plt.subplots(1, 9, figsize=(27, 3))
            axs[0].imshow(thresh_gray, cmap='gray')
            axs[1].imshow(opening_sqr, cmap='gray')
            axs[2].imshow(opr_copy, cmap='gray')
            axs[3].imshow(opening_adl, cmap='gray')
            axs[4].imshow(opening_dnl, cmap='gray')
            axs[5].imshow(arms, cmap='gray')
            axs[6].imshow(arms_copy, cmap='gray')
            axs[7].imshow(erode, cmap='gray')
            axs[8].imshow(dilate, cmap='gray')
            plt.show()

        return vis_img, defect_on_arms


def checks(height, width, corner_dist, border_intrusion, critical_region, arms):
        
    size_check = "Pass"
    if height < 490 or height > 510 or width < 490 or width > 510:
        size_check = "Fail"
            
    corner_check = "Pass"
    if min(corner_dist) < -20:
        corner_check = "Fail"
            
    border_check = "Pass"
    if border_intrusion:
        border_check = "Fail"
        
    critical_region_check = "Pass"
    if critical_region:
        critical_region_check = "Fail"
            
    arms_check = "Pass"
    if arms:
        arms_check = "Fail"
            
    total_check = "Pass" if size_check == corner_check == border_check == critical_region_check == arms_check == "Pass" else "Fail"
        
    return size_check, (corner_check, border_check), critical_region_check, arms_check, total_check


