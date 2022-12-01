import os
from roi_detection_module.model import log
from roi_detection_module.config import Config
import numpy as np
import cv2




def model_predict(image_url, model):

    image_cv = cv2.imread(image_url)

    r = model.detect([image_cv], verbose=None)

    return r

def model_output(r):

    #class_id = r[0]['class_ids'][0] # 1 = Late Regeneration, 2 = Early Regeneration
    class_id = 1
    try:
        mask =  r[0]['masks'][:, :, 0]
        mask = 255 * (r[0]['masks'][:, :, 0])
        Rcontours, hier_r = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(Rcontours, key=cv2.contourArea)[-1]
        arclen = cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(cnt, arclen * 0.0005, True)

        list_circles = []

        for p in approx :
            point = (p[0][0], p[0][1])
            list_circles.append(point)
    except:
        list_circles = []

    return list_circles, class_id

def model_worker(image_url, model):

    roi_inference = model_predict(image_url, model)
    roi_output, class_id = model_output(roi_inference)

    return roi_output, class_id