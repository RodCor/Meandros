import os
import roi_detection_module.model as modellib
from roi_detection_module.model import log
from roi_detection_module.config import Config
import numpy as np
import cv2



class AxoHandConfig(Config):

        NAME = "axol_hand"

        NUM_CLASSES = 1 + 2 

        DETECTION_MIN_CONFIDENCE = 0.9

        GPU_COUNT = 1

        IMAGES_PER_GPU = 1


def model_prepare(image_url):

    inference_config = AxoHandConfig()
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=MODEL_DIR)

    model_path = os.path.join(ROOT_DIR, "roi_detection_module/mask_rcnn_axol_hand.h5")
    class_names = ['BG','axol_hand', 'axol_hand_early']

    model.load_weights(model_path, by_name=True)

    image_cv = cv2.imread(image_url)

    r = model.detect([image_cv], verbose=None)

    return r

def model_output(r):

    #class_id = r[0]['class_ids'][0] # 1 = Late Regeneration, 2 = Early Regeneration
    class_id = 1
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

    return list_circles, class_id

def model_worker(image_url):

    roi_inference = model_prepare(image_url)
    roi_output, class_id = model_output(roi_inference)

    return roi_output, class_id