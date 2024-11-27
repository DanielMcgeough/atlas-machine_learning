#!/usr/bin/env python3
"""Module defines the class Yolo v3
This is the cornerstone of Object Detection
YOLO is supposed to strike a balance
between speed and accuracy"""
from tensorflow import keras as K
import numpy as np

class Yolo:
    """ Class initiliazes the Yolo v3 algorithm to perform object detection
    It accomplish this by addressing certain things needed
    to establish bounding boxes and other aspects of
    image detection with localization"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor
        Arguments:
            model_path (str): path to where Darknet Keras model is stored.
            Darknet model is considered a good set for object detection.
            It is also a good tool to use when localization is needed.
            classes_path (str): path to list of class names in order of index
                used for the model. All 80 of them which is making me
                concernicus.

            class_t (float): box score threshold to filter boxes by

            nms_t (float): IOU threshold for non-max suppression

            anchors (numpy.ndarray): array of shape (outputs, anchor_boxes, 2)
                containing all anchor boxes:
                    outputs: number of outputs (predictions) made
                    anchor_boxes: number of anchor boxes for each prediction
                    2: [anchor_box_width, anchor_box_height]

        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

def process_outputs(self, outputs, image_size):
        boxes, box_confidences, box_class_probs = [], [], []
        image_height, image_width = image_size

        for idx, outpout in enumerate(outputs):
            grid_height, grid_width, anchor+boxes, _ = output.shape

            cx = np.arange(grid_width).reshape(1, grid_width, 1)
            cy = np.arange(grid_height).reshape(grid_height, 1, 1)

            bx = output[..., 0]
            by = output[..., 1]
            bw = output[..., 2]
            bh = output[..., 3]

            pw = self.anchors[idx, :, 0]
            ph = self.anchors[idx, :, 1]
