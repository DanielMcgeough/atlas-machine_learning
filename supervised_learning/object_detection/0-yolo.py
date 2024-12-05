#!/usr/bin/env python3
""" Module defines the lass Yolo """
from tensorflow import keras as K


class Yolo:
    """Class for YOLO that is an
    object detection system that
    has a balance between speed
    and accuracy it is popular in
    applications like self driving
    cars"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        initializes Yolo class.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
