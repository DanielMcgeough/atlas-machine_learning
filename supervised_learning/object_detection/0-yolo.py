#!/usr/bin/env python3
"""Module defines the class Yolo v3
This is the cornerstone of Object Detection
YOLO is supposed to strike a balance
between speed and accuracy"""
from tensorflow import keras as K


def load_class_names(filepath):
    """Load the class"""
    with open(filepath, "r") as file:
        class_names = file.readlines()
    return [name.strip() for name in class_names]


class Yolo:
    """Class for YOLO that is an
    object detection system that
    has a balance between speed
    and accuracy it is popular in
    applications like self driving
    cars"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """initialization of yolo"""
        self.model = K.models.load_model(model_path)
        self.class_names = load_class_names(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
