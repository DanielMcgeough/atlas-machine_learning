#!/usr/bin/env python3
"""This Object Detection Stuff is Wild."""

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import keras as K


class YOLO:
    """Initilaize YOLO v3 class and setup some other vars
    Its concerning because it will get long after this."""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
