#!/usr/bin/env python3
"""This Object Detection Stuff is Wild."""

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


class YOLO:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
