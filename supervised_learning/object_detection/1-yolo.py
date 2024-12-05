
#!/usr/bin/env python3
"""Module defines the class Yolo v3
This is the cornerstone of Object Detection
YOLO is supposed to strike a balance
between speed and accuracy"""
from tensorflow import keras as K
import numpy as np


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
def sigmoid(self, x):
        """Makes a sigmoid to return
        Idk if I'll need it"""
        return 1 / (1 + np.exp(-x))

def process_outputs(self, outputs, image_size):
        """Processes the outputs"""
        boxes = [output[..., :4] for output in outputs]
        box_confidences = []
        box_class_probs = []

        image_height = image_size[0]
        image_width = image_size[1]

        input_layer = self.model.input
        input_width, input_height = input_layer.shape[1:3]

        for i, output in enumerate(outputs):

            box = output
            box_x = box[..., 0]
            box_y = box[..., 1]
            box_w = box[..., 2]
            box_h = box[..., 3]

            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

            grid_height = box.shape[0]
            grid_width = box.shape[1]
            num_anchors = output.shape[2]

            center_x = np.arange(grid_width).reshape(1, grid_width)
            center_x = np.repeat(center_x, grid_height, axis=0)

            center_y = np.arange(grid_width).reshape(1, grid_width)
            center_y = np.repeat(center_y, grid_height, axis=0).T

            center_x = np.repeat(center_x[..., np.newaxis],
                                 num_anchors, axis=2)
            center_y = np.repeat(center_y[..., np.newaxis],
                                 num_anchors, axis=2)

            pred_x = (self.sigmoid(box_x) + center_x) / grid_width
            pred_y = (self.sigmoid(box_y) + center_y) / grid_height
            pred_w = (np.exp(box_w) * self.anchors[i, :, 0]) / input_width
            pred_h = (np.exp(box_h) * self.anchors[i, :, 1]) / input_height

            boxes[i][..., 0] = (pred_x - (pred_w / 2)) * image_width
            boxes[i][..., 1] = (pred_y - (pred_h / 2)) * image_height
            boxes[i][..., 2] = (pred_x + (pred_w / 2)) * image_width
            boxes[i][..., 3] = (pred_y + (pred_h / 2)) * image_height

            # boxes.append(box)

        return boxes, box_confidences, box_class_probs
