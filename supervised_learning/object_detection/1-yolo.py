#!/usr/bin/env python3
""" Module defines the class Yolo """
from tensorflow import keras as K
import numpy as np


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


    def process_outputs(self, outputs, image_size):
        """Processes outputs from Darknet model for one image"""
        boxes, box_confidences, box_class_probs = []
        image_height, image_width = image_size

        for idx, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Create meshgrid for x and y coordinates
            cx = np.arange(grid_width).reshape(1, grid_width, 1)
            cy = np.arange(grid_height).reshape(grid_height, 1, 1)

            # Extract and process box coordinates
            bx = output[..., 0]
            by = output[..., 1]
            bw = output[..., 2]
            bh = output[..., 3]

            # Apply sigmoid to tx, ty, and confidence
            bx = (1 / (1 + np.exp(-bx)) + cx) / grid_width
            by = (1 / (1 + np.exp(-by)) + cy) / grid_height

            # Get anchor boxes for this output scale
            pw = self.anchors[idx, :, 0]
            ph = self.anchors[idx, :, 1]

            # Calculate width and height of boxes
            bw = pw * np.exp(bw) / self.model.input.shape[1]
            bh = ph * np.exp(bh) / self.model.input.shape[2]

            # Calculate x1, y1, x2, y2
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            # Update boxes
            output[..., 0] = x1
            output[..., 1] = y1
            output[..., 2] = x2
            output[..., 3] = y2

            # Process confidences and class probabilities
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            boxes.append(output[..., :4])
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
