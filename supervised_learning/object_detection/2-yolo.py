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
        """Processes outputs from
        Darknet model for one image"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size

        for idx, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Create meshgrid for x and y coordinates
            cx = np.arange(grid_width).reshape(1, grid_width, 1)
            cy = np.arange(grid_height).reshape(grid_height, 1, 1)

            # Extract and process box coordinates
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            # Apply sigmoid to tx, ty, and confidence
            bx = (1 / (1 + np.exp(-tx)) + cx) / grid_width
            by = (1 / (1 + np.exp(-ty)) + cy) / grid_height

            # Get anchor boxes for this output scale
            pw = self.anchors[idx, :, 0]
            ph = self.anchors[idx, :, 1]

            # Calculate width and height of boxes
            bw = pw * np.exp(tw) / self.model.input.shape[1]
            bh = ph * np.exp(th) / self.model.input.shape[2]

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
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters boxes and returns filtered boxes, classes, and scores.

        Args:
            boxes: A list of numpy arrays of shape (grid_height, grid_width, anchor_boxes, 4).
            box_confidences: A list of numpy arrays of shape (grid_height, grid_width, anchor_boxes, 1).
            box_class_probs: A list of numpy arrays of shape (grid_height, grid_width, anchor_boxes, classes).

        Returns:
            A tuple of (filtered_boxes, box_classes, box_scores).
        """

        all_boxes = np.concatenate([a.reshape(-1, 4) for a in boxes], axis=0)
        all_scores = np.concatenate([a.reshape(-1) for a in box_confidences], axis=0)
        all_classes = np.argmax(np.concatenate([a.reshape(-1, -1) for a in box_class_probs], axis=0), axis=1)

        # Apply class probability threshold
        indices = np.where(all_scores > self.class_t)[0]
        all_boxes = all_boxes[indices]
        all_scores = all_scores[indices]
        all_classes = all_classes[indices]

        # Apply NMS
        nms_indices = self.non_max_suppression(all_boxes, all_scores, self.nms_t)
        filtered_boxes = all_boxes[nms_indices]
        box_classes = all_classes[nms_indices]
        box_scores = all_scores[nms_indices]

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, boxes, scores, threshold):
        """Non-Max Suppression.

        Args:
            boxes: A numpy array of shape (N, 4) representing bounding boxes.
            scores: A numpy array of shape (N,) representing confidence scores.
            threshold: The overlap threshold.

        Returns:
            A numpy array of indices to keep.
        """

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        scores, order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(ovr <= threshold)[0]
            order = order[indices + 1]

        return keep
