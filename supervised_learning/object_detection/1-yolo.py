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

        for output in outputs:
            grid_h, grid_w, num_boxes = output.shape[:3]

            # Reshape the output
            output = output.reshape(grid_h, grid_w, num_boxes, 4 + 1 + len(self.class_names))

            # Decode bounding box coordinates
            box_xy, box_wh, box_confidence, box_class_probs = output[..., :2], output[..., 2:4], output[..., 4], output[..., 5:]

            # Convert (x, y) to (x1, y1, x2, y2)
            box_xy = (box_xy + np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')) / (grid_h, grid_w)
            box_wh = np.exp(box_wh) * self.anchors[output_idx] / (416, 416)
            box_xy -= box_wh / 2
            box_xy *= image_size
            box_wh *= image_size
            boxes.append(np.concatenate([box_xy, box_xy + box_wh], axis=-1))

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_probs)

        # Concatenate outputs from different layers
        boxes = np.concatenate(boxes, axis=0)
        box_confidences = np.concatenate(box_confidences, axis=0)
        box_class_probs = np.concatenate(box_class_probs, axis=0)

        # Apply non-maximum suppression
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.convert_to_tensor(boxes),
            scores=tf.convert_to_tensor(box_confidences),
            classes=tf.convert_to_tensor(np.argmax(box_class_probs, axis=-1)),
            max_output_size=100,
            iou_threshold=self.nms_t,
            score_threshold=self.class_t
        )

        return boxes.numpy(), scores.numpy(), classes.numpy()
