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
        """Filter boxes based on class confidence threshold"""
        filtered_boxes_list = []
        filtered_classes_list = []
        filtered_scores_list = []

        for output_boxes, output_confidences, output_class_probs in zip(
            boxes, box_confidences, box_class_probs
        ):
            # Reshape and flatten the inputs
            box_count = (
                output_boxes.shape[0] *
                output_boxes.shape[1] *
                output_boxes.shape[2]
            )
            boxes_flat = output_boxes.reshape(box_count, 4)
            confidences_flat = output_confidences.reshape(box_count)
            class_probs_flat = output_class_probs.reshape(box_count, -1)

            # Compute box scores
            box_scores = confidences_flat * np.max(class_probs_flat, axis=1)

            # Find indices of boxes above threshold
            mask = box_scores >= self.class_t

            # Get filtered boxes, classes, and scores
            filtered_boxes = boxes_flat[mask]
            filtered_classes = np.argmax(class_probs_flat[mask], axis=1)
            filtered_scores = box_scores[mask]

            filtered_boxes_list.append(filtered_boxes)
            filtered_classes_list.append(filtered_classes)
            filtered_scores_list.append(filtered_scores)

        # Combine results from all scales
        filtered_boxes = np.concatenate(filtered_boxes_list)
        box_classes = np.concatenate(filtered_classes_list)
        box_scores = np.concatenate(filtered_scores_list)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Perform non-maximum suppression on filtered boxes"""
        # Get unique classes
        unique_classes = np.unique(box_classes)

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            # Find indices of boxes for current class
            class_indices = np.where(box_classes == cls)[0]
            class_boxes = filtered_boxes[class_indices]
            class_scores = box_scores[class_indices]

            # Sort boxes by score in descending order
            sorted_indices = np.argsort(-class_scores)
            class_boxes = class_boxes[sorted_indices]
            class_scores = class_scores[sorted_indices]

            # Store selected boxes
            selected_boxes = []
            selected_scores = []

            while len(class_boxes) > 0:
                # Keep the box with highest score
                best_box = class_boxes[0]
                best_score = class_scores[0]
                selected_boxes.append(best_box)
                selected_scores.append(best_score)

                # Remove the best box from the list
                class_boxes = class_boxes[1:]
                class_scores = class_scores[1:]

                # If no more boxes, break
                if len(class_boxes) == 0:
                    break

                # Compute IoU (Intersection over Union)
                ious = calculate_iou(best_box, class_boxes)

                # Remove boxes with IoU above threshold
                mask = ious <= self.nms_t
                class_boxes = class_boxes[mask]
                class_scores = class_scores[mask]

            # Add selected boxes for this class
            box_predictions.extend(selected_boxes)
            predicted_box_classes.extend([cls] * len(selected_boxes))
            predicted_box_scores.extend(selected_scores)

        # Convert to numpy arrays
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores
