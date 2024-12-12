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
        """Applies non-max suppression to filtered boxes"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Get unique classes
        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            # Get indices of boxes for current class
            indices = np.where(box_classes == cls)[0]

            # Get boxes and scores for current class
            class_boxes = filtered_boxes[indices]
            class_scores = box_scores[indices]

            # Sort by score in descending order
            score_sort = np.argsort(-class_scores)
            class_boxes = class_boxes[score_sort]
            class_scores = class_scores[score_sort]

            while len(class_boxes) > 0:
                # Take the box with highest score
                box_predictions.append(class_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(class_scores[0])

                if len(class_boxes) == 1:
                    break

                # Compare with rest of the boxes
                box = class_boxes[0]

                # Calculate coordinates of intersection
                x1 = np.maximum(box[0], class_boxes[1:, 0])
                y1 = np.maximum(box[1], class_boxes[1:, 1])
                x2 = np.minimum(box[2], class_boxes[1:, 2])
                y2 = np.minimum(box[3], class_boxes[1:, 3])

                # Calculate intersection area
                intersection_area = np.maximum(0, x2 - x1) * \
                    np.maximum(0, y2 - y1)

                # Calculate union area
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                class_boxes_area = (class_boxes[1:, 2] - class_boxes[1:, 0]) *\
                    (class_boxes[1:, 3] - class_boxes[1:, 1])
                union_area = box_area + class_boxes_area - intersection_area

                # Calculate IoU
                iou = intersection_area / union_area

                # Keep boxes with IoU less than threshold
                keep_indices = np.where(iou < self.nms_t)[0]
                class_boxes = class_boxes[keep_indices + 1]
                class_scores = class_scores[keep_indices + 1]

        # Convert results to numpy arrays
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    def preprocess_images(self, images):
        """Preprocess images for YOLO model input"""
        # Get input shape from the model
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        # Store original image shapes
        image_shapes = np.array([img.shape[:2] for img in images])

        # Preprocess images
        pimages = []
        for img in images:
            # Resize image using inter-cubic interpolation
            resized_img = K.preprocessing.image.smart_resize(
                img, (input_h, input_w), 
                interpolation='bicubic'
            )
        
            # Rescale pixel values to [0, 1]
            normalized_img = resized_img / 255.0
        
            pimages.append(normalized_img)

        # Convert to numpy array
        pimages = np.array(pimages)

        return pimages, image_shapes
