import cv2
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch


def draw_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    label,
    color: tuple[int, int, int],
):
    if bbox == []:
        return
    label = str(label)
    bbox = [int(x) for x in bbox]

    x, y, w, h = bbox[:4]
    x1 = x - w // 2
    y1 = y - h // 2
    x2 = x + w // 2
    y2 = y + h // 2

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    text_x_pos = x1
    text_y_pos = y1 - 10

    cv2.putText(
        image,
        label,
        (text_x_pos, text_y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


class ObjectFinder:
    def __init__(self, model_path: str):
        self.session = onnxruntime.InferenceSession(model_path)

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45):
        """Applies Non-Maximum Suppression to filter overlapping detections and return boxes with their scores."""
        boxes = prediction[..., :4]
        scores = prediction[..., 4]

        # Filter by confidence threshold
        mask = scores > conf_thres
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return [], []  # Return empty lists for both boxes and scores

        # Sort by confidence
        indices = scores.argsort(descending=True)

        # Apply NMS
        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i.item())  # Convert tensor to integer
            iou = self.box_iou(boxes[i], boxes[indices[1:]])  # Calculate IoU
            indices = indices[1:][iou <= iou_thres]

        return boxes[keep], scores[keep]  # Return boxes and their corresponding scores

    def box_iou(self, box1, box2):
        """Calculates IoU (Intersection over Union) between two sets of boxes."""
        x1 = torch.max(box1[0], box2[:, 0])
        y1 = torch.max(box1[1], box2[:, 1])
        x2 = torch.min(box1[2], box2[:, 2])
        y2 = torch.min(box1[3], box2[:, 3])

        inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        return inter_area / (box1_area + box2_area - inter_area)

    def detect_object_in_image(self, img, draw_result=False):
        def convertbbox2list(bbox):
            bbox = bbox.tolist()
            return [int(x) for x in bbox]

        def convert_bbox_to_input_dims(bbox, input_image_size, output_image_size):
            normalized_bbox = [i / output_image_size for i in bbox]
            bbox_for_input_image = [int(i * input_image_size) for i in normalized_bbox]
            return bbox_for_input_image

        input_img_width = img.shape[1]
        input_img_height = img.shape[0]

        print(f"model recieved an image of size {input_img_width}x{input_img_height}")

        expected_model_dims = self.session.get_inputs()[0].shape[2]

        print(f"This model expects an image of size {expected_model_dims}")

        # resize the image to the expected model dims

        img = cv2.resize(img, (expected_model_dims, expected_model_dims))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = np.transpose(img, (2, 0, 1)).reshape(
            1, 3, expected_model_dims, expected_model_dims
        )
        self.output_image_size = expected_model_dims

        # predict
        pred = self.session.run(None, {self.session.get_inputs()[0].name: img})

        boxes, scores = self.non_max_suppression(
            torch.tensor(pred[0]), conf_thres=0.25, iou_thres=0.45
        )

        if len(boxes) == 0:
            return [], 0

        scores = np.array(scores)
        max_score = np.max(scores)
        index_of_max_score = np.where(scores == max_score)[0][0]
        best_bbox_xywh = boxes[index_of_max_score]
        best_bbox_xywh = convertbbox2list(best_bbox_xywh)

        print(f'raw best_bbox_xywh": {best_bbox_xywh}')
        normalized_best_bbox = [i / expected_model_dims for i in best_bbox_xywh]
        print(f'normalized_best_bbox": {normalized_best_bbox}')
        true_bbox_xywh = [
            normalized_best_bbox[0] * input_img_width,
            normalized_best_bbox[1] * input_img_height,
            normalized_best_bbox[2] * input_img_width,
            normalized_best_bbox[3] * input_img_height,
        ]
        print(f'true_bbox_xywh": {true_bbox_xywh}')

        return true_bbox_xywh, max_score


def load_images(image_paths):
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        images.append(img)
    return images


def resize_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height))


def randomly_resize_images(image, width_range, height_range):
    def select_ranges(width_range, height_range):
        random_w = random.randint(width_range[0], width_range[1])
        random_h = random.randint(height_range[0], height_range[1])
        if abs(random_w - random_h) > 200:
            return select_ranges(width_range, height_range)

        return random_w, random_h

    new_images = []
    for img in images:
        new_width, new_height = select_ranges(width_range, height_range)
        print(f"Using new dims of {new_width}:{new_height}")
        new_img = resize_image(img, new_width, new_height)
        new_images.append(new_img)
    return new_images


def show_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    image_paths = [
        r"H:\my_files\my_data\fishbot\datasets\first_person\annotations\bbox\train\images\89428378478BC1P4ECJM46HPQ.jpg",
        r"H:\my_files\my_data\fishbot\datasets\first_person\annotations\bbox\train\images\89415614876J33B4088IX18D7.jpg",
        r"H:\my_files\my_data\fishbot\datasets\first_person\annotations\bbox\train\images\893719639928O8WPF79516F6O.jpg",
        r"H:\my_files\my_data\fishbot\datasets\kpis\artathi\annotations\bbox\train\images\380161203146LK8CETTL6N5OW.jpg",
        r"H:\my_files\my_data\fishbot\datasets\kpis\artathi\annotations\bbox\train\images\380174241916D08N3CI82DG2T.jpg",
        r"H:\my_files\my_data\fishbot\datasets\kpis\artathi\annotations\bbox\train\images\3801797534173GXN387G1W8R6.jpg",
        r"H:\my_files\my_data\fishbot\datasets\kpis\artathi\annotations\bbox\train\images\380179289416860V967X80UOI.jpg",
    ]
    images = load_images(image_paths)
    images = randomly_resize_images(images, (300, 1200), (300, 1200))

    models_folder = r"H:\my_files\my_data\fishbot\archived_models\bobber"
    model_paths = [
        os.path.join(models_folder, f)
        for f in os.listdir(models_folder)
        if f.endswith(".onnx")
    ]
    object_finders = [ObjectFinder(model_path) for model_path in model_paths]
    for image in images:
        show_image(image)
        for object_finder in object_finders:
            bbox, score = object_finder.detect_object_in_image(image, draw_result=True)
            print(bbox, score)
            draw_bbox(image, bbox, str(bbox) + str(score), (0, 255, 0))
            show_image(image)