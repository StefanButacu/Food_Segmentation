import colorsys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color

import sys
sys.path.append("E:\PythonModels\segment-anything")
from segment_anything import sam_model_registry, SamPredictor

class Service:

    def __init__(self):
        sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        device = "cpu" # TODO - get the device from as parameter
        # self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # self.sam.to(device=device)
        # self.predictor = SamPredictor(self.sam)

    def get_dummy_overlay_with_map(self):
        image = Image.open('E:\Licenta_DOC\App\src\main\\resources\image\\target_python.jpg')
        image = np.array(image)
        prediction = np.random.randint(0, 5, (image.shape[0], image.shape[1]))
        prediction, label_color_map = self.generate_colors_for_prediction(prediction)
        overlay_photo = color.label2rgb(prediction, image, saturation=1, alpha=0.5, bg_color=None)
        overlay_photo = np.rot90(overlay_photo)
        overlay_photo = np.rot90(overlay_photo)
        overlay_photo = np.rot90(overlay_photo)
        overlay_photo = np.fliplr(overlay_photo)
        Image.fromarray((overlay_photo * 255).astype(np.uint8)).save('overlay-photo.png')
        ###
        return (overlay_photo * 255).astype(np.uint8).tolist(), label_color_map

    def get_overlay_with_map(self, image, prediction):
        overlay_heatmap, label_color_map = self.generate_colors_for_prediction(prediction)
        overlay_photo = self.apply_overlay_with_matplotlib(image, overlay_heatmap, label_color_map, alpha=0.52)
        overlay_photo = np.rot90(overlay_photo)
        overlay_photo = np.rot90(overlay_photo)
        overlay_photo = np.rot90(overlay_photo)
        overlay_photo = np.fliplr(overlay_photo)
        return overlay_photo.tolist(), label_color_map

    def apply_overlay_with_matplotlib(self, image, mask, label_colors, alpha=0.5):
        overlay = np.zeros_like(image)
        for label, color in label_colors.items():
            overlay[mask == label] = color

        blended = (1 - alpha) * image + alpha * overlay
        return blended.astype(np.uint8)

    # TODO extract method remove noise
    def generate_colors_for_prediction(self, prediction):
        self.remove_noise(prediction)
        without_zero, without_zero_counts = np.unique(prediction[prediction != 0], return_counts=True) # Keep track of top aparitions
        number_of_colors = np.unique(without_zero).size
        generated_colors = self.generate_distinct_colors(number_of_colors)

        without_zero = sorted(without_zero, key=lambda el: without_zero_counts[np.where(without_zero == el)[0]][0])

        label_color_dict = {}
        for label, generated_color in zip(without_zero, generated_colors):
            label_color_dict[label] = generated_color

        return prediction, label_color_dict

    def remove_noise(self, prediction):
        total_size = prediction.shape[0] * prediction.shape[1]
        threshold_count = total_size * 0.02
        unique_values, counts = np.unique(prediction, return_counts=True)
        mask = unique_values[counts < threshold_count]
        prediction[np.isin(prediction, mask)] = 0

    def generate_distinct_colors(self, n):
        colors = []

        for i in range(n):
            hue = i / n
            rgb = colorsys.hls_to_rgb(hue, 0.5, 1)
            rgb_255 = [int(x * 255) for x in rgb]
            colors.append(rgb_255)
        return colors

    def find_bounding_box_for_label(self, mask, label):
        min_row, min_col, max_row, max_col = None, None, None, None
        rows, cols = mask.shape
        for row in range(rows):
            for col in range(cols):
                if mask[row, col] == label:
                    if min_row is None or row < min_row:
                        min_row = row
                    if min_col is None or col < min_col:
                        min_col = col
                    if max_row is None or row > max_row:
                        max_row = row
                    if max_col is None or col > max_col:
                        max_col = col

        return min_row, min_col, max_row, max_col

    def get_bounding_boxes(self, mask):
        boxes = []
        self.remove_noise(mask)
        unique_values, counts = np.unique(mask, return_counts=True)
        for label in unique_values:
            if label != 0:
                min_row, min_col, max_row, max_col = self.find_bounding_box_for_label(mask, label)
                boxes.append([label, (min_row, min_col), (max_row, max_col)])
        return boxes

    def get_sam_mask(self,image, current_pred):
        pass
        # self.predictor.set_image(image)
        # input_boxes = self.get_bounding_boxes(current_pred) # Todo unpack input box T_x, T_y, B_x, B_y
        # transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        # masks, _, _ = self.predictor.predict_torch(
        #     point_coords=None,
        #     point_labels=None,
        #     boxes=transformed_boxes,
        #     multimask_output=False,
        # )
        # masks.shape  # (batch_size) x (num_predicted_masks_per_input) x H x W   # True,False
        # return masks[0]