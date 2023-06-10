import colorsys
import sys

import numpy as np

sys.path.append("E:\PythonModels\segment-anything")

class Service:

    def __init__(self):
        # self.NOISE = 0.02 SAM
        self.NOISE = 0.02

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
    def generate_colors_for_prediction(self, prediction):
        self.remove_noise(prediction)
        without_zero, without_zero_counts = np.unique(prediction[prediction != 0], return_counts=True) # Keep track of top aparitions
        number_of_colors = np.unique(without_zero).size
        generated_colors = self.generate_distinct_colors(number_of_colors)

        without_zero = sorted(without_zero, key=lambda el: without_zero_counts[np.where(without_zero == el)[0]][0], reverse=True)

        label_color_dict = {}
        for label, generated_color in zip(without_zero, generated_colors):
            label_color_dict[label] = generated_color

        return prediction, label_color_dict

    def remove_noise(self, prediction):
        total_size = prediction.shape[0] * prediction.shape[1]
        threshold_count = total_size * self.NOISE
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

