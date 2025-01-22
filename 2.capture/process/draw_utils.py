import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (255, 255, 255)  # Main text color
thickness = 1
background_color = (0, 0, 0)  # Black rectangle color

def draw_left_top(y: int, text: str, frame: np.ndarray):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = 10  # Padding from the left
    text_y = 20 + y * (text_size[1] + 7)  # Dynamic top offset

    # Calculate the rectangle coordinates
    rect_x1 = text_x - 5
    rect_y1 = text_y - text_size[1] - 5
    rect_x2 = text_x + text_size[0] + 5
    rect_y2 = text_y + 5

    # Draw the rectangle
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, -1)

    # Draw the text
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

def draw_right_bottom(y: int, text: str, frame: np.ndarray):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = frame.shape[1] - text_size[0] - 10  # Padding from the right
    text_y = frame.shape[0] - y * (text_size[1] + 7) - 10  # Dynamic bottom offset

    # Calculate the rectangle coordinates
    rect_x1 = text_x - 5
    rect_y1 = text_y - text_size[1] - 5
    rect_x2 = text_x + text_size[0] + 5
    rect_y2 = text_y + 5

    # Draw the rectangle
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, -1)

    # Draw the text
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
