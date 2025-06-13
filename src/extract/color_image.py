
from utils.image import create_color_image, create_rgba_image, save_raster
import cv2
def generate_rgba_image(r_path, g_path, b_path, output_path):
    red = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
    green = cv2.imread(g_path, cv2.IMREAD_GRAYSCALE)
    blue = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)

    image = create_rgba_image(red, green, blue)
    save_raster(output_path, image, r_path, count=4)
