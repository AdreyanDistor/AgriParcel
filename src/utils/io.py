
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_images(folder, extensions=('.tif', '.tiff', '.jpg', '.png')):
    return [f for f in os.listdir(folder) if f.lower().endswith(extensions)]

def get_basename(path):
    return os.path.splitext(os.path.basename(path))[0]
