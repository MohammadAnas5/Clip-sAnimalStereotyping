from PIL import Image
import os

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
            return True
    except Exception:
        return False
