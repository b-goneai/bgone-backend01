import numpy as np
from PIL import Image

def preprocess(image):
    image = image.resize((512, 512))
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np.transpose(2, 0, 1)
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

def postprocess(result, original_image):
    mask = result[0][0]
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask).resize(original_image.size)
    rgba_image = original_image.convert("RGBA")
    datas = rgba_image.getdata()
    newData = []
    for item, alpha in zip(datas, mask.getdata()):
        newData.append((*item[:3], alpha))
    rgba_image.putdata(newData)
    return rgba_image
