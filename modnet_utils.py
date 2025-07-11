from modnet.model import MODNet  # update to match your model path
import torch
import numpy as np
from PIL import Image

def load_modnet_model(ckpt_path):
    model = MODNet(backbone_pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def run_modnet(modnet, image: Image.Image) -> Image.Image:
    img = image.resize((512, 512))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).unsqueeze(0)

    with torch.no_grad():
        matte = modnet(img_tensor)[0][0].numpy()

    matte = (matte * 255).astype(np.uint8)
    alpha = Image.fromarray(matte).convert("L")
    img.putalpha(alpha)
    return img
