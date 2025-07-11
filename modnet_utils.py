from modnet.model.modnet import MODNet
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def load_modnet_model(ckpt_path):
    model = MODNet(backbone_pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def run_modnet(modnet, image: Image.Image) -> Image.Image:
    img = image.resize((512, 512)).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        matte = modnet(img_tensor)[0][0].numpy()

    matte = (matte * 255).astype(np.uint8)
    alpha = Image.fromarray(matte).convert("L")
    result = img.copy()
    result.putalpha(alpha)
    return result
