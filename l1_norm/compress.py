import numpy as np
import torch
import config
from torchvision.utils import make_grid


def compress_binary(img):
    img_4_bit = bit8to4(img)
    img_combined = combinebit(img_4_bit)

    return img_combined

def bit8to4(img):
    return (img//16).astype(np.uint8)

def bit4to8(img):
    return img*16

def split(img):
    bottom_img = bit8to4(img)
    top_img = img - bottom_img*16
    return np.vstack((top_img, bottom_img))

def combinebit(img_4_bit):
    h,w, _ = img_4_bit.shape

    if h % 2 == 0:
        top_img = img_4_bit[:h//2]
        bottom_img = img_4_bit[h//2:]

    else:
        raise ValueError("height shape should be even, please resize your height of image into become even number")

    return top_img+(bottom_img*16)

def recovery_binary(img):
    split_img = split(img)
    img_8_bit = bit4to8(split_img)

    return img_8_bit

def recovery_srgan(img, gen):
    with torch.no_grad():
        upscaled_img = gen(
            config.test_transform(image=np.asarray(img))["image"]
            .unsqueeze(0)
            .to(config.DEVICE)
        )
        upscaled_img = upscaled_img * 0.5 + 0.5
        upscaled_img = make_grid(upscaled_img)
        upscaled_img = upscaled_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        
        return upscaled_img
