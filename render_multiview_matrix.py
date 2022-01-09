import argparse
import json
import numpy as np
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont

import curriculums

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_image(generator, z, **kwargs):
    with torch.no_grad():
        img, depth_map = generator.staged_forward(z, **kwargs)

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min)/(img_max-img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, depth_map


def make_curriculum(curriculum):
    # verify file system
    curriculum = getattr(curriculums, curriculum, None)
    if curriculum is None:
        raise ValueError(f'{curriculum} is not a valid curriculum')
    curriculum['psi'] = 0.7
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
    return curriculum


def load_generator(model_path):
    generator = torch.load(os.path.join(model_path, 'generator.pth'), map_location=torch.device(device))
    ema = torch.load(os.path.join(model_path, 'ema.pth'))
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()
    return generator


def write_labels(canvas, yaw, pitch, img_size, text_height=20, left_margin=0):
    draw = ImageDraw.Draw(canvas)
    for iy, y in enumerate(yaw):
        draw.text((img_size*iy + left_margin, 0), f'{y:.3f}', fill=(0, 0, 0))
    for ip, p in enumerate(pitch):
        draw.text((0, img_size*ip + text_height), f'{p:.3f}', fill=(0, 0, 0))
    return canvas


def make_matrix(generator, curriculum, seed, yaw, pitch, img_size, text_height=20, left_margin=0):
    torch.manual_seed(seed)
    z = torch.randn((1, 256), device=device)
    curriculum = make_curriculum(curriculum)
    curriculum['img_size'] = img_size
    canvas = Image.new(
        # channels
        'RGBA',
        (
            # width
            img_size*len(yaw) + text_height,
            # height
            img_size*len(pitch) + text_height
        ),
        # fill color
        (255, 255, 255, 255)
    )
    canvas_w, canvas_h = canvas.size
    for iy, y in enumerate(yaw):
        for ip, p in enumerate(pitch):
            curriculum['h_mean'] = y
            curriculum['v_mean'] = p
            img, depth_img = generate_image(generator, z, **curriculum)
            canvas.paste(img, (img_size*iy + left_margin, img_size*ip + text_height))
    canvas = write_labels(canvas, yaw, pitch, img_size, text_height, left_margin)
    return canvas


def main():
    model_path = './models/'
    curriculum = 'ShapeNetCar'
    yaw = np.linspace(-np.pi, np.pi, 48, endpoint=False)
    pitch = np.linspace(0, np.pi/2, 12, endpoint=False)
    img_size = 128
    text_height = 20
    left_margin = 50
    seed = 0
    image = make_matrix(load_generator(model_path), curriculum, seed, yaw, pitch, img_size, text_height, left_margin)
    image.save('test.png')
    return


if __name__ == 'main':
    main()