import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet
import time
import cv2

if __name__ == '__main__':
    start_time = time.time()
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path of input image')
    args = parser.parse_args()


    # read pretraiend model
    ckpt_path = "./pretrained/modnet_photographic_portrait_matting.ckpt"
    if not os.path.exists(ckpt_path):
        print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
        exit()

    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet).cpu()
    modnet.load_state_dict(torch.load(ckpt_path,map_location=torch.device('cpu')))
    modnet.eval()

    # inference images
    print('Process image: {0}'.format(args.input))

    # read image
    image = Image.open(args.input)

    # unify image channels to 3
    im = np.asarray(image)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cpu(), False)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    matte = Image.fromarray(((matte * 255).astype('uint8')), mode='L')


    def combined_display(image, matte):
      # calculate display resolution
      w, h = image.width, image.height
      rw, rh = 800, int(h * 800 / (3 * w))

      # obtain predicted foreground
      image = np.asarray(image)
      if len(image.shape) == 2:
        image = image[:, :, None]
      if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
      elif image.shape[2] == 4:
        image = image[:, :, 0:3]
      matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
      foreground = image * matte + np.full(image.shape, 255) * (1 - matte)

      return foreground
    fg = combined_display(image, matte)

    #final image --> extracted foreground from image
    foreground = Image.fromarray(np.uint8(fg))
    foreground.save('foreground.png','PNG')

    print("Done: "+ args.input, '\n')

    print("--- %s seconds ---" % (time.time() - start_time))
