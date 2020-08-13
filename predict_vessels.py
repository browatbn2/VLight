import os
import torch
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

import albumentations as alb
from albumentations.pytorch import transforms as alb_torch

from csl_common.utils.nn import to_numpy
from csl_common.vis import vis


_crop_to_tensor = alb.Compose([
    alb_torch.ToTensor(normalize=dict(mean=[0.518, 0.418, 0.361], std=[1, 1, 1]))
])


def _predict_center_crop(net, image, crop_size=544, gpu=True):
    h, w, c = image.shape
    image_probs = torch.zeros((h, w))

    x = (w - crop_size) // 2
    y = (h - crop_size) // 2
    image_crop = image[y:y+crop_size, x:x+crop_size]

    input = _crop_to_tensor(image=image_crop)['image']
    if gpu:
        input = input.cuda()

    with torch.no_grad():
        t = time.time()
        crop_probs = net.forward(input.unsqueeze(0))
        print(f'time forward: {int(1000 * (time.time() - t))}ms')

    show = False
    if show:
        disp_crop = vis.to_disp_image(input, denorm=True)
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(disp_crop)
        ax[1].imshow(to_numpy(crop_probs[0,0]), cmap=plt.cm.viridis, vmin=0, vmax=1)
        plt.tight_layout()
        plt.show()

    image_probs[y:y+crop_size, x:x+crop_size] = crop_probs.squeeze().squeeze()
    return image, to_numpy(image_probs)



def _predict_image(net, image, input_size, gpu=True):
    """
    Predict vessel probabilities for image
    :param image: torch.tensor
    :param scale: scale factor
    :return: recon, probs
    """

    net.eval()
    h, w, c = image.shape
    d = 32  # overlap
    inner_size = input_size - d*2

    npx = int(np.ceil(w / inner_size))
    npy = int(np.ceil(h / inner_size))

    w_pad = int(npx * inner_size) + 2*d
    h_pad = int(npy * inner_size) + 2*d

    padx = int(np.round((w_pad - w) / 2))
    pady = int(np.round((h_pad - h) / 2))
    image_pad = np.pad(image, [(pady, pady), (padx, padx), (0,0)])

    print(f'shape image_pad: {image_pad.shape}')

    s = input_size  # network input size
    ncrops = npx * npy
    print(f"{ncrops} crops...")

    def predict_sequential():
        image_probs = torch.zeros((h_pad, w_pad))
        for ix in range(npx):
            for iy in range(npy):
                x = ix * inner_size
                y = iy * inner_size

                crop = image_pad[y:y+s, x:x+s]
                input = _crop_to_tensor(image=crop)['image'].cuda()

                with torch.no_grad():
                    crop_probs = net.forward(input.unsqueeze(0))

                show = False
                if show:
                    disp_crop = vis.to_disp_image(input, denorm=True)
                    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
                    ax[0].imshow(disp_crop)
                    ax[1].imshow(to_numpy(crop_probs[0,0]), cmap=plt.cm.viridis, vmin=0, vmax=1)
                    plt.tight_layout()
                    plt.show()

                image_probs[y+d:y+d+inner_size, x+d:x+d+inner_size] = \
                    crop_probs.squeeze().squeeze()[d:d+inner_size, d:d+inner_size]

        return to_numpy(image_probs)

    def batch_predict():
        image_probs = np.zeros((h_pad, w_pad))
        inputs = []
        for ix in range(npx):
            for iy in range(npy):

                x = ix * inner_size
                y = iy * inner_size

                crop = image_pad[y:y+s, x:x+s]
                input = _crop_to_tensor(image=crop)['image']
                inputs.append(input)

        inputs = torch.stack(inputs)

        with torch.no_grad():
            crop_probs = net.forward(inputs.cuda())

        crop_probs = to_numpy(crop_probs)

        crop_id = 0
        for ix in range(npx):
            for iy in range(npy):
                x = ix * inner_size
                y = iy * inner_size

                image_probs[y+d:y+d+inner_size, x+d:x+d+inner_size] = \
                    crop_probs[crop_id, 0, d:d+inner_size, d:d+inner_size]
                crop_id += 1
        return image_probs

    t = time.time()

    # image_probs = batch_predict()
    image_probs = predict_sequential()

    image_probs = image_probs[pady:pady+h, padx:padx+w]
    print(f'cnn time: {int(1000*(time.time()-t))}ms')

    return image, image_probs


def _process_scale(net, full_image, scale, patch_size, gt_mask=None, gpu=True):
    """
    Rescale image and predict
    :param image: np.array
    :param scale: scale factor
    :return: recon, probs
    """
    print(f"Scale {scale}")
    if scale is not None and scale != 1:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        image = cv2.resize(full_image, None, fx=scale, fy=scale, interpolation=interpolation)
    else:
        # image = cv2.resize(full_image, (544, 544), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        image = full_image

    t = time.perf_counter()
    image_recon, image_probs = _predict_image(net, image, input_size=patch_size, gpu=gpu)
    # image_recon, image_probs = _predict_center_crop(net, image, crop_size=patch_size, gpu=gpu)
    # torch.cuda.synchronize()

    print(f'_predict_image time: {int(1000*(time.perf_counter()-t))}ms')

    full_h, full_w = full_image.shape[:2]
    if scale is not None and scale != 1:
        print('resizing....', scale)
        interpolation = cv2.INTER_AREA if scale > 1 else cv2.INTER_CUBIC
        image_probs = cv2.resize(image_probs, dsize=(full_w, full_h), interpolation=interpolation)

    # image_probs = cv2.resize(image_probs, dsize=(full_w, full_h), interpolation=cv2.INTER_CUBIC)
    image_probs = np.clip(image_probs, a_min=0, a_max=1.0)

    # print(eval_vessels.calculate_metrics(image_probs, full_mask))
    # eval_vessels.show_segmentation_results(full_image, image_recon, image_probs, gt_mask)
    # plt.show()

    return image_recon, image_probs


def segment_image(net, full_image, patch_size, scales=None, show=False, gt_mask=None, gpu=True):
    """
    Segment retinal vessel on full scale image. Prediction is performed on
    multiple image scales using overlappig crops.
    :param full_image:
    :param scales:
    :return:
    """

    if scales is None:
        scales = [1]

    assert isinstance(full_image, np.ndarray) or torch.is_tensor(full_image)
    assert len(full_image.shape) == 3

    stack_recon = []
    stack_probs = []

    for scale in scales:
        scale_recon, scale_probs = _process_scale(net, full_image, scale, patch_size=patch_size, gt_mask=gt_mask, gpu=gpu)
        stack_recon.append(scale_recon)
        stack_probs.append(scale_probs[np.newaxis])

    # print(f"Merging results...")
    if len(stack_probs) == 1:
        return full_image, stack_probs[0].squeeze()

    stack_probs = np.vstack(stack_probs)

    image_recon_merged = full_image
    image_probs_merged = stack_probs.mean(axis=0)

    if show:
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
        ax[0].imshow(full_image)
        ax[1].imshow(vis.to_disp_image(image_recon_merged, denorm=True))
        ax[2].imshow(image_probs_merged, cmap=plt.cm.viridis, vmin=0, vmax=1)
        plt.tight_layout()
        plt.show()

    return image_recon_merged, image_probs_merged

