import cv2
from matplotlib import pyplot as plt
import numpy as np
from csl_common.utils.nn import to_numpy


def denormalize(tensor):
    if tensor.shape[1] == 3:
        tensor[:, 0] += 0.518
        tensor[:, 1] += 0.418
        tensor[:, 2] += 0.361
    elif tensor.shape[-1] == 3:
        tensor[..., 0] += 0.518
        tensor[..., 1] += 0.418
        tensor[..., 2] += 0.361

def denormalized(tensor):
    if isinstance(tensor, np.ndarray):
        t = tensor.copy()
    else:
        t = tensor.clone()
    denormalize(t)
    return t


def color_map(data, vmin=None, vmax=None, cmap=plt.cm.viridis):
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    val = np.maximum(vmin, np.minimum(vmax, data))
    norm = (val-vmin)/(vmax-vmin)
    cm = cmap(norm)
    if isinstance(cm, tuple):
        return cm[:3]
    if len(cm.shape) > 2:
        cm = cm[:,:,:3]
    return cm


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def make_grid(data, padsize=2, padval=255, nCols=10, dsize=None, fx=None, fy=None, normalize=False):
    # if not isinstance(data, np.ndarray):
    data = np.array(data)
    if data.shape[0] == 0:
        return
    if data.shape[1] == 3:
        data = data.transpose((0,2,3,1))
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    if normalize:
        data -= data.min()
        data /= data.max()
    else:
        data[data < 0] = 0
    #     data[data > 1] = 1

    # force the number of filters to be square
    # n = int(np.ceil(np.sqrt(data.shape[0])))
    c = nCols
    r = int(np.ceil(data.shape[0]/float(c)))

    padding = ((0, r*c - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((r, c) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((r * data.shape[1], c * data.shape[3]) + data.shape[4:])

    if dsize is not None or fx is not None or fy is not None:
        data = cv2.resize(data, dsize=dsize, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)

    return data


def vis_square(data, padsize=1, padval=0, wait=0, nCols=10, title='results', dsize=None, fx=None, fy=None, normalize=False):
    img = make_grid(data, padsize=padsize, padval=padval, nCols=nCols, dsize=dsize, fx=fx, fy=fy, normalize=normalize)
    cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(wait)


def cvt32FtoU8(img):
    return (img * 255.0).astype(np.uint8)


def to_disp_image(img, denorm=False, output_dtype=np.uint8):
    if not isinstance(img, np.ndarray):
        img = img.detach().cpu().numpy()
    img = img.astype(np.float32).copy()
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0)).copy()
    if denorm:
        img = denormalized(img)
    if img.max() > 2.00:
        if isinstance(img, np.ndarray):
            img /= 255.0
        else:
            raise ValueError("Image data in wrong value range (min/max={:.2f}/{:.2f}).".format(img.min(), img.max()))
    img = np.clip(img, a_min=0, a_max=1)
    if output_dtype == np.uint8:
        img = cvt32FtoU8(img)
    return img


def to_disp_images(images, denorm=True):
    return [to_disp_image(i, denorm) for i in images]


def add_frames_to_images(images, labels, label_colors, gt_labels=None):
    import collections
    if not isinstance(labels, (collections.Sequence, np.ndarray)):
        labels = [labels] * len(images)
    new_images = to_disp_images(images)
    for idx, (disp, label) in enumerate(zip(new_images, labels)):
        frame_width = 3
        bgr = label_colors[label]
        cv2.rectangle(disp,
                      (frame_width // 2, frame_width // 2),
                      (disp.shape[1] - frame_width // 2, disp.shape[0] - frame_width // 2),
                      color=bgr,
                      thickness=frame_width)

        if gt_labels is not None:
            radius = 8
            color = (0, 1, 0) if gt_labels[idx] == label else (1, 0, 0)
            cv2.circle(disp, (disp.shape[1] - 2*radius, 2*radius), radius, color, -1)
    return new_images


def add_cirle_to_images(images, intensities, cmap=plt.cm.viridis, radius=10):
    new_images = to_disp_images(images)
    for idx, (disp, val) in enumerate(zip(new_images, intensities)):
        # color = (0, 1, 0) if gt_labels[idx] == label else (1, 0, 0)
        # color = plt_colors.to_rgb(val)
        if isinstance(val, float):
            color = cmap(val).ravel()
        else:
            color = val
        cv2.circle(disp, (2*radius, 2*radius), radius, color, -1)
        # new_images.append(disp)
    return new_images


def get_pos_in_image(loc, text_size, image_shape):
    bottom_offset = int(6*text_size)
    right_offset = int(95*text_size)
    line_height = int(35*text_size)
    mid_offset = right_offset
    top_offset = line_height + int(0.05*line_height)
    if loc == 'tl':
        pos = (2, top_offset)
    elif loc == 'tr':
        pos = (image_shape[1]-right_offset, top_offset)
    elif loc == 'tr+1':
        pos = (image_shape[1]-right_offset, top_offset + line_height)
    elif loc == 'tr+2':
        pos = (image_shape[1]-right_offset, top_offset + line_height*2)
    elif loc == 'bl':
        pos = (2, image_shape[0]-bottom_offset)
    elif loc == 'bl-1':
        pos = (2, image_shape[0]-bottom_offset-line_height)
    elif loc == 'bl-2':
        pos = (2, image_shape[0]-bottom_offset-2*line_height)
    # elif loc == 'bm':
    #     pos = (mid_offset, image_shape[0]-bottom_offset)
    # elif loc == 'bm-1':
    #     pos = (mid_offset, image_shape[0]-bottom_offset-line_height)
    elif loc == 'br':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset)
    elif loc == 'br-1':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset-line_height)
    elif loc == 'br-2':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset-2*line_height)
    elif loc == 'bm':
        pos = (image_shape[1]-right_offset*2, image_shape[0]-bottom_offset)
    elif loc == 'bm-1':
        pos = (image_shape[1]-right_offset*2, image_shape[0]-bottom_offset-line_height)
    elif loc == 'bm-2':
        pos = (image_shape[1]-right_offset*2, image_shape[0]-bottom_offset-2*line_height)
    else:
        raise ValueError("Unknown location {}".format(loc))
    return pos


def add_id_to_images(images, ids, gt_ids=None, loc='tl', color=(1,1,1), size=0.7, thickness=1):
    new_images = to_disp_images(images)
    for idx, (disp, val) in enumerate(zip(new_images, ids)):
        if gt_ids is not None:
            color = (0,1,0) if ids[idx] == gt_ids[idx] else (1,0,0)
        # if val != 0:
        pos = get_pos_in_image(loc, size, disp.shape)
        cv2.putText(disp, str(val), pos, cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)
    return new_images


def add_error_to_images(images, errors, loc='bl', size=0.65, vmin=0., vmax=30.0, thickness=1,
                        format_string='{:.1f}', colors=None):
    new_images = to_disp_images(images)
    if colors is None:
        colors = color_map(to_numpy(errors), cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        if images[0].dtype == np.uint8:
            colors *= 255
    for disp, err, color in zip(new_images, errors, colors):
        pos = get_pos_in_image(loc, size, disp.shape)
        cv2.putText(disp, format_string.format(err), pos, cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)
    return new_images


def overlay_heatmap(img, hm, heatmap_opacity=0.45):
    img_dtype = img.dtype
    img_new = img.copy()
    if img_new.dtype == np.uint8:
        img_new = img_new.astype(np.float32) / 255.0

    hm_colored = color_map(hm**1, vmin=0, vmax=1.0, cmap=plt.cm.inferno)
    if len(hm.shape) > 2:
        mask = cv2.blur(hm, ksize=(3, 3))
        print('mask', mask.dtype)
        mask = mask.mean(axis=2)
        mask = mask > 0.05
        for c in range(3):
            # img_new[...,c] = img[...,c] + hm[...,c]
            img_new[..., c][mask] = img[..., c][mask] * 0.7 + hm[..., c][mask] * 0.3
    else:
        if hm_colored.shape != img.shape:
            hm_colored = cv2.resize(hm_colored, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        img_new = img_new + hm_colored * heatmap_opacity

    img_new = img_new.clip(0, 1)
    if img_dtype == np.uint8:
        img_new = cvt32FtoU8(img_new)
    assert img_new.dtype == img.dtype
    assert img_new.shape == img.shape
    return img_new

