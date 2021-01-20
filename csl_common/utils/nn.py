import json
import os

import numpy as np
import torch


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def to_numpy(ft):
    if ft is None:
        return None
    if isinstance(ft, np.ndarray):
        return ft
    try:
        return ft.detach().cpu().numpy()
    except AttributeError:
        pass
    return np.array(ft)


def to_image(m):
    img = to_numpy(m)
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0)).copy()
    return img


def unsqueeze(x):
    if isinstance(x, np.ndarray):
        return x[np.newaxis, ...]
    else:
        return x.unsqueeze(dim=0)


def atleast4d(x):
    if len(x.shape) == 3:
        return unsqueeze(x)
    return x


def atleast3d(x):
    if len(x.shape) == 2:
        return unsqueeze(x)
    return x


class Batch:

    def __init__(self, data, n=None, gpu=True, eval=False):
        self.images = atleast4d(data['image'])
        try:
            self.masks = atleast4d(data['mask'])
        except KeyError:
            self.masks = None

        self.eval = eval

        try:
            self.ids = data['id']
            try:
                if self.ids.min() < 0 or self.ids.max() == 0:
                    self.ids = None
            except AttributeError:
                self.ids = np.array(self.ids)
        except KeyError:
            self.ids = None

        try:
            self.target_images = data['target_image']
        except KeyError:
            self.target_images = None

        try:
            self.clips = np.array(data['vid'])
        except KeyError:
            self.clips = None

        try:
            self.fnames = data['fnames']
        except:
            self.fnames = None

        for k, v in self.__dict__.items():
            if v is not None:
                try:
                    self.__dict__[k] = v[:n]
                except TypeError:
                    pass

        if gpu:
            for k, v in self.__dict__.items():
                if v is not None:
                    try:
                        self.__dict__[k] = v.cuda()
                    except AttributeError:
                        pass

    def __len__(self):
        return len(self.images)


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def read_model(in_dir, model_name, model):
    filepath_mdl = os.path.join(in_dir, model_name+'.mdl')
    snapshot = torch.load(filepath_mdl)
    try:
        model.load_state_dict(snapshot['state_dict'], strict=False)
    except RuntimeError as e:
        print(e)


def read_meta(in_dir):
    with open(os.path.join(in_dir, 'meta.json'), 'r') as outfile:
        data = json.load(outfile)
    return data


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