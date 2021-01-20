import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import skimage.io
import torchvision.datasets as tdv
import torch.utils.data as td
import albumentations as alb

batch_crop_size = 512


class RepeatDataset(td.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        factor (float): Repeat the Dataset that many times
    """
    _repr_indent = 4

    def __init__(self, dataset, factor):
        assert factor > 0
        self.dataset = dataset
        self.factor = factor

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

    def __len__(self):
        return int(len(self.dataset) * self.factor)

    def __str__(self):
        str = self.dataset.__str__()
        pad = ' ' * self._repr_indent
        str += f'\n{pad}Repeat dataset factor: {self.factor} (=> {self.__len__()} total datapoints)'
        return str


class Subset(td.Subset):
    def __str__(self):
        str = self.dataset.__str__()
        str += f'\nSubset indices: {self.indices}'
        return str


class DRIVE(tdv.VisionDataset):

    def __init__(self, root, train, transform=None, **kwargs):
        super().__init__(root, transform=transform)

        self.train = train
        self.masks_1st = {}
        self.masks_2nd = {}
        self.fov_masks = {}
        self.images = {}

        if self.train:
            ids = range(21, 41)
            self.image_file_pattern = '{:02d}_training.tif'
        else:
            ids = range(1, 21)
            self.image_file_pattern = '{:02d}_test.tif'

        self.mask_file_pattern = '{:02d}_manual{}.gif'
        self.fov_file_pattern = '{:02d}_{}_mask.gif'

        annotator_id = 1

        for id in ids:
            train_test = 'training' if self.train else 'test'

            img_path = os.path.join(root, train_test, 'images', self.image_file_pattern.format(id))
            mask1st_path = os.path.join(root, train_test, '1st_manual', self.mask_file_pattern.format(id, annotator_id))
            fov_path = os.path.join(root, train_test, 'mask', self.fov_file_pattern.format(id, train_test))

            image = skimage.io.imread(img_path)
            fov = skimage.io.imread(fov_path).astype(np.uint8) // 255
            image[fov == 0,:] = 0

            self.images[id] = image
            self.fov_masks[id] = fov
            self.masks_1st[id] = skimage.io.imread(mask1st_path) #// 255

            if not self.train:
                mask2nd_path = os.path.join(root, train_test, '2nd_manual', self.mask_file_pattern.format(id, 2))
                self.masks_2nd[id] = skimage.io.imread(mask2nd_path)

        if self.transform is None:
            self.transform = alb.Compose([])


    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):

        img_ids = list(self.images.keys())
        img_id = img_ids[item]

        image = self.images[img_id]
        mask = self.masks_1st[img_id]

        sample = self.transform(image=image, mask=mask)

        sample['fname'] = img_id

        return sample


class RetinaSegmentationDataset(tdv.VisionDataset):

    def __init__(self, root, train, transform=None):
        self.train = train
        self.masks_1st = {}
        self.masks_2nd = {}
        self.fov_masks = {}
        self.images = {}
        if transform is None:
           transform = alb.Compose([])
        super().__init__(root, transform=transform)

    def image_size(self):
        raise NotImplementedError

    @staticmethod
    def create_border_mask(img, threshold):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 3)
        mask = gray > threshold
        show_mask = False
        if show_mask:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            ax[0].imshow(gray)
            ax[1].imshow(mask)
            plt.tight_layout()
            plt.show()
        return mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_ids = list(self.images.keys())
        img_id = img_ids[idx]

        image = self.images[img_id]
        mask = self.masks_1st[img_id]

        sample = self.transform(image=image, mask=mask)
        sample['fname'] = img_id

        return sample


class CHASE(RetinaSegmentationDataset):

    def __init__(self, root, train, transform=None):
        super().__init__(root, train, transform=transform)

        if self.train:
            ids = range(1, 11)
        else:
            ids = range(11, 15)

        self.image_file_pattern = 'Image_{:02d}{}'

        for id in ids:
            for side in ['L', 'R']:
                fname = self.image_file_pattern.format(id, side)

                image = skimage.io.imread(os.path.join(root, fname+'.jpg'))
                mask = skimage.io.imread(os.path.join(root, fname+'_1stHO.png'))
                fov = self.create_border_mask(image, threshold=6)
                image[fov == 0,:] = 0

                self.images[fname] = image
                self.fov_masks[fname] = fov
                self.masks_1st[fname] = mask

                if not self.train:
                    self.masks_2nd[fname] = skimage.io.imread(os.path.join(root, fname + '_2ndHO.png'))

    def image_size(self):
        return 1024


class HRF(RetinaSegmentationDataset):

    def __init__(self, root, train, transform=None):
        super().__init__(root, train, transform=transform)

        if self.train:
            ids = range(1, 6)
        else:
            ids = range(6, 16)

        self.image_file_pattern = '{:02d}_{}'

        for id in ids:
            for type in ['dr', 'g', 'h']:
                fname = self.image_file_pattern.format(id, type)

                jpg_str = '.JPG' if type == 'dr' else '.jpg'
                image = skimage.io.imread(os.path.join(root, 'images', fname + jpg_str))

                self.fov_masks[fname] = skimage.io.imread(os.path.join(root, 'mask', fname+'_mask.tif'),
                                                       as_gray=True).astype(np.uint8)
                image[self.fov_masks[fname] == 0, :] = 0

                self.images[fname] = image
                self.masks_1st[fname] = skimage.io.imread(os.path.join(root, 'manual1', fname+'.tif'),
                                                       as_gray=True).astype(np.uint8)

    def image_size(self):
        return 2560


import config as cfg
cfg.register_dataset(DRIVE)
cfg.register_dataset(CHASE)
cfg.register_dataset(HRF)


def create_dataset(dataset_name, train, transform=None, indices=None, repeat_factor=None):
    root, cache_root = cfg.get_dataset_paths(dataset_name)
    dataset_cls = cfg.get_dataset_class(dataset_name)

    ds = dataset_cls(root=root, train=train, transform=transform, )

    if indices is not None:
        indices = [i for i in indices if i < len(ds)]
        ds = Subset(ds, indices)

    if repeat_factor is not None:
        ds =  RepeatDataset(ds, repeat_factor)

    return ds


def create_dataset_multi(dsnames, transform, num_samples=None, indices=None, train=False, repeat_factor=None):
    assert num_samples is None or indices is None, "num_sample and indices can not both be defines"
    if indices is None:
        try:
            indices = range(num_samples)
        except TypeError:
            indices = None

    datasets_for_phase = []
    for name in dsnames:
        ds = create_dataset(name, train, transform, indices, repeat_factor)
        datasets_for_phase.append(ds)

    is_single_dataset = isinstance(dsnames, str) or len(dsnames) == 1

    if is_single_dataset:
        dataset = datasets_for_phase[0]
    else:
        dataset = td.ConcatDataset(datasets_for_phase)
    print(dataset)
    return dataset

