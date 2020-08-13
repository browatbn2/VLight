import os
import time
import pandas as pd
import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn.modules.distance
import torch.nn.functional as F
import torch.utils.data as td

import csl_common.utils.log as log

import retinadataset
import eval_vessels
import unet
import vlight

import training
from csl_common.utils.nn import Batch, to_numpy
import torch.optim as optim

import retina_vis
import albumentations as alb
from albumentations.pytorch import transforms as alb_torch

TRAIN = 'train'
VAL = 'val'


def get_image_PRs(vessels, masks):
    return [eval_vessels.calculate_metrics(pred, mask, verbose=False)['PR']
            for pred, mask in zip(vessels, masks)]


class VesselTraining(training.Training):
    def __init__(self, datasets, params, **kwargs):
        super().__init__(datasets, params, **kwargs)

        betas = (self.args.beta1, self.args.beta2)
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr, betas=betas)

        self.dataloaders = {}
        if TRAIN in self.datasets:
            self.dataloaders[TRAIN] = td.DataLoader(self.datasets[TRAIN], batch_size=self.args.batchsize,
                                                    num_workers=self.args.workers, drop_last=True, shuffle=True)
        if VAL in self.datasets:
            self.dataloaders[VAL] = td.DataLoader(self.datasets[VAL], batch_size=self.args.batchsize_eval, num_workers=0)
            common.init_random(0)
            self.fixed_val_data = []
            for data in self.dataloaders[VAL]:
                self.fixed_val_data.append(data)

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True,
        #                                                             factor=0.25, min_lr=0.000025)

    def _get_network(self, pretrained):
        # return unet.UNet(n_channels=3, n_classes=1).cuda()
        return vlight.VLight(n_channels=3, n_classes=1).cuda()

    def _print_iter_stats(self, stats):
        means = pd.DataFrame(stats).mean().to_dict()
        current = stats[-1]
        str_stats = '[{ep}][{i}/{iters_per_epoch}] loss={avg_loss:.4f} PR={avg_PR:.3f} {t_data:.2f}/{t_proc:.3f}/{t:.2f}s ({total_iter:06d} {total_time})'
        log.info(str_stats.format(
            ep=current['epoch'] + 1, i=current['iter'] + 1, iters_per_epoch=self.iters_per_epoch,
            avg_loss=means.get('loss', -1),
            avg_PR=means.get('avg_PR', -1),
            t=means['iter_time'],
            t_data=means['time_dataloading'],
            t_proc=means['time_processing'],
            total_iter=self.total_iter + 1, total_time=str(datetime.timedelta(seconds=self._training_time()))
        ))

    def _print_epoch_summary(self, epoch_stats, epoch_starttime):
        means = pd.DataFrame(epoch_stats).mean().to_dict()
        duration = int(time.time() - epoch_starttime)
        log.info("{}".format('-' * 100))
        str_stats = '          loss={avg_loss:.4f} PR={avg_PR:.3f} \tT: {time_epoch}'
        log.info(str_stats.format(
            iters_per_epoch=self.iters_per_epoch,
            avg_loss=means.get('loss', -1),
            avg_PR=means.get('avg_PR', -1),
            t=means['iter_time'],
            t_data=means['time_dataloading'],
            t_proc=means['time_processing'],
            total_iter=self.total_iter + 1, total_time=str(datetime.timedelta(seconds=self._training_time())),
            time_epoch=str(datetime.timedelta(seconds=duration))))

    def train(self, num_epochs):

        log.info("")
        log.info("Starting training session '{}'...".format(self.session_name))
        # log.info("")

        while num_epochs is None or self.epoch < num_epochs:
            log.info('')
            log.info('=' * 5 + ' Epoch {}/{}'.format(self.epoch+1, num_epochs))

            self.epoch_stats = []
            epoch_starttime = time.time()

            self.net.train()
            self._run_epoch(self.dataloaders[TRAIN])

            # save model every few epochs
            if (self.epoch+1) % self.args.save_freq == 0:
                log.info("*** saving snapshot *** ")
                self._save_snapshot(is_best=False)

            # print average loss and accuracy over epoch
            self._print_epoch_summary(self.epoch_stats, epoch_starttime)

            if self._is_eval_epoch():
                self.evaluate()

            self.epoch += 1

        time_elapsed = time.time() - self.time_start_training
        log.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    def evaluate(self):
        log.info("")
        log.info("Evaluating '{}'...".format(self.session_name))
        # log.info("")

        self.iters_per_epoch = len(self.fixed_val_data)
        self.iter_in_epoch = 0
        self.iter_starttime = time.time()
        epoch_starttime = time.time()
        epoch_stats = []

        self.net.eval()

        for data in self.fixed_val_data:
            batch = Batch(data, eval=True)
            targets = batch.masks.float()

            time_proc_start = time.time()
            time_dataloading = time.time() - self.iter_starttime
            with torch.no_grad():
                X_vessels = self.net(batch.images)
            loss = F.binary_cross_entropy(X_vessels, targets)

            iter_stats = {
                'loss': loss.item(),
                'epoch': self.epoch,
                'timestamp': time.time(),
                'time_dataloading': time_dataloading,
                'time_processing': time.time() - time_proc_start,
                'iter_time': time.time() - self.iter_starttime,
                'iter': self.iter_in_epoch,
                'total_iter': self.total_iter,
                'batch_size': len(batch)
            }
            epoch_stats.append(iter_stats)

            if self._is_printout_iter(eval=True):
                nimgs = 1
                avg_PR = eval_vessels.calculate_metrics(X_vessels, targets)['PR']
                PRs = get_image_PRs(X_vessels[:nimgs], targets[:nimgs])
                iter_stats.update({'avg_PR': avg_PR})
                self._print_iter_stats(epoch_stats[-self._print_interval(True):])

                #
                # Batch visualization
                #
                if self.args.show:
                    retina_vis.visualize_vessels(batch.images, batch.images, vessel_hm=targets, scores=PRs,
                                                 pred_vessel_hm=X_vessels, wait=self.args.wait, f=1.0,
                                                 overlay_heatmaps_recon=True, nimgs=nimgs, horizontal=True)

            self.iter_starttime = time.time()
            self.iter_in_epoch += 1

        # print average loss and accuracy over epoch
        self._print_epoch_summary(epoch_stats, epoch_starttime)

        # update scheduler
        means = pd.DataFrame(epoch_stats).mean().to_dict()
        val_loss = means['loss']
        val_PR = means['avg_PR']
        # self.scheduler.step(val_loss, self.epoch)


    def _run_epoch(self, dataloader, eval=False):
        self.iters_per_epoch = len(dataloader.dataset) // dataloader.batch_size
        self.iter_starttime = time.time()
        self.iter_in_epoch = 0

        for data in dataloader:
            self._run_batch(data, eval=eval)
            self.total_iter += 1
            self.net.total_iter = self.total_iter
            self.iter_in_epoch += 1


    def _run_batch(self, data, eval=False, ds=None):
        time_dataloading = time.time() - self.iter_starttime
        time_proc_start = time.time()
        iter_stats = {'time_dataloading': time_dataloading}

        batch = Batch(data, eval=eval, gpu=True)

        targets = batch.masks.float()
        images = batch.images

        self.net.zero_grad()

        with torch.set_grad_enabled(not eval):
            X_vessels = self.net(images)

        loss = F.binary_cross_entropy(X_vessels, targets)

        loss.backward()
        self.optimizer.step()

        # statistics
        iter_stats.update({
            'loss': loss.item(),
            'epoch': self.epoch,
            'timestamp': time.time(),
            'iter_time': time.time() - self.iter_starttime,
            'time_processing': time.time() - time_proc_start,
            'iter': self.iter_in_epoch,
            'total_iter': self.total_iter,
            'batch_size': len(batch)
        })
        self.iter_starttime = time.time()
        self.epoch_stats.append(iter_stats)

        # print stats every N mini-batches
        if self._is_printout_iter(eval):
            nimgs = 5
            avg_PR = eval_vessels.calculate_metrics(X_vessels, targets)['PR']
            PRs = get_image_PRs(X_vessels[:nimgs], targets[:nimgs])
            iter_stats.update({'avg_PR': avg_PR})

            self._print_iter_stats(self.epoch_stats[-self._print_interval(eval):])

            # Batch visualization
            if self.args.show:
                retina_vis.visualize_vessels(images, images, vessel_hm=targets, scores=PRs,
                                             pred_vessel_hm=X_vessels, ds=ds, wait=self.args.wait, f=1.0,
                                             overlay_heatmaps_recon=True, nimgs=1, horizontal=True)



def run():

    if args.seed is not None:
        from csl_common.utils.common import init_random
        init_random(args.seed)

    # log.info(json.dumps(vars(args), indent=4))

    full_sizes = {
        'drive': 512,
        'stare': 512,
        'chase': 1024,
        'hrf': 2560,
    }
    full_size = full_sizes[args.dataset_train[0]]

    transform_train = alb.Compose([
        alb.Rotate(60, border_mode=cv2.BORDER_CONSTANT),

        alb.RandomSizedCrop(
            min_max_height=(int(full_size*0.25), int(full_size*0.5)),
            height=args.input_size,
            width=args.input_size,
            p=1.0
        ),

        # alb.RandomSizedCrop(
        #     min_max_height=(int(full_size*0.5), int(full_size*0.5)),
        #     height=1600, width=1600,
        # ),

        # alb.Resize(width=565*2, height=584*2),
        # alb.RandomCrop(args.input_size, args.input_size),

        # alb.CenterCrop(args.input_size, args.input_size),
        # alb.Resize(args.input_size, args.input_size),

        alb.RGBShift(p=0.5),
        alb.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
        alb.RandomGamma(),

        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb_torch.ToTensor(normalize=dict(mean=[0.518, 0.418, 0.361], std=[1, 1, 1]))
    ])

    transform_val = alb.Compose([
        alb.RandomSizedCrop(
            min_max_height=(int(full_size * 0.25), int(full_size * 0.5)),
            height=args.input_size, width=args.input_size
        ),
        alb.Resize(args.input_size, args.input_size, always_apply=True),
        alb_torch.ToTensor(normalize=dict(mean=[0.518, 0.418, 0.361], std=[1, 1, 1]))
    ])

    torch.backends.cudnn.benchmark = True

    datasets = {}
    datasets[VAL] = retinadataset.create_dataset_multi(args.dataset_val, transform_val,
                                                       num_samples=args.val_count, repeat_factor=5,
                                                       train=False)

    if args.eval:
        fntr = VesselTraining(datasets, args)
        fntr.evaluate()
    else:
        datasets[TRAIN] = retinadataset.create_dataset_multi(args.dataset_train, transform_train,
                                                             num_samples=args.train_count, train=True,
                                                             repeat_factor=args.n_dataset_repeats)
        fntr = VesselTraining(datasets, args)
        fntr.train(num_epochs=args.epochs)


if __name__ == '__main__':

    import sys
    import configargparse
    from training import bool_str
    from csl_common.utils import common

    common.init_random(0)

    np.set_printoptions(linewidth=np.inf)

    # Disable traceback on Ctrl+c
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = configargparse.get_argument_parser()
    defaults = dict(
        input_size=512,
        print_freq=50,
        eval_freq=1,
        batchsize=10,
        save_freq=1,
        batchsize_eval=25,
        print_freq_eval=2,
        lr=0.0001
    )
    training.add_arguments(parser, defaults)

    parser.add_argument('--n-dataset-repeats', default=100, type=int, help='upscale numper of datapoints')
    parser.add_argument('--dataset-train', default=['drive'], type=str, nargs='+', help='dataset(s) for training.')
    parser.add_argument('--dataset-val', default=['drive'], type=str, nargs='+', help='dataset(s) for training.')
    args = parser.parse_known_args()[0]

    # args.wait = 0
    # args.eval = True
    if args.eval:
        args.workers = 0
        args.wait = 0

    if args.sessionname is None:
        if args.resume:
            modelname = os.path.split(args.resume)[0]
            args.sessionname = modelname
        else:
            args.sessionname = 'debug'

    try:
        run()
    except BrokenPipeError:
        pass
