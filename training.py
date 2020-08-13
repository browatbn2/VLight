import time

import datetime
import cv2
import json
import os
import matplotlib.pyplot as plt
import sklearn.utils
import numpy as np

import torch
import torch.utils.data as td
import torch.nn.modules.distance

from csl_common import vis
from csl_common.utils import nn, io_utils
from csl_common.utils.nn import to_numpy, Batch, set_requires_grad, count_parameters
import csl_common.utils.log as log
import config as cfg


eps = 1e-8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENCODING_DISTRIBUTION = 'normal'
TRAIN = 'train'
VAL = 'val'


# save some samples to visualize the training progress
def get_fixed_samples(ds, num):
    dl = td.DataLoader(ds, batch_size=num, shuffle=False, num_workers=0)
    data = next(iter(dl))
    return Batch(data, n=num)


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.xavier_uniform(m)
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)


class Training(object):

    def __init__(self, datasets, args, snapshot_dir=cfg.SNAPSHOT_DIR):

        self.args = args
        self.session_name = args.sessionname
        self.datasets = datasets
        self.net = self._get_network(pretrained=False)

        log.info("Learning rate: {}".format(self.args.lr))

        self.snapshot_dir = snapshot_dir
        self.total_iter = 0
        self.total_images = 0
        self.iter_in_epoch = 0
        self.epoch = 0
        self.best_score = 999
        self.epoch_stats = []

        if ENCODING_DISTRIBUTION == 'normal':
            self.enc_rand = torch.randn
            self.enc_rand_like = torch.randn_like
        elif ENCODING_DISTRIBUTION == 'uniform':
            self.enc_rand = torch.rand
            self.enc_rand_like = torch.rand_like
        else:
            raise ValueError()

        self.total_training_time_previous = 0
        self.time_start_training = time.time()

        snapshot = args.resume
        if snapshot is not None:
            log.info("Resuming session {} from snapshot {}...".format(self.session_name, snapshot))
            self._load_snapshot(snapshot)

        self.net = self.net.cuda()

        log.info("Total model params: {:,}".format(count_parameters(self.net)))

        n_fixed_images = 10
        self.fixed_batch = {}
        for phase in datasets.keys():
            self.fixed_batch[phase] = get_fixed_samples(datasets[phase], n_fixed_images)

    def _get_network(self, pretrained):
        raise NotImplementedError

    def _run_batch(self, data, eval):
        raise NotImplementedError

    def _run_epoch(self, dataloader, eval=False):
        self.iters_per_epoch = len(dataloader.dataset) // dataloader.batch_size
        self.iter_starttime = time.time()
        self.iter_in_epoch = 0

        for data in dataloader:
            self._run_batch(data, eval=eval)
            self.total_iter += 1
            self.net.total_iter = self.total_iter
            self.iter_in_epoch += 1

    def _save_snapshot(self, is_best=False):
        def write_model(out_dir, model_name, model):
            filepath_mdl = os.path.join(out_dir, model_name+'.mdl')
            snapshot = {
                        'arch': type(model).__name__,
                        'input_size': self.args.input_size,
                        'state_dict': model.state_dict(),
                        }
            io_utils.makedirs(filepath_mdl)
            torch.save(snapshot, filepath_mdl)

        def write_meta(out_dir):
            with open(os.path.join(out_dir, 'meta.json'), 'w') as outfile:
                data = {'epoch': self.epoch+1,
                        'total_iter': self.total_iter,
                        'total_images': self.total_images,
                        'total_time': self.total_training_time(),
                        'best_score': self.best_score}
                json.dump(data, outfile)

        model_data_dir = os.path.join(self.snapshot_dir, self.session_name)
        model_snap_dir =  os.path.join(model_data_dir, '{:05d}'.format(self.epoch+1))
        write_model(model_snap_dir, 'saae', self.net)
        write_meta(model_snap_dir)

        # save a copy of this snapshot as the best one so far
        if is_best:
            io_utils.copy_files(src_dir=model_snap_dir, dst_dir=model_data_dir, pattern='*.mdl')

    def _load_snapshot(self, snapshot_name, data_dir=None):
        if data_dir is None:
            data_dir = self.snapshot_dir

        model_snap_dir = os.path.join(data_dir, snapshot_name)
        try:
            nn.read_model(model_snap_dir, 'saae', self.net)
        except KeyError as e:
            print(e)

        meta = nn.read_meta(model_snap_dir)
        self.epoch = meta['epoch']
        self.total_iter = meta['total_iter']
        self.total_training_time_previous = meta.get('total_time', 0)
        self.total_images = meta.get('total_images', 0)
        self.best_score = meta['best_score']
        self.net.total_iter = self.total_iter
        str_training_time = str(datetime.timedelta(seconds=self.total_training_time()))
        log.info("Model {} trained for {} iterations ({}).".format(snapshot_name, self.total_iter, str_training_time))

    def _is_snapshot_iter(self):
        return (self.total_iter+1) % self.args.snapshot_interval == 0 and (self.total_iter+1) > 0

    def _print_interval(self, eval):
        return self.args.print_freq_eval if eval else self.args.print_freq

    def _is_printout_iter(self, eval):
        return (self.iter_in_epoch+1) % self._print_interval(eval) == 0

    def _is_eval_epoch(self):
        return (self.epoch+1) % self.args.eval_freq == 0 and VAL in self.datasets

    def _training_time(self):
        return int(time.time() - self.time_start_training)

    def total_training_time(self):
        return self.total_training_time_previous + self._training_time()


def bool_str(x):
    return str(x).lower() in ['True', 'true', '1']

def add_arguments(parser, defaults=None):

    if defaults is None:
        defaults = {}

    # model params
    parser.add_argument('--sessionname',  default=defaults.get('sessionname'), type=str, help='output filename (without ext)')
    parser.add_argument('-r', '--resume', default=defaults.get('resume'), type=str, metavar='PATH', help='path to snapshot (default: None)')
    parser.add_argument('-i','--input-size', default=defaults.get('input_size', 256), type=int, help='CNN input size')

    # training
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('-e', '--epochs', default=None, type=int, metavar='N', help='maximum epoch count')
    parser.add_argument('-b', '--batchsize', default=defaults.get('batchsize', 50), type=int, metavar='N', help='batch size')
    parser.add_argument('--eval', default=False, action='store_true',  help='run evaluation instead of training')
    parser.add_argument('--phases', default=[TRAIN, VAL], nargs='+')
    parser.add_argument('--lr', default=defaults.get('lr', 0.0001), type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam beta 1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam beta 2')

    # reporting
    parser.add_argument('--save-freq', default=defaults.get('save_freq', 1), type=int, metavar='N', help='save snapshot every N epochs')
    parser.add_argument('--print-freq', '-p', default=defaults.get('print_freq', 20), type=int, metavar='N', help='print every N steps')
    parser.add_argument('--print-freq-eval', default=defaults.get('print_freq_eval', 1), type=int, metavar='N', help='print every N steps')
    parser.add_argument('--eval-freq', default=defaults.get('eval_freq', 10), type=int, metavar='N', help='evaluate every N steps')
    parser.add_argument('--batchsize-eval', default=defaults.get('batchsize_eval', 20), type=int, metavar='N', help='batch size for evaluation')

    # data
    parser.add_argument('--train-count', default=defaults.get('train_count', None), type=int, help='number of training images per dataset')
    parser.add_argument('--train-count-multi', default=None, type=int, help='number of total training images for training using multiple datasets')
    parser.add_argument('--val-count',  default=None, type=int, help='number of test images')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N', help='number of data loading workers (default: 6)')
    parser.add_argument('--workers_eval', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')

    # visualization
    parser.add_argument('--show', type=bool_str, default=True, help='visualize training')
    parser.add_argument('--wait', default=10, type=int)

