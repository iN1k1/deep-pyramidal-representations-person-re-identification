"""
************************************************************************************************************************
*  Author: Niki Martinel
*  Date: 2019
*
*  Paper: Aggregating Deep Pyramidal Representations for Person Re-Identifications
*         International Conference on Computer Vision and Pattern Reconition (CVPR) workshop
*
*  [2019]
*  Copyright (C) 2019 Niki Martinel. All Rights Reserved.
*  This file is subject to the terms and conditions defined in
*  file 'LICENSE', which is part of this source code package.
************************************************************************************************************************
"""

import argparse
import os
import copy
from datetime import datetime
import torch
import torchvision.transforms as transforms
from src.ml.net import PyNet, NetOpts, DispOpts, OptimOpts
from src.ml.net.pt import factory as model_factory
from src.configs import *
from src.datamanager import *
from src.datamanager import DataProvider
from src.datamanager import utils as datautils
from PIL import Image
import src.pyrnet.model as reid_model
from src.pyrnet.test import evaluate
from src.pyrnet.sampler import HardTripletSampler

""" ================================================================================================================
         CONFIG
    ============================================================================================================ """
# Arg parser
parser = argparse.ArgumentParser(description='ReID Net')

parser.add_argument('--dataset', default='Market-1501', type=str, metavar='STR', help='dataset name (default: Market-1501)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('--epochs_eval', default=10, type=int, metavar='N', help='number of training epochs after which an evaluation step is run (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int, metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 0.0005)')
parser.add_argument('--nesterov', default=True, type=bool, metavar='B', help='nesterov acceleration (default: true)')
parser.add_argument('--print-freq', '--p', default=20, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--net', default='densenet', type=str, metavar='STR', help='network model (default: densenet)')
parser.add_argument('--depth', default=201, type=int, metavar='N', help='network model depth (default: 201)')
parser.add_argument('--bottleneck-size', default=512, type=int, metavar='N', help='classifier bottleneck size (default: 512)')
parser.add_argument('--pyr-feature-size', default=256, type=int, metavar='N', help='pyramidal maps (default: 256)')
parser.add_argument('--pyr-feature-size-dynamic', default=True, type=bool, metavar='B', help='pyramidal feature size dependent on detail level (default: True)')
parser.add_argument('--pyr-operator', default='max_pool', type=str, metavar='STR', help='pyramidal operator (default: max_pool)')
parser.add_argument('--pyr-levels', default=-1, type=int, metavar='N', help='pyramidal levels (default: -1 => dynamic)')
parser.add_argument('--alpha', default=0.05, type=int, metavar='N', help='alpha weight for id-tripet loss (default: 0.05)')
parser.add_argument('--metric', default='euclidean', type=str, metavar='STR', help='metric (default: euclidean')
parser.add_argument('--pretrained', default=True, type=bool, metavar='B', help='use pre-trained model (default: True)')
parser.add_argument('--optimizer', '--o', default='SGD', type=str, metavar='STR', help='optimizer (default: SGD')
parser.add_argument('--scheduler', default='step', type=str, metavar='STR', help='LR Scheduler (default: step')
parser.add_argument('--reda', default=True, type=bool, metavar='B', help='random erasing data augmentation (default: True')
parser.add_argument('--increase-to-margin', default=True, type=bool, metavar='B', help='Linearly increase the weight to the margin loss (default: True)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', help='filename of checkpoint to load for restart (default: none)')


""" ================================================================================================================
         TRAINING
    ============================================================================================================ """
def train(args, sample_size=(3, 384, 192), img_load_size=(408, 204)):

    """ ----------------------------------------------------------------------------------------------------------------
         DATA
        ------------------------------------------------------------------------------------------------------------ """
    # Imagenet Normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Random erasing?
    reda = None
    if args.reda:
        read_prob = 0.5
        reda = RandomErasing(random_values=tuple([int(0.485*255), int(0.456*255), int(0.406*255)]), probability=read_prob)

    # Data transformations
    transformations = DataTransformer([
        transforms.ColorJitter(brightness=0.1, saturation=0.1, contrast=0.1, hue=0),
        transforms.Resize(img_load_size, interpolation=Image.BICUBIC),
        transforms.RandomCrop(sample_size[1:]),
        reda,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])


    #  Dataset
    dset_opts = DatasetConfig(args.dataset, None, (0.5, 0.5), cam_pair=(-1, -1))
    dset = DatasetReID(dset_opts.name, os.path.join('data', dset_opts.name),
                       im_size=dset_opts.imsize, in_memory=False, keep_aspect_ratio=True)

    # Splits
    dsetTr, dsetTe = dset.split(dset_opts.split, save_load=True, make_each_split_contiguous=True)
    num_classes = len(dsetTr.classes)

    """ ----------------------------------------------------------------------------------------------------------------
             NETWORK
        ------------------------------------------------------------------------------------------------------------ """
    # Init Net
    net = PyNet()

    # Exp folder
    net.exp_folder = os.path.join('data', 'experiments', args.dataset, datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    os.makedirs(net.exp_folder)

    # Net model
    model = reid_model.get_model(args.net, args.depth,  sample_size, num_classes,
                                 bottleneck_size=args.bottleneck_size, pyr_feature_size=args.pyr_feature_size,
                                 pyr_operator=args.pyr_operator,
                                 pyr_feature_size_dynamic=args.pyr_feature_size_dynamic)

    # Parallelize it!
    net.model = model_factory.make_it_parallel(model, 'multigpu')

    # get the number of model parameters
    print(' ==> Number of model parameters: {}'.format(net.get_num_parameters()))

    # Criterion
    net.criterion = reid_model.get_loss(use_triplet=True, alpha=args.alpha)

    # To GPU()
    net.to_gpu()

    # Display /Optimization options
    disp_opts = DispOpts(disp_freq=args.print_freq)
    optim_opts = OptimOpts(method=args.optimizer, epochs=args.epochs, lr=args.lr, nesterov=args.nesterov,
                           momentum=args.momentum, weight_decay=args.weight_decay,
                           scheduler=args.scheduler, scheduler_args={'step_size': 40, 'gamma': 0.1})
    net.opts = NetOpts(disp_opts, optim_opts)

    """ ----------------------------------------------------------------------------------------------------------------
             SAMPLER
        ------------------------------------------------------------------------------------------------------------ """
    # Data loaders
    layer_embeddings = ['emb\\bottleneck1', 'emb\\bottleneck2', 'emb\\bottleneck3', 'emb\\bottleneck4']
    data_provider = HardTripletSampler(dsetTr, loader=datautils.load_image, net=net, metric=args.metric,
                                           transform=transformations, layer_embeddings=layer_embeddings,
                                           sample_size=sample_size[1:], hard_sampling=True)

    # Need to drop the last batch due to the BatchNorm layer which cannot take a batch of size 1... which may happen depending on the args batchsize..
    # Let's also scale the batch size for the number of gpus
    num_gpus = 1 if (not torch.cuda.is_available() or net.use_gpu is False) else torch.cuda.device_count()
    net.set_data_providers(data_provider, None, args.batch_size * num_gpus, None,
                           num_workers=args.workers, train_drop_last_batch=True)
    # Init meters
    net.init_meters_and_plots(num_classes)

    """ ----------------------------------------------------------------------------------------------------------------
             OPTIM PARAMETERS
        ------------------------------------------------------------------------------------------------------------ """

    # Handle pre-trained params differently...
    new_params = list(map(id, net.model.module.emb.pyr1.parameters())) + \
                 list(map(id, net.model.module.emb.pyr2.parameters())) + \
                 list(map(id, net.model.module.emb.pyr3.parameters())) + \
                 list(map(id, net.model.module.emb.pyr4.parameters())) + \
                 list(map(id, net.model.module.emb.bottleneck1.parameters())) + \
                 list(map(id, net.model.module.emb.bottleneck2.parameters())) + \
                 list(map(id, net.model.module.emb.bottleneck3.parameters())) + \
                 list(map(id, net.model.module.emb.bottleneck4.parameters())) + \
                 list(map(id, net.model.module.emb.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in new_params, net.model.parameters())
    net_params = [
        {'params': base_params, 'lr': net.opts.optim.lr/10},
        {'params': net.model.module.emb.pyr1.parameters(), 'lr': net.opts.optim.lr},
        {'params': net.model.module.emb.pyr2.parameters(), 'lr': net.opts.optim.lr},
        {'params': net.model.module.emb.pyr3.parameters(), 'lr': net.opts.optim.lr},
        {'params': net.model.module.emb.pyr4.parameters(), 'lr': net.opts.optim.lr},
        {'params': net.model.module.emb.bottleneck1.parameters(), 'lr': net.opts.optim.lr},
        {'params': net.model.module.emb.bottleneck2.parameters(), 'lr': net.opts.optim.lr},
        {'params': net.model.module.emb.bottleneck3.parameters(), 'lr': net.opts.optim.lr},
        {'params': net.model.module.emb.bottleneck4.parameters(), 'lr': net.opts.optim.lr},
        {'params': net.model.module.emb.classifier.parameters(), 'lr': net.opts.optim.lr}
    ]

    """ ----------------------------------------------------------------------------------------------------------------
             HOOKS
        ------------------------------------------------------------------------------------------------------------ """
    # Update sampler every epoch
    net.hooks['on_begin_epoch'].append(data_provider.update)

    # Linearly increase the difficulty by moving from the ID loss to the margin one (s)
    if args.increase_to_margin:
        margin_step = (1-args.alpha) / args.epochs
        def _increase_id_to_triplet_loss_margin(epoch, **kwargs):
            # No need to udpate during evaluation
            if len(kwargs) > 0 and 'is_train' in kwargs and not kwargs['is_train']:
                return
            net.criterion.alpha = args.alpha + (epoch * margin_step)
        net.hooks['on_end_epoch'].append(_increase_id_to_triplet_loss_margin)


    # Linearly increase the random erasing probability
    if args.reda:
        max_reda_prob = 0.95
        reda_step = (max_reda_prob-read_prob)/args.epochs
        def _increase_reda_probability(epoch, **kwargs):
            # No need to udpate during evaluation
            if len(kwargs) > 0 and 'is_train' in kwargs and not kwargs['is_train']:
                return
            reda.probability = read_prob + (epoch * reda_step)
        net.hooks['on_end_epoch'].append(_increase_reda_probability)

    # Custom accuracy computation for triplet inputs..
    net.hooks['accuracy'] = reid_model.accuracy

    # Validation every X epochs
    def _evaluate(epoch, is_train=True):
        if not is_train and (epoch+1) % args.epochs_eval == 0:
            args_eval = copy.deepcopy(args)
            args_eval.batch_size=256
            args_eval.rerank = False
            evaluate(args_eval, net=net, sample_size=sample_size[1:], dset_train=dsetTr, dset_test=dsetTe,
                     layer_embeddings=layer_embeddings)
    net.hooks['on_end_epoch'].append(_evaluate)

    """ ----------------------------------------------------------------------------------------------------------------
             TRAIN
        ------------------------------------------------------------------------------------------------------------ """
    # Train
    net.train(params=net_params, checkpoint=args.checkpoint, load_optimizer=True)


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
