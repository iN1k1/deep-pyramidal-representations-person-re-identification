from .nmnet import NMNet
from .nmnet import AverageMeter
from .pt import utils
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from skimage.transform import resize as imresize
import time
import shutil
import os
import torchnet as tnt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

class PyNet(NMNet):
    def __init__(self):
        super(PyNet, self).__init__()
        self.distributed_train_loader = None

    def _get_num_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _set_data_providers(self, train_data_provider, val_data_provider, train_batch_size, val_batch_size, num_workers=4,
                            train_drop_last_batch=False, val_drop_last_batch=False,
                            train_collate_fn=default_collate, val_collate_fn=default_collate):
        train_sampler = None
        if self.is_distributed:
            train_sampler = DistributedSampler(train_data_provider, num_replicas=self.world_size, rank=self.rank)

        if train_collate_fn is None:
            train_collate_fn = default_collate

        if val_collate_fn is None:
            val_collate_fn = default_collate

        if train_data_provider is not None:
            self.train_loader = DataLoader(train_data_provider, batch_size=train_batch_size,
                                       shuffle=(train_sampler is None), num_workers=num_workers, pin_memory=True,
                                       drop_last=train_drop_last_batch,
                                       sampler=train_sampler, collate_fn=train_collate_fn)

        if val_data_provider is not None:
            self.val_loader = DataLoader(val_data_provider, batch_size=val_batch_size, shuffle=False,
                                         drop_last=val_drop_last_batch,
                                         num_workers=num_workers, pin_memory=True,
                                         collate_fn=val_collate_fn)

    def _forward(self, input, **kwargs):
        if len(kwargs) > 0 and kwargs['extra_pars'] is not None:
            return self.model(input, **kwargs)
        return self.model(input)

    def _loss(self, input, target, **kwargs):
        if len(kwargs) > 0 and kwargs['extra_pars'] is not None:
            return self.criterion(input, target, **kwargs)
        return self.criterion(input, target)

    def forward_embedding(self, tensor):
        return self.forward(tensor)

    def to_gpu(self):
        device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")
        print(' ==> Moving the network model to {}'.format(device))

        # If the model is trained with the distributed architecture, it has already been transferred to CUDA
        if not self.is_distributed:
            self.model.to(device)
        if self.criterion is not None:
            self.criterion.to(device)

    def get_preprocessed_tensor(self, numpy_image, mu, std, size):
        transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mu, std)])
        numpy_image = imresize(numpy_image, (size[0], size[1], size[2])).astype(dtype=numpy_image.dtype)
        im1 = transform(numpy_image)
        im1.resize_(1, size[2], size[0], size[1])
        if self.use_gpu:
            im1 = im1.to("cuda")
        return im1

    def extract_features(self, tensor, layer_names=None, linearize=False):
        if layer_names is None:
            layer_names = self.embedding_layers
        self.model.eval()
        self.set_extract_features_layers(layer_names)
        self.forward_embedding(tensor)
        return self.get_embeddings(linearize=linearize)

    def clear_embedding_listeners(self):

        # Remove all embedding hooks
        for ha in self.embedding_hook_handles:
            ha.remove()
        self.embedding_hook_handles = []

    def set_embeddings_listeners(self, layer_names):

        # Clear any previous set embedding listener
        self.clear_embedding_listeners()

        # Reset embedding dicts
        self.embedding = []
        embedding = dict()
        id_module_mapping = dict()

        # Loop over embedding layers
        for ii, layer in enumerate(layer_names):
            m = self.model
            n_dims = 1

            # Handle parallel nets
            if isinstance(m, torch.nn.DataParallel):
                m = m.module
                n_dims = len(self.model.device_ids)

            # Get down to the selected layer
            for ll in layer.split('\\'):
                m = m._modules.get(ll)

            # Initialize the embedding tensors
            # id_module_mapping[id(m)] = layer
            # NB => this was the best way.. however if we are using a parallel net, then the id of the modules are different
            # on each replica.. thus embedding[id(m)] will fail at the hooks..
            # To overcome this, we use the module name as string.. hopefully.. we use different names for the layers..
            id_module_mapping[str(m)] = layer

            for ii in range(n_dims):
                key = '{}~~~{}'.format(layer,ii)
                embedding[key] = torch.FloatTensor()

            # Hook function that fills the embedding vectors
            def hook_fun(m, i, o):

                # tvec is a vector of BATCH x FEATS
                tvec = o.to("cpu").clone()
                ii = 0
                if o.is_cuda:
                    ii = o.get_device()
                layer = id_module_mapping[str(m)]
                key = '{}~~~{}'.format(layer,ii)

                # Concatenate on batch dimension
                embedding[key] = torch.cat((embedding[key], tvec), dim=0)

            # Save hook handles
            self.embedding_hook_handles.append(m.register_forward_hook(hook_fun))

        # Link the embedding vectors to the class object
        self.embedding = embedding

    def get_embeddings(self, linearize=False):
        """

        :type linearize: boolean
        """
        embedding = np.array([])
        if not linearize:
            embedding = dict()

        # Get the unique keys that we set up to let the embeddings run on different GPUs (self.model is parallelized..)
        all_keys = [key.split('~~~')[0] for key in self.embedding.keys()]
        # unique_keys = list(set(all_keys)) => does not keep the order!
        unique_keys = []
        for key in all_keys:
            if key not in unique_keys:
                unique_keys.append(key)
        bs = 1
        if self.val_loader is not None:
            bs = self.val_loader.batch_size
        for key in unique_keys:
            valid_keys = [k for k in self.embedding.keys() if k.split('~~~')[0] == key]

            # Get the embedded vector split over the batches and gpus
            emb_whole = [self.embedding[vk] for vk in valid_keys]
            emb = [self.embedding[vk].split(int(bs / len(valid_keys))) for vk in valid_keys]

            # Init embedding vector
            emb_vec = torch.cat(emb_whole,0).zero_()

            # From:to indexes to loop over batches/GPUs
            f = 0
            t = 0

            # Loop over batches
            for jj in range(len(self.val_loader)): # range(len(emb[ii])):

                # Loop over GPUs
                for ii in range(len(valid_keys)):

                    # Copy data
                    try:
                        t = f + emb[ii][jj].size(0)
                        emb_vec[f:t] = emb[ii][jj]
                        f = t
                    except IndexError as e:
                        print(e)

            # Make it a numpy vec
            emb_vec = emb_vec.numpy()

            if linearize:
                if len(embedding) == 0:
                    embedding = emb_vec
                else:
                    embedding = np.concatenate([embedding, emb_vec], axis=1)
            else:
                embedding[key] = emb_vec

        # Clear data store within the embedding listeners
        for key in self.embedding.keys():
            self.embedding[key] = torch.FloatTensor()

        return embedding

    def _init_optimizer(self, params=None):
        """
        Initialize the network optimizer using the options specified in self.opts.optim


        :type params (default=None): the parameters that have to be optimized. If None, the self.model.parameters() will be used
        :return: optimizer (the initialized optimizer) and scheduler (used to decay the learning rate)
        :rtype:
        """
        print(' ==> Creating {} optimizer with lr = {}'.format(self.opts.optim.method, self.opts.optim.lr))
        if params is None:
            params = self.model.parameters()
        optimizer = None
        if self.opts.optim.method == 'SGD':
            optimizer = torch.optim.SGD(params, self.opts.optim.lr, momentum=self.opts.optim.momentum,
                                        weight_decay=self.opts.optim.weight_decay, nesterov=self.opts.optim.nesterov)
        elif self.opts.optim.method == 'Adam':
            optimizer = torch.optim.Adam(params, self.opts.optim.lr, weight_decay=self.opts.optim.weight_decay)

        # Init scheduler
        scheduler = self._init_scheduler(optimizer)

        return optimizer, scheduler

    def _init_scheduler(self, optimizer):
        scheduler = None  # StepLR(self.opt, gamma=0.1, step_size=3)
        print(' ==> Creating {} learning rate scheduler'.format(self.opts.optim.scheduler))
        if self.opts.optim.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                   patience=self.opts.optim.scheduler_args['patience'],
                                                                   factor=self.opts.optim.scheduler_args['factor'])
        elif self.opts.optim.scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        elif self.opts.optim.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=self.opts.optim.scheduler_args['step_size'],
                                                        gamma=self.opts.optim.scheduler_args['gamma'])
        return scheduler

    def _to_var(self, x, is_target=False, binary_classes=None):
        if isinstance(x, list):
            return [self.to_var(xx, is_target=is_target, binary_classes=binary_classes) for xx in x]
        else:
            if is_target and binary_classes is not None:
                x = torch.sparse.torch.eye(binary_classes).index_select(dim=0, index=x)
            if torch.cuda.is_available() and self.use_gpu:
                x = x.to("cuda")
            return x

    def _on_begin_training(self):
        pass

    def train_epoch(self, epoch):

        cudnn.benchmark = True

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (input, target) in enumerate(self.train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            # Transfer to variable
            input_var = self.to_var(input)
            target_var = self.to_var(target, is_target=True, binary_classes=self.binary_target_classes)

            # Get extra model/loss pars
            extra_model_pars = self.hooks['on_get_extra_model_pars'](**{'epoch':epoch, 'iteration':i, 'target':target_var})
            extra_loss_pars = self.hooks['on_get_extra_loss_pars'](**{'epoch':epoch, 'iteration':i, 'target':target_var})

            # Forward
            output = self.forward(input_var, **{'extra_pars':extra_model_pars} )

            # Compute loss
            loss = self.loss(output, target_var, **{'extra_pars': extra_loss_pars})

            # measure accuracy and record loss
            prec, batch_dim = self.accuracy(output, target_var, topk=(1, 5))
            losses.update(loss.item(), batch_dim)
            top1.update(prec[0], batch_dim)
            top5.update(prec[1], batch_dim)

            # compute gradient and do optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Any post forward operation?
            for hook_handle in self.hooks['on_post_forward']:
                hook_handle(input, output, target, loss, prec, epoch, i, True)

            # Display iter information
            if self.verbose:
                self.display_iter_results(input, output, self.opts.disp.freq, i, epoch, len(self.train_loader), batch_time, data_time, losses, top1, top5)

            if self.post_iteration_hook is not None:
                self.post_iteration_hook()

    def _on_begin_epoch(self, epoch):
        # Distributed?
        if self.is_distributed:
            if self.distributed_train_loader is None:
                self.distributed_train_loader = self.train_loader.sampler
            self.distributed_train_loader.set_epoch(epoch)

        # Reset meters
        self.reset_meters()

    def _on_end_epoch(self, epoch, is_train=True):
        # Display epoch results
        if self.verbose and self.meter_accuracy[0].n > 0:
            self.display_epoch_results(self.meter_accuracy[0].value()[0], self.meter_accuracy[1].value()[0])

        # Reset meters
        self.reset_meters()

    def _on_post_forward(self, input, output, target, loss, accuracy, epoch, iter, is_training):
        # Top 1 and top 5 accuracy!
        self.meter_accuracy[0].add(accuracy[0])
        self.meter_accuracy[1].add(accuracy[1])

        # Loss
        self.meter_loss.add(loss.item())

    def _display_results(self, input, output, epoch, iter):
        pass

    def validate_epoch(self, accumulate_outputs=False):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        # Need to accumulate outputs?
        all_outputs = []

        # No gradients needed in the validation phase!
        with torch.no_grad():

            # Loop over inputs
            for i, (input, target) in enumerate(self.val_loader):

                # measure data loading time
                data_time.update(time.time() - end)

                # Transfer to variable
                input_var = self.to_var(input)
                target_var = self.to_var(target, is_target=True, binary_classes=self.binary_target_classes)

                # Get extra model/loss pars
                extra_model_pars = self.hooks['on_get_extra_model_pars']( **{'iteration': i, 'target': target_var})
                extra_loss_pars = self.hooks['on_get_extra_loss_pars']( **{'iteration': i, 'target': target_var})

                # Forward
                output = self.forward(input_var, **{'extra_pars': extra_model_pars})

                # Compute loss
                loss = self.loss(output, target_var, **{'extra_pars': extra_loss_pars})

                # measure accuracy and record loss
                prec, batch_dim = self.accuracy(output, target_var, topk=(1, 5))
                losses.update(loss.item(), batch_dim)
                top1.update(prec[0], batch_dim)
                top5.update(prec[1], batch_dim)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # Any post forward operation?
                for hook_handle in self.hooks['on_post_forward']:
                    hook_handle(input, output, target, loss, prec, 1, i, False)

                # Display iter information
                if self.verbose:
                    self.display_iter_results(input, output, self.opts.disp.freq, i, 0, len(self.val_loader), batch_time, data_time, losses, top1, top5)

                # Need to accumulate outputs?
                if isinstance(output, list) or isinstance(output, tuple):
                    all_outputs.append(output)
                else:
                    all_outputs.append(output.data.to("cpu").numpy().copy())

        # Display epoch results
        if accumulate_outputs:
            return top1.avg, top5.avg, np.concatenate(all_outputs)
        return top1.avg, top5.avg

    def adjust_learning_rate(self, optimizer, epoch):
        """ Adjust the optimizer learning rate according to the instantiated scheduler """
        if self.scheduler is not None:

            # Update scheduler
            self.scheduler.step(epoch)

    def init_meters_and_plots(self, num_classes):
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.meter_accuracy = [tnt.meter.AverageValueMeter() for _ in range(2)] # top 1 and top 5
        self.confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)

    def reset_meters(self):
        if self.is_master:
            if self.meter_accuracy is not None:
                [meter_accuracy.reset() for meter_accuracy in self.meter_accuracy]
            if self.meter_loss is not None:
                self.meter_loss.reset()
            if self.confusion_meter is not None:
                self.confusion_meter.reset()

    def _accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        return utils.accuracy(output, target, topk)

    def save_checkpoint(self, epoch, top1, top5, is_best, filename=None):

        data = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict() if not hasattr(self.model, 'module') else self.model.module.state_dict(), # Handle parallel data..
            'top_1': top1,
            'top_5': top5,
            'optimizer': self.optimizer.state_dict()
        }
        if filename is None:
            filename = 'checkpoint_epoch-{}.pth.tar'.format(epoch)
        filename = os.path.join(self.exp_folder, filename)
        torch.save(data, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.exp_folder, 'checkpoint_best.pth.tar'))
        print(" ==> Saved checkpoint at: '{}'".format(filename))

    def load_checkpoint(self, filename, load_optimizer):
        epoch = 0
        top1 = 0
        top5 = 0
        if filename[0] == '/':
            filename = filename
        elif self.exp_folder is not None:
            filename = os.path.join(self.exp_folder, filename)
        if os.path.isfile(filename):
            print(" ==> Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']-1 # Epochs when training starts from 0..
            top1 = checkpoint['top_1']
            top5 = checkpoint['top_5']
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_from_checkpoint(checkpoint)
            else:
                self.model.load_from_checkpoint(checkpoint)

            # Load optimizer
            if load_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

                # re-initialize the lr scheculer
                self.scheduler = self._init_scheduler(self.optimizer)
                self.scheduler.step(epoch)

            else:
                epoch = 0
                top1 = 0
                top5 = 0

            # Call on_end_epoch handlers (might need to update somthing with the current epoch..)
            for on_end_epoch_fun_handle in self.hooks['on_end_epoch']:
                on_end_epoch_fun_handle(epoch)

        else:
            print(" ==> no checkpoint found at '{}'".format(filename))

        return epoch, top1, top5
