from abc import ABCMeta, abstractmethod

import os

class NMNet(object):
    """
    
    """

    # Meta class: https://stackoverflow.com/questions/17402622/is-it-possible-in-python-to-declare-that-method-must-be-overridden
    # We want each subclass to have a proper forward_features method implemented!
    __metaclass__ = ABCMeta

    def __init__(self):
        super(NMNet, self).__init__()
        self.model = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.embedding = dict()
        self.embedding_layers = []
        self.use_gpu = True
        self.embedding_hook_handles = []
        self.name = ''
        self.opts = NetOpts()

        # Optimizer +  scheduler for LR adaptation
        self.optimizer = None
        self.scheduler = None

        # Exp folder
        self.exp_folder = None

        # Hooks
        self.post_iteration_hook = None
        self.hooks = self._init_hooks()

        # Do we want the class to be converted into a one-hot vector? => if so, set the below variable to the number of classes
        self.binary_target_classes = None

        # Distributed..
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1

        # Meters
        self.meter_loss = None
        self.meter_accuracy = None
        self.confusion_meter = None

        # Plots
        self.train_loss_logger = None
        self.train_error_logger = None
        self.test_loss_logger = None
        self.test_accuracy_logger = None
        self.confusion_logger = None
        self.input_logger = None
        self.weights_logger = None
        self.train_txt_logger = None

        # Verbose
        self.verbose = True

        # Display string
        # Display string
        self.display_string = 'Epoch: [{0}/{1}][{2}/{3}]\t' \
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'

    def _init_hooks(self):
        return {'forward': self._forward,
        'loss': self._loss,
        'accuracy': self._accuracy,
        'to_var': self._to_var,
        'on_begin_training': [self._on_begin_training],
        'on_begin_epoch': [self._on_begin_epoch],
        'on_end_epoch': [self._on_end_epoch],
        'on_begin_validation': '',
        'on_end_validation': '',
        'on_post_forward': [self._on_post_forward],
        'on_get_extra_model_pars': self._on_get_extra_model_pars,
        'on_get_extra_loss_pars': self._on_get_extra_loss_pars,
        'on_display_results': self._display_results}

    def reset(self):
        self.model = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.embedding = dict()
        self.embedding_layers = []
        self.use_gpu = True
        self.embedding_hook_handles = []
        self.name = ''
        self.opts = NetOpts()

        # Optimizer +  scheduler for LR adaptation
        self.optimizer = None
        self.scheduler = None

        # Exp folder
        self.exp_folder = None

        # Hooks
        self.post_iteration_hook = None
        self.hooks = self._init_hooks()

        # Do we want the class to be converted into a one-hot vector? => if so, set the below variable to the number of classes
        self.binary_target_classes = None

        # Distributed..
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1

        # Meters
        self.meter_loss = None
        self.meter_accuracy = None
        self.confusion_meter = None

        # Plots
        self.train_loss_logger = None
        self.train_error_logger = None
        self.test_loss_logger = None
        self.test_accuracy_logger = None
        self.confusion_logger = None
        self.input_logger = None
        self.weights_logger = None
        self.train_txt_logger = None

        self.verbose = True

        # Display string
        self.display_string = 'Epoch: [{0}/{1}][{2}/{3}]\t' \
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'


    def build(self):
        pass

    def to_gpu(self):
        pass

    def forward(self, input, **kwargs):
        return self.hooks['forward'](input, **kwargs)

    def _forward(self, input, **kwargs):
        pass

    def loss(self, input, target, **kwargs):
        return self.hooks['loss'](input, target, **kwargs)

    def _loss(self, input, target, **kwargs):
        pass

    def accuracy(self, output, target, topk=(1,)):
        return self.hooks['accuracy'](output, target, topk)

    def _accuracy(self, output, target, topk=(1,)):
        pass

    def to_var(self, x, is_target=False, binary_classes=None):
        return self.hooks['to_var'](x, is_target, binary_classes)

    def _to_var(self, x, is_target=False, binary_classes=None):
        pass

    def _on_get_extra_model_pars(self, **kwargs):
        return None

    def _on_get_extra_loss_pars(self, **kwargs):
        return None

    def reset(self):
        pass

    def _get_num_parameters(self):
        pass

    def get_num_parameters(self):
        return self._get_num_parameters()

    def _print_model_summary(self, input_size):
        pass

    def print_model_summary(self, input_size):
        return self._print_model_summary(input_size)

    def get_preprocessed_tensor(self, numpy_image, mu, std, size):
        pass

    def extract_features(self, tensor, layerNames=None, linearize=False):
        pass

    def set_extract_features_layers(self, layer_names):
        self.embedding_layers = layer_names
        self.set_embeddings_listeners(layer_names)

    def clear_embedding_listeners(self):
        pass

    def set_embeddings_listeners(self, layer_names):
        pass

    def get_embeddings(self, linearize=False):
        pass

    # def set_save_info(self, save_folder, experiment_id=None):
    #     self.save_folder = save_folder
    #     if experiment_id is None:
    #         experiment_id = self.experiment_id
    #     self.summary_writer.close()
    #     self.summary_writer = SummaryWriter(log_dir=os.path.join(self.save_folder))

    @abstractmethod
    def _init_optimizer(self, params=None):
        """
        Initialize the network optimizer using the options specified in self.opts.optim

        :type params (default=None): the parameters that have to be optimized. If None, the self.model parameters will be used
        :return: optimizer (the initialized optimizer) and scheduler (used to decay the learning rate)
        :rtype:
        """
        pass

    def adjust_learning_rate(self, optimizer, epoch):
        pass

    def init_meters_and_plots(self, num_classes):
        pass

    def reset_meters(self):
        pass

    def is_master(self):
        return self.rank == 0

    def set_distributed(self, is_distributed, rank, worldsize):
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = worldsize

    @abstractmethod
    def _set_data_providers(self, train_data_provider, val_data_provider, train_batch_size, val_batch_size, num_workers=4, train_drop_last_batch=False, val_drop_last_batch=False, train_collate_fn=None, val_collate_fn=None):
        pass

    def set_data_providers(self, train_data_provider, val_data_provider, train_batch_size, val_batch_size, num_workers=4, train_drop_last_batch=False, val_drop_last_batch=False, train_collate_fn=None, val_collate_fn=None):
        self._set_data_providers(train_data_provider, val_data_provider, train_batch_size, val_batch_size, num_workers, train_drop_last_batch, val_drop_last_batch, train_collate_fn, val_collate_fn)

    def train(self, train_loader=None, params=None, checkpoint=None, load_optimizer=True):
        """

        :param train_loader (default: None): train data loader. If set to None (default), self.train_loader will be used
        :type train_loader:
        :param params (default: None): network parameters to be optimized.
         If set to None (default), self.init_optimizer will take care of them by selecting the ones in self.model
        :type params:
        """
        # Set train loader
        if train_loader is not None:
            self.train_loader = train_loader

        # Setup optimizer + lr scheduler
        self.optimizer, self.scheduler = self._init_optimizer(params=params)

        # Accuracy perf on validation
        top1_best = 0

        # Start epoch
        start_epoch = 0

        # Load checkpoint?
        if checkpoint is not None and checkpoint != '':
            start_epoch, top1_best, top5_best = self.load_checkpoint(checkpoint, load_optimizer)

        # Begin training hook
        for on_begin_training_fun_handle in self.hooks['on_begin_training']:
            on_begin_training_fun_handle()

        # Run for n epochs
        for epoch in range(start_epoch, self.opts.optim.epochs):

            # Run all the 'on_begin_epoch' functions
            for on_begin_epoch_fun_handle in self.hooks['on_begin_epoch']:
                on_begin_epoch_fun_handle(epoch)

            # Update the learning rate..
            self.adjust_learning_rate(self.optimizer, epoch)

            # train for one epoch
            self.train_epoch(epoch)

            # End epoch -- train
            # Run all the 'on_end_epoch' functions (train)

            for on_end_epoch_fun_handle in self.hooks['on_end_epoch']:
                on_end_epoch_fun_handle(epoch)
                #self.hooks['on_end_epoch'](epoch)

            # evaluate on validation set
            top1, top5 = self.validate()

            # End epoch -- validation
            # Run all the 'on_end_epoch' functions (validation)
            for on_end_epoch_fun_handle in self.hooks['on_end_epoch']:
                on_end_epoch_fun_handle(epoch, is_train=False)
                #self.hooks['on_end_epoch'](epoch, is_train=False)

            # remember best prec@1 and save checkpoint
            is_best = top1 > top1_best
            top1_best = max(top1, top1_best)
            self.save_checkpoint(epoch+1, top1, top5, is_best)


    @abstractmethod
    def train_epoch(self, epoch):
        pass

    def _on_begin_training(self):
        pass

    def _on_begin_epoch(self, epoch):
        self.reset_meters()

    def _on_end_epoch(self, epoch, is_train=True):
        pass

    def _on_post_forward(self, input, output, target, loss, accuracy, epoch, iter, is_training):
        pass

    def _display_results(self, input, output, epoch, iter):
        pass

    def display_iter_results(self, input, output, disp_freq, iter, epoch, dset_length, batch_time, data_time, losses, top1, top5):
        if iter % disp_freq == 0:
            str_to_display = self.display_string.format(
                            epoch+1, self.opts.optim.epochs, iter, dset_length, batch_time=batch_time,
                            data_time=data_time, loss=losses, top1=top1, top5=top5)

            print(str_to_display)

            if self.train_txt_logger is not None:
                self.train_txt_logger.log(str_to_display, (epoch * 1000) + iter)

            # Display epoch/iter input output results
            self.hooks['on_display_results'](input, output, epoch, iter)

    def display_epoch_results(self, top1, top5):
        print('Average Epoch Performance: Prec@1 {top1:.3f} Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))

    def validate(self, val_loader=None, accumulate_outputs=False):
        if val_loader is not None:
            self.val_loader = val_loader

        # May happen that there is no validation loader in case we just want to train the net
        if self.val_loader is not None:
            return self.validate_epoch(accumulate_outputs=accumulate_outputs)

        return 0, 0 ## Top1, top5..

    @abstractmethod
    def validate_epoch(self, accumulate_outputs=False):
        pass

    @abstractmethod
    def save_checkpoint(self, epoch, top1, top5, is_best, filename='checkpoint.tar'):
        pass

    @abstractmethod
    def load_checkpoint(self, filename, load_optimizer):
        pass


class NetOpts(object):
    """
    
    """
    def __init__(self, dispOpts=None, optimOpts=None):
        super(NetOpts, self).__init__()
        self.disp = dispOpts
        self.optim = optimOpts
        if self.disp is None:
            self.disp = DispOpts()
        if self.optim is None:
            self.optim = OptimOpts(1)



class DispOpts(object):
    def __init__(self, disp_freq=20, disp_freq_gradients=9999999):
        super(DispOpts, self).__init__()
        self.freq = disp_freq
        self.freq_gradients = disp_freq_gradients


class OptimOpts(object):
    def __init__(self, epochs, method='SGD', lr=1, momentum=0.9, weight_decay=5e-4, nesterov=False,
        scheduler=None ,scheduler_args=None):
        super(OptimOpts, self).__init__()
        self.method = method
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.nesterov = nesterov
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count