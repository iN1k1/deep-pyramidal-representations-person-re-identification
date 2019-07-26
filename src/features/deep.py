from .features import FeatureExtractor
from ..ml.net import PyNet
import os
import torch


class DeepFeatures(FeatureExtractor):

    def __init__(self, name, model, layer_names, sample_size=(), mu=(), std=(), use_gpu=True, linearize=True, normalization=()):
        super(DeepFeatures, self).__init__(dense=None, normalization=normalization)

        if isinstance(model, PyNet):
            self.net = model
        else:
            self.net = PyNet()
            self.net.model = model
            self.net.name = name
            self.net.use_gpu = use_gpu
            self.net.to_gpu()  # Within this we are checking the net.use_gpu flag..
        self.sample_size = sample_size
        self.mu = mu
        self.std = std
        self.linearize = linearize
        self.net.set_extract_features_layers(layer_names)
        self.type = 'deep'

    def extract(self, numpy_image, apply_normalizations=False):
        z = self.net.get_preprocessed_tensor(numpy_image, self.mu, self.std, self.sample_size)
        return (self.net.extract_features(z, linearize=self.linearize), self.type)

    def extract_data_provider(self, data_provider, batch_size=1, num_workers=1):

        # Old data provider
        old_val_data_loader = self.net.val_loader

        # Set data provider
        self.net.set_data_providers(train_data_provider=None, val_data_provider=data_provider, train_batch_size=0,
                                    val_batch_size=batch_size, num_workers=num_workers)

        # Force re-setting embedding layers again.. this is to prevent getting the embedded data more than once, if this
        # function is called more than once..
        # By calling this we clear any previous store data
        self.net.set_extract_features_layers(self.net.embedding_layers)

        # Unlink hooks for loss and post forward computation
        def pass_fun(*args, **kwargs):
            return self.net.to_var(torch.FloatTensor(1).zero_())
        def pass_acc_fun(*args, **kwargs):
            return (0,0), 1
        old_loss = self.net.hooks['loss']
        old_post_forward = self.net.hooks['on_post_forward']
        old_accuracy = self.net.hooks['accuracy']
        self.net.hooks['loss'] = pass_fun
        self.net.hooks['on_post_forward'] = [pass_fun]
        self.net.hooks['accuracy'] = pass_acc_fun

        # Run validation... through this, each layer hook will be called and fill the embedding vectors
        old_verbose = self.net.verbose
        self.net.verbose = False
        self.net.validate(accumulate_outputs=False)
        self.net.verbose = old_verbose

        # Get the embedded vectors
        np_mat = self.net.get_embeddings(linearize=self.linearize)

        # Restore back hooks
        self.net.hooks['loss'] = old_loss
        self.net.hooks['on_post_forward'] = old_post_forward
        self.net.hooks['accuracy'] = old_accuracy
        # self.net.hooks['on_get_extra_model_pars'] = old_model_pars_f

        # Set back any previous validation data provider
        self.net.val_loader = old_val_data_loader

        # Clear listeners
        self.net.clear_embedding_listeners()

        # Return back
        return np_mat

    def get_file_path(self, main_path):
        return os.path.join(main_path, '{}-{}_{}-{}.npy'.format(self.type, self.net.name, self.sample_size[0], self.sample_size[1]))


