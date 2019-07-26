import torch
from torch import nn
from abc import ABCMeta, abstractmethod


class PTModel(nn.Module):

    # Meta class: https://stackoverflow.com/questions/17402622/is-it-possible-in-python-to-declare-that-method-must-be-overridden
    # We want each subclass to have a proper forward_features method implemented!
    __metaclass__ = ABCMeta

    def __init__(self):
        super(PTModel, self).__init__()

    @abstractmethod
    def forward_features(self, x):
        ''' To override '''
        pass

    def get_feature_size(self, shape):
        bs = 1
        x = torch.Tensor(torch.rand(bs, *shape))
        output_feat = self.forward_features(x)
        n_size = output_feat.view(bs, -1).size(1)
        return n_size

    def load_from_checkpoint_path(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.load_from_checkpoint(checkpoint)

    def load_from_checkpoint(self, checkpoint):
        self.load_state_dict(checkpoint['state_dict'])
