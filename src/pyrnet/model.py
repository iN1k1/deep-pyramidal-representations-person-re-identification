import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from src.ml.net.pt import *
import numpy as np
from src.features import DeepFeatures

class DenseNetReID(PTModel):
    def __init__(self, net_name, depth, num_classes, inshape=(3,224,224), pretrained=False, bottleneck_size=512,
                 pyr_feature_size=256, pyr_pooling_operator='max_pool', pyr_levels=-1, pyr_feature_size_dynamic=True):
        super(DenseNetReID, self).__init__()

        # Get densenet backbone
        if net_name == 'densenet':
            self.base = get_densenet_backbone(depth, pretrained)

            # Block ranges
            self.block_ranges = [slice(0,4), slice(4,5), slice(5,7), slice(7,9), slice(9,11), slice(11,12)]

        else:
            self.base = nn.Sequential()
            self.base.add_module('features', get_resnet_backbone(depth, pretrained))

            # Block ranges
            self.block_ranges = [slice(0, 4), slice(4, 5), slice(5, 6), slice(6, 7), slice(7, 8)]

        # Low level feature sizes
        bs = 2
        x_in = torch.Tensor(torch.rand(bs, *inshape))
        x, xd1, xd2, xd3, xd4 = self.forward_low_level_features(x_in)

        pyr_sizes = [pyr_feature_size] * 4
        if pyr_feature_size_dynamic is True:
            pyr_sizes[0] = int(pyr_feature_size/8)
            pyr_sizes[1] = int(pyr_feature_size/4)
            pyr_sizes[2] = int(pyr_feature_size/2)
            pyr_sizes[3] = int(pyr_feature_size)

        pyr_level = [7,6,5,4]
        if pyr_levels != -1:
            pyr_level = [pyr_levels] * 4

        self.pyr1 = self._get_pyramid_block(xd1.size(1), pyr_level[0], pyr_sizes[0], pyr_pooling_operator)
        self.pyr2 = self._get_pyramid_block(xd2.size(1), pyr_level[1], pyr_sizes[1], pyr_pooling_operator)
        self.pyr3 = self._get_pyramid_block(xd3.size(1), pyr_level[2], pyr_sizes[2], pyr_pooling_operator)
        self.pyr4 = self._get_pyramid_block(xd4.size(1), pyr_level[3], pyr_sizes[3], pyr_pooling_operator)


        # Pyramid features
        pyr_feats = self.forward_features(x_in)#.view(bs,-1).size(1)

        # Bottlenecks
        self.bottleneck1 = self._get_bottleneck_block(pyr_feats[0].size(1), bottleneck_size)
        self.bottleneck2 = self._get_bottleneck_block(pyr_feats[1].size(1), bottleneck_size)
        self.bottleneck3 = self._get_bottleneck_block(pyr_feats[2].size(1), bottleneck_size)
        self.bottleneck4 = self._get_bottleneck_block(pyr_feats[3].size(1), bottleneck_size)

        # Bottleneck features
        feats = self.forward_bottlenecks(pyr_feats)

        # Classifier
        classifiers = []
        for x in feats:
            classifiers.append(self._get_classifier_block(x.size(1), num_classes))
        self.classifier = nn.ModuleList(classifiers)

    def _get_pyramid_block(self, insize, n_pyr, n_feat_maps, pyr_pooling_operator):
        pyramid_block = nn.Sequential(
            nn.Conv2d(insize, n_feat_maps, 1),
            nn.BatchNorm2d(n_feat_maps),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.AdaptiveMaxPool2d((1, 1))
            SpatialPyramidPooling(n_pyr, pyr_pooling_operator, stripe=True)
        )
        pyramid_block.apply(init_weights_model_kaiming)
        return pyramid_block

    def _get_bottleneck_block(self, insize, bottleneck_size):
        bottleneck = nn.Sequential(
            nn.Linear(insize, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5)
            )
        bottleneck.apply(init_weights_model_kaiming)
        return bottleneck

    def _get_classifier_block(self, bottleneck_size, num_classes):
        classifier = nn.Sequential(
            nn.Linear(bottleneck_size, num_classes)
        )
        classifier.apply(init_weights_classifier_model)
        return classifier

    def forward_low_level_features(self, x):
        x = self.base.features[self.block_ranges[0]](x)
        x_lev1 = self.base.features[self.block_ranges[1]](x) # First dense block
        x_lev2 = self.base.features[self.block_ranges[2]](x_lev1) # second dense block
        x_lev3 = self.base.features[self.block_ranges[3]](x_lev2)  # third dense block
        x_lev4 = self.base.features[self.block_ranges[4]](x_lev3)  # fourth dense block
        if len(self.block_ranges) > 5:
            x = self.base.features[self.block_ranges[5]](x_lev4) # batch norm
        else:
            x = x_lev4

        return x, x_lev1, x_lev2, x_lev3, x_lev4

    def forward_pyramids(self, x, x_lev1, x_lev2, x_lev3, x_lev4, normalize=False):
        x_pyr1 = self.pyr1(x_lev1)
        x_pyr2 = self.pyr2(x_lev2)
        x_pyr3 = self.pyr3(x_lev3)
        x_pyr4 = self.pyr4(x_lev4)
        x = [x_pyr1, x_pyr2, x_pyr3, x_pyr4]
        if normalize:
            x = [F.normalize(xpyr, p=2, dim=1) for xpyr in x]
        return x

    def forward_bottlenecks(self, x):
        xb1 = self.bottleneck1(x[0])
        xb2 = self.bottleneck2(x[1])
        xb3 = self.bottleneck3(x[2])
        xb4 = self.bottleneck4(x[3])
        return (xb1, xb2, xb3, xb4)

    def forward_features(self, x):
        x, xd1, xd2, xd3, xd4 = self.forward_low_level_features(x)
        x = self.forward_pyramids(x, xd1, xd2, xd3, xd4)
        return x

    def forward_classifier(self, x):
        return [classifier(x_pyr) for classifier, x_pyr, in zip(self.classifier,x)]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_bottlenecks(x)
        x = self.forward_classifier(x)
        return x


class TripleNet(PTModel):
    def __init__(self, emb):
        super(TripleNet, self).__init__()
        self.emb = emb

    def forward(self, input, **pars):
        if isinstance(input, list):
            # Get embedding (get transformed input first..)
            feat = [self.emb.forward_bottlenecks(self.emb.forward_features(i)) for i in input]

            # Get class info
            cl = [self.emb.forward_classifier(f) for f in feat]

            # Needed to handle items of different size for which we need only the class info
            if len(pars) > 0 and pars is not None and pars['extra_pars'][0] is True:
                return torch.cat(cl)

            return (cl, feat)

        return self.emb(input)


class TripletReIDLoss(nn.Module):
    def __init__(self, triplet_margin=1.0, norm=2, alpha=0.95, use_triplet=True):
        super(TripletReIDLoss, self).__init__()
        n_losses = 4
        self.classification_losses = nn.ModuleList([nn.CrossEntropyLoss() for _ in range(n_losses)])
        self.triplet_losses = nn.ModuleList([nn.TripletMarginLoss(margin=triplet_margin, p=norm, swap=False) for _ in range(n_losses)])
        self.alpha = alpha
        self.use_triplet = use_triplet

    def forward(self, input, target, **pars):

        # Class / features
        if self.use_triplet:
            cl, feat = input

            # Classification loss for each sample
            c_anchor = self.forward_class_loss(cl[0], target[0])
            c_pos = self.forward_class_loss(cl[1], target[1])
            c_neg = self.forward_class_loss(cl[2], target[2])
            b = 1
            d = 3
            c = ((c_anchor + c_pos)*b + c_neg) / d

            # Triplet loss
            t = self.forward_triplet_loss(feat)

            # weighted sum of the two losses..
            return ((1-self.alpha) * c) + (self.alpha * t)
        else:
            return self.forward_class_loss(input, target)

    def forward_class_loss(self, input, target):
        return torch.sum(torch.stack([loss(x, target) for x, loss in zip(input, self.classification_losses)]))

    def forward_triplet_loss(self, input):
        triplet_loss = torch.Tensor([0]).to("cuda")
        for ii, loss in enumerate(self.triplet_losses):
            triplet_loss += loss(input[0][ii], input[1][ii], input[2][ii])
        return triplet_loss


def accuracy(output, target, topk):
    if (isinstance(output, list) or isinstance(output, tuple)) and (isinstance(target, list) or isinstance(target, tuple)):
        # Only classif info
        output = output[0]
        if len(output) > 0:
            if isinstance(output[0], list) or isinstance(output[0], tuple):
                anchor = np.zeros((1, len(topk)))
                pos = np.zeros((1, len(topk)))
                neg = np.zeros((1, len(topk)))
                for ii in range(len(output[0])):
                    a, bs = utils.accuracy(output[0][ii], target[0], topk)
                    p, bs = utils.accuracy(output[1][ii], target[1], topk)
                    n, bs = utils.accuracy(output[2][ii], target[2], topk)

                    anchor += np.array(a)
                    pos += np.array(p)
                    neg += np.array(n)

                prec = [(a + p + n) / 3 / len(output[0]) for a, p, n in zip(anchor, pos, neg)]
                prec = prec[0].tolist()

            else:
                anchor, bs = utils.accuracy(output[0], target[0], topk)
                if len(output) > 1:
                    pos, bs = utils.accuracy(output[1], target[1], topk)
                    neg, bs = utils.accuracy(output[2], target[2], topk)

                    prec = [(a+p+n)/3 for a,p,n in zip(anchor, pos, neg)]
                else:
                    prec = anchor
        else:
            prec = [0] * len(topk)
            bs = target[0].size(0)
        #bs *= 3

    # Output is list, target not
    elif isinstance(output, list) or isinstance(output, tuple):
        prec = np.array([0] * len(topk)).astype(np.float)
        for o in output:
            prec_o, bs = utils.accuracy(o, target, topk)
            prec_o = np.array(prec_o, dtype=np.float)
            prec += prec_o
        prec = (prec / len(output)).tolist()

    # Output and target are tensors..
    else:
        prec, bs = utils.accuracy(output, target, topk)
    return prec, bs


def get_model(model_name, depth, in_shape, num_classes, bottleneck_size, pyr_feature_size, pyr_operator,
              pyr_levels=-1, pyr_feature_size_dynamic=True, checkpoint_path=''):

    emb = DenseNetReID(model_name, depth=depth, num_classes=num_classes, inshape=in_shape, pretrained=True,
                       bottleneck_size=bottleneck_size,
                       pyr_feature_size=pyr_feature_size, pyr_feature_size_dynamic=pyr_feature_size_dynamic,
                       pyr_pooling_operator=pyr_operator, pyr_levels=pyr_levels)
    model = TripleNet(emb)

    if checkpoint_path != '':
        print(' ==> Loading from checkpoint: {}'.format(checkpoint_path))
        model.load_from_checkpoint_path(checkpoint_path)

    return model


def get_loss(use_triplet=True, triplet_margin=0.1, alpha=0.5):
    return TripletReIDLoss(triplet_margin=triplet_margin, alpha=alpha, use_triplet=use_triplet)
