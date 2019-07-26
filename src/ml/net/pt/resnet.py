from .models import PTModel
from torch import nn
import torchvision.models as pytorchmodels
from .utils import *

def get_resnet_backbone(depth, pretrained=False, remove_avg=True):
    resnet = pytorchmodels.resnet50(pretrained=pretrained)
    if depth == 18:
        resnet = pytorchmodels.resnet18(pretrained=pretrained)
    elif depth == 34:
        resnet = pytorchmodels.resnet34(pretrained=pretrained)
    elif depth == 101:
        resnet = pytorchmodels.resnet101(pretrained=pretrained)
    elif depth == 152:
        resnet = pytorchmodels.resnet152(pretrained=pretrained)

    n_to_del = 1
    if remove_avg == True:
        n_to_del +=1
    modules = list(resnet.named_children())[:-n_to_del]  # delete the avg and the fc layers.
    resnet_backbone = nn.Sequential()
    for key, module in modules:
        resnet_backbone.add_module(key, module)
    return resnet_backbone

class ResNet(PTModel):
    """
    definition
    """

    def __init__(self, depth, num_classes, inshape=(3,224,224), pretrained=False):
        super(ResNet, self).__init__()

        # Get resnet backbone
        self.model = nn.Sequential()
        self.model.features = get_resnet_backbone(depth, pretrained, remove_avg=False)

        # Get feature size
        sz = self.get_feature_size(inshape)

        # Classifier
        self.model.fc = nn.Linear(sz, num_classes)
        #self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        self.init_weights(pretrained)

    def init_weights(self, pretrained):
        nn.init.orthogonal_(self.model.fc.weight)
        # self.fc.bias = torch.zeros(self.fc.bias.size())
        if not pretrained:
            pass

    def forward_features(self, input):
        return self.model.features(input)

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        return self.model.fc(x)

