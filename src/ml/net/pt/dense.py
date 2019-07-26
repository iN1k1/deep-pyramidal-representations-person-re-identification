from .models import PTModel
import torchvision.models as pytorchmodels
from .utils import *

def get_densenet_backbone(depth, pretrained=False):
    dense_net = pytorchmodels.densenet121(pretrained=pretrained)
    if depth == 161:
        dense_net = pytorchmodels.densenet161(pretrained=pretrained)
    elif depth == 169:
        dense_net = pytorchmodels.densenet169(pretrained=pretrained)
    elif depth == 201:
        dense_net = pytorchmodels.densenet201(pretrained=pretrained)

    modules = list(dense_net.named_children())[:-1]  # delete the last fc layer.
    dense_net_backbone = nn.Sequential()
    for key, module in modules:
        dense_net_backbone.add_module(key, module)
    return dense_net_backbone

class DenseNet(PTModel):
    """
    definition
    """

    def __init__(self, depth, num_classes, pretrained=False):
        super(DenseNet, self).__init__()

        # Get densenet backbone
        self.model = get_densenet_backbone(depth, pretrained)

        # Add relu and av pooling layers
        self.model.features.add_module('features_relu', nn.ReLU(inplace=True))
        self.model.features.add_module('features_avg_pool', nn.AvgPool2d(kernel_size=7))

        # Final FC layer
        self.model.add_module('fc', nn.Linear(2048, num_classes))

        self.init_weights(pretrained)

    def init_weights(self, pretrained):
        nn.init.orthogonal_(self.model.fc.weight)
        # self.fc.bias = torch.zeros(self.fc.bias.size())
        if not pretrained:
            pass

    def forward_features(self, input):
        return self.model.features(input)

    def forward(self, x):
        x = self.forward_features(x).view(x.size(0), -1)
        x = self.model.fc(x)
        return x

