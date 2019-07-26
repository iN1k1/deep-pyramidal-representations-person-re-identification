from torch.nn import init
from torch import nn
import sklearn.metrics as skmetrics


def init_weights_model_kaiming(model):
    for m in model.modules():
        init_weights_module_kaiming(m)


def init_weights_module_kaiming(module):
    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
    elif isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight.data, a=0, mode='fan_out')
        init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        init.normal_(module.weight.data, 1.0, 0.02)
        init.constant_(module.bias.data, 0)


def init_weights_classifier_model(model):
    for m in model.modules():
        init_weights_classifier_module(m)


def init_weights_classifier_module(module):
    if isinstance(module, nn.Linear):
        init.normal_(module.weight.data, mean=0, std=0.001)
        init.constant_(module.bias.data, 0.0)


def init_weights_normal_model(model):
    for m in model.modules():
        init_weights_normal_module(m)


def init_weights_normal_module(module):
    if isinstance(module, nn.Conv2d):
        init.normal_(module.weight.data, 1.0, 0.02)
        init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        init.normal_(module.weight.data, 1.0, 0.02)
        init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        init.normal_(module.weight.data, 1.0, 0.01)
        init.constant_(module.bias.data, 0)


def init_weights_orthogonal_model(model):
    for m in model.modules():
        init_weights_orthogonal_module(m)


def init_weights_orthogonal_module(module):
    if isinstance(module, nn.Linear):
        init.orthogonal_(module.weight.data)
        init.constant_(module.bias.data, 0.0)

def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k, v in params.items()}
    else:
        return getattr(params.cuda(), dtype)()


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return [r.cpu().item() for r in res], batch_size
