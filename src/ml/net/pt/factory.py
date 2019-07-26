import torch


def make_it_parallel(model, parallel_type=''):
    if parallel_type == 'multigpu':
        print(' ==> Parallelizing model')
        model = torch.nn.DataParallel(model)
    return model
