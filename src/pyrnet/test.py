import argparse
import os
import torchvision.transforms as transforms
from src.datamanager import *
from src.datamanager import DataProvider
import src.datamanager.utils as datautils
from PIL import Image
from src.configs import *
from src.ml.net import PyNet
from src.results import performance
from src.results.reid import ReIDPerformance
import torchvision.transforms.functional as F
from src.ml.net.pt import factory as model_factory
from operator import itemgetter
from src.visualization import visualizer
import src.pyrnet.model as reid_model
import src.pyrnet.features as features
import src.pyrnet.metric as metric

# Arg parser
parser = argparse.ArgumentParser(description='ReID Net')

parser.add_argument('--dataset', default='Market-1501', type=str, metavar='STR', help='dataset name (default: Market-1501)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 10)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '--p', default=20, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--net', default='densenet', type=str, metavar='STR', help='network model (default: densenet)')
parser.add_argument('--depth', default=201, type=int, metavar='N', help='network model depth (default: 201)')
parser.add_argument('--bottleneck-size', default=512, type=int, metavar='N', help='classifier bottleneck size (default: 512)')
parser.add_argument('--pyr-feature-size', default=256, type=int, metavar='N', help='pyramidal maps (default: 256)')
parser.add_argument('--pyr-feature-size-dynamic', default=True, type=bool, metavar='B', help='pyramidal feature size dependent on detail level (default: True)')
parser.add_argument('--pyr-operator', default='max_pool', type=str, metavar='STR', help='pyramidal operator (default: max_pool)')
parser.add_argument('--pyr-levels', default=-1, type=int, metavar='N', help='pyramidal levels (default: -1 => dynamic)')
parser.add_argument('--metric', default='euclidean', type=str, metavar='STR', help='metric (default: euclidean')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', help='filename of latest checkpoint (default: empty => latest experiment)')
parser.add_argument('--epoch', default=100, type=int, metavar='N', help='evaluation epoch, used only if --checkpoint is not set (default: 100)')
parser.add_argument('--rerank', default=False, type=bool, metavar='B', help='enable re-ranking (default: False)')


def get_args():
    return parser.parse_args()


""" ================================================================================================================
         EVALUATION
    ============================================================================================================ """


def evaluate(args, net=None, dset_train=None, dset_test=None,
             display_ranking_image_index=(0, 2, 10, 40, 60, 100, 120, 140, 160, 180, 200),
             layer_embeddings=('emb\\bottleneck1', 'emb\\bottleneck2', 'emb\\bottleneck3', 'emb\\bottleneck4'),
             sample_size=(384, 192)):

    # Just check the parsed arguments
    print(vars(args))

    """ ----------------------------------------------------------------------------------------------------------------
             DATA
        ------------------------------------------------------------------------------------------------------------ """
    # Imagenet Normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Data transformations
    transformations = DataTransformer([
        transforms.Resize(sample_size, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize
        ])

    transformations_flipped = DataTransformer([
        transforms.Resize(sample_size, interpolation=Image.BICUBIC),
        transforms.Lambda(lambda x: F.hflip(x)),
        transforms.ToTensor(),
        normalize])

    #  Dataset
    if dset_train is None or dset_test is None:
        dset_opts = DatasetConfig(args.dataset, None, (0.5, 0.5), cam_pair=(-1, -1))
        dset = DatasetReID(dset_opts.name, os.path.join('data', dset_opts.name),
                       im_size=dset_opts.imsize, in_memory=False, keep_aspect_ratio=True)
        # Splits
        dset_train, dset_test = dset.split(dset_opts.split, save_load=True, make_each_split_contiguous=True)

    # Data provider
    data_provider = DataProvider(dset_test, loader=datautils.load_image, transform=transformations)
    num_classes = len(dset_train.classes)

    # Data provider flipped
    data_provider_flipped = DataProvider(dset_test, loader=datautils.load_image, transform=transformations_flipped)

    """ ----------------------------------------------------------------------------------------------------------------
            MODEL
        ------------------------------------------------------------------------------------------------------------ """
    if net is None:

        # From which checkpoint do we need to load the model?
        checkpoint = args.checkpoint
        if checkpoint == '':
            folder = os.path.join('data', 'experiments', args.dataset, os.listdir(os.path.join('data', 'experiments', args.dataset))[-1])
            checkpoint = os.path.join(folder, 'checkpoint_epoch-{}.pth.tar'.format(args.epoch))
        folder = os.path.dirname(checkpoint)

        # Get model (load it from checkpoint!)
        model = reid_model.get_model(args.net, args.depth,
                                     data_provider[0][0].size(), num_classes,
                                     bottleneck_size=args.bottleneck_size,
                                     pyr_feature_size=args.pyr_feature_size,
                                     pyr_operator=args.pyr_operator, pyr_feature_size_dynamic=args.pyr_feature_size_dynamic,
                                     checkpoint_path=checkpoint)

        # Make it parallel..
        model = model_factory.make_it_parallel(model, 'multigpu')

        # Net initialization
        net = PyNet()
        net.model = model
        net.exp_folder = folder

        # Move to GPU (if available)
        net.to_gpu()

    """ ----------------------------------------------------------------------------------------------------------------
            FEATURES
        ------------------------------------------------------------------------------------------------------------ """
    X_norm = []
    data_providers = [data_provider, data_provider_flipped]

    # Get features from the data providers
    for ii, dp in enumerate(data_providers):
        X_norm_new = features.get_features(net, [dp], layer_embeddings=layer_embeddings, batch_size=args.batch_size, workers=args.workers)

        # Concat
        X_norm.extend(X_norm_new)

    """ ----------------------------------------------------------------------------------------------------------------
           MATCH
        ------------------------------------------------------------------------------------------------------------ """

    # Match images (re-rank if needed)
    D, D_rerank, probe_info, gallery_info = metric.get_distance(dset_test, X_norm, args.metric, re_rank=args.rerank)

    # Unpack matching info
    probe_idx, probe_id, probe_cam = probe_info
    gallery_idx, gallery_id, gallery_cam = gallery_info

    """ ----------------------------------------------------------------------------------------------------------------
               PERFORMANCE
        ------------------------------------------------------------------------------------------------------------ """
    # CMC
    reid_perf = ReIDPerformance()
    reid_perf.compute(-D, probe_idx, gallery_idx,probe_id, gallery_id, probe_cam=probe_cam, gallery_cam=gallery_cam)
    data_to_print = [reid_perf.cmc[0], reid_perf.cmc[4], reid_perf.cmc[9], reid_perf.cmc[19], reid_perf.cmc[49], reid_perf.nauc, reid_perf.ap.mean()*100]
    res_string = 'CMC [1-5-10-20-50]: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} -- nAUC: {:.2f} -- mAP: {:.2f}'.format(*data_to_print)
    print(res_string)

    # CMC plot
    visualizer.plot_cmc(reid_perf.cmc, legend='Rank-1: {:.2f} - mAP: {:.2f}'.format(reid_perf.cmc[0], reid_perf.ap.mean()*100), title=str(layer_embeddings), render_on_screen=True)

    reid_perf_rerank = ReIDPerformance()
    if D_rerank is not None:
        # CMC with rerank
        reid_perf_rerank.compute(-D_rerank, probe_idx, gallery_idx,probe_id, gallery_id, probe_cam=probe_cam, gallery_cam=gallery_cam)
        data_to_print = [reid_perf_rerank.cmc[0], reid_perf_rerank.cmc[4], reid_perf_rerank.cmc[9], reid_perf_rerank.cmc[19], reid_perf_rerank.cmc[49], reid_perf_rerank.nauc, reid_perf_rerank.ap.mean()*100]
        res_string = 'Re-Rank => CMC [1-5-10-20-50]: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} -- nAUC: {:.2f} -- mAP: {:.2f}'.format(*data_to_print)
        print(res_string)

        img = visualizer.plot_cmc(reid_perf_rerank.cmc, legend='Rank-1: {:.2f} - mAP: {:.2f}'.format(reid_perf_rerank.cmc[0], reid_perf_rerank.ap.mean()*100), title=str(layer_embeddings), render_on_screen=True)

    # Matching images
    dp = DataProvider(dset_test, loader=datautils.load_image)
    matching_images = performance.get_matching_images(dp, dp, reid_perf.matching_indexes, N=15, selected_indexes=display_ranking_image_index)
    matching_ids = itemgetter(*display_ranking_image_index)(reid_perf.matching_ids)
    visualizer.display_ranked_matching_images(matching_images, matching_ids=matching_ids, im_size=(256, 256), render_on_screen=True, true_match_line_width=10)

    return reid_perf, reid_perf_rerank

if __name__ == '__main__':
    args = get_args()
    evaluate(args)
