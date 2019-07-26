import os
import os.path
import numpy as np
from PIL import Image
from operator import itemgetter
import itertools
from multiprocessing import Pool
import functools

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
TENSOR_EXTENSIONS = ['.npy']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_tensor_file(filename):
    return any(filename.endswith(extension) for extension in TENSOR_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_contiguous_targets(targets):
    classes = np.sort(np.unique(targets))
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    new_targets = [class_to_idx[t] for t in targets]
    return new_targets, classes, class_to_idx


def make_dataset_images(dir, class_to_idx):
    images = []
    folders = os.listdir(dir)
    folders.sort()
    for target in folders:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            fnames.sort()
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def make_dataset_tensor(dir, class_to_idx):
    tensors = []
    folders = os.listdir(dir)
    folders.sort()
    for target in folders:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            fnames.sort()
            for fname in fnames:
                if is_tensor_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    tensors.append(item)

    return tensors


def load_reid_dataset(dir, in_memory=False, im_size=None, keep_aspect_ratio=True):
    images = []
    IDs = []
    cams = []
    frames = []
    if os.path.exists(os.path.join(dir, 'train')):
        images_tr, IDs_tr, cams_tr, frames_tr, len_tr = load_reid_dataset(os.path.join(dir, 'train'), in_memory=in_memory, im_size=im_size, keep_aspect_ratio=keep_aspect_ratio)
        images_te, IDs_te, cams_te, frames_te, len_te = load_reid_dataset(os.path.join(dir, 'test'), in_memory=in_memory, im_size=im_size, keep_aspect_ratio=keep_aspect_ratio)

        images_qr = []
        IDs_qr = []
        cams_qr = []
        frames_qr = []
        if os.path.exists(os.path.join(dir, 'query')):
            images_qr, IDs_qr, cams_qr, frames_qr, len_qr = load_reid_dataset(os.path.join(dir, 'query'), in_memory=in_memory, im_size=im_size, keep_aspect_ratio=keep_aspect_ratio)

        images_qr_m = []
        IDs_qr_m = []
        cams_qr_m = []
        frames_qr_m = []
        if os.path.exists(os.path.join(dir, 'query_multi')):
            images_qr_m, IDs_qr_m, cams_qr_m, frames_qr_m, len_qr_m = load_reid_dataset(os.path.join(dir, 'query_multi'), in_memory=in_memory, im_size=im_size, keep_aspect_ratio=keep_aspect_ratio)

        images = list(itertools.chain(images_tr, images_te, images_qr, images_qr_m))
        IDs = list(itertools.chain(IDs_tr, IDs_te, IDs_qr, IDs_qr_m))
        cams = list(itertools.chain(cams_tr, cams_te, cams_qr, cams_qr_m))
        frames = list(itertools.chain(frames_tr, frames_te, frames_qr, frames_qr_m))
    else:
        files = os.listdir(dir)
        files.sort()
        for file in files:
            path = os.path.join(dir, file)
            if not os.path.isfile(path) and is_image_file(path):
                continue
            if file.find('_') == -1:
                IDs.append(int(file[:4]))
                cams.append(int(file[4:8]))
                frames.append(int(file[8:12]))
            else:
                parts = file.split('_')
                IDs.append(int(parts[0])) # ID
                cams.append(int(parts[1][1:])) # Remove 'c'
                frames.append(int(parts[2][1:parts[2].index('.')])) # Remove 'f' and extension

            if in_memory:
                item = load_image(path, im_size, keep_aspect_ratio)
            else:
                item = path
            images.append(item)
    #IDs.sort()
    #cams.sort()
    return images, IDs, cams, frames, range(0,len(IDs))


def load_image(path, sz=None, keepAspectRatio=True):
    im = Image.open(path).convert('RGB')
    if sz is not None:
        aspectRatio = 1
        wsize = sz[0]
        hsize = sz[1]
        if keepAspectRatio:
            if im.width <= im.height:
                aspectRatio = (sz[0] / float(im.width))
            else:
                aspectRatio = (sz[1] / float(im.height))
            wsize = int((float(im.width) * float(aspectRatio)))
            hsize = int((float(im.height) * float(aspectRatio)))

        im = im.resize((wsize, hsize), Image.BICUBIC)
    return im


def load_npy(path, sz=None, keepAspectRatio=True):
    im = np.load(path)
    return im


def pil2np(img, sampleSize=None, keepAspectRatio=True, dtype=np.float32):
    return np.array(copy_image(img, sampleSize, keepAspectRatio), dtype=dtype)


def copy_image(img, sz=None, keepAspectRatio=True):
    im = img.copy()
    if sz is not None:
        aspectRatio = 1
        wsize = sz[0]
        hsize = sz[1]
        if keepAspectRatio:
            if im.width <= im.height:
                aspectRatio = (sz[0] / float(im.width))
            else:
                aspectRatio = (sz[1] / float(im.height))
            wsize = int((float(im.width) * float(aspectRatio)))
            hsize = int((float(im.height) * float(aspectRatio)))

        im = im.resize((wsize, hsize), Image.BICUBIC)
    return im


def compute_pairs_cams(images, classes, cams, indexes, camPair, N):
    pair_images = []
    pair_IDs = []
    pair_cams = []
    pair_indexes = []
    if indexes != [] and N > 0:

        idxs1 = np.array([], np.uint32)
        idxs2 = np.array([], np.uint32)
        #idxs_cam = np.array([], np.uint8)

        images = itemgetter(*indexes)(images)
        classes = itemgetter(*indexes)(classes)
        cams = itemgetter(*indexes)(cams)

        unique_classes = list(set(classes))


        for cl in unique_classes:
            idx1 = [ii for ii in range(0, len(images)) if classes[ii] == cl and cams[ii] == camPair[0]]
            idx2 = [ii for ii in range(0, len(images)) if classes[ii] == cl and cams[ii] == camPair[1]]
            if idx1 != []:
                idx1 = np.random.permutation(idx1)[: np.min([len(idx1), N])]
                idxs1 = np.concatenate((idxs1, idx1))
                #idx_cam = cam * np.ones((len(idx),), np.uint8)
                #idxs_cam = np.concatenate((idxs_cam, idx_cam))
            if idx2 != []:
                idx2 = np.random.permutation(idx2)[: np.min([len(idx2), N])]
                idxs2 = np.concatenate((idxs2, idx2))

        pairs = list(itertools.product(idxs1, idxs2))
        for pair in pairs:
            pair_images.append((images[pair[0]], images[pair[1]]))
            pair_IDs.append((classes[pair[0]], classes[pair[1]]))
            pair_cams.append((cams[pair[0]], cams[pair[1]]))
            pair_indexes.append((indexes[pair[0]], indexes[pair[1]]))

    return pair_images, pair_IDs, pair_cams, pair_indexes


def compute_triplets(images, classes, indexes, cams=None, max_neg=None):
    triplet_images = []
    triplet_classes = []
    triplet_indexes = []
    triplet_cams = []
    if indexes != []:
        with Pool(processes=20) as pool:
            triplets = pool.map(functools.partial(_get_triplet, images=images, classes=classes, indexes=indexes, cams=cams, max_neg=max_neg), indexes)
        for triplet in triplets:
            triplet_images.extend(triplet[0])
            triplet_classes.extend(triplet[1])
            triplet_indexes.extend(triplet[2])
            triplet_cams.extend(triplet[3])
    return triplet_images, triplet_classes, triplet_indexes, triplet_cams


def _get_triplet(anchor, images, classes, indexes, cams, max_neg):
    triplet_images = []
    triplet_classes = []
    triplet_indexes = []
    triplet_cams = []

    pos = [idx for idx in set(indexes) - set([anchor]) if classes[idx] == classes[anchor]]
    neg = list(set(indexes) - set(pos))
    if max_neg is not None:
        neg = np.random.permutation(neg)[: np.min([len(neg), max_neg])]

    triplets = list(itertools.product([anchor], pos, neg))
    for triplet in triplets:
        triplet_images.append(itemgetter(*triplet)(images))
        triplet_classes.append(itemgetter(*triplet)(classes))
        triplet_indexes.append(triplet)
        if cams is not None:
            triplet_cams.append(itemgetter(*triplet)(cams))

    return triplet_images, triplet_classes, triplet_indexes, triplet_cams


