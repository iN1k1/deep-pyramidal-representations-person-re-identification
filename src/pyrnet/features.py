from src.features.deep import DeepFeatures
import time


def get_features(net, data_providers, layer_embeddings=['emb\\model\\features\\features_pool'], batch_size=128, workers=4):

    t = time.time()
    print(' ==> Computing Features...', end='')

    # Init feature extractor
    feature_extractor = DeepFeatures('reid_net', net, layer_names=layer_embeddings, linearize=False)

    # Extract deep features
    X_norm = []
    for data_provider in data_providers:
        x_dict = feature_extractor.extract_data_provider(data_provider, batch_size=batch_size, num_workers=workers)

        # Normalize each feature
        for key, x in x_dict.items():
            X_norm.append(feature_extractor.normalize(x, norm_types=['l2']))

    print('done in {}'.format(time.time()-t))

    return X_norm