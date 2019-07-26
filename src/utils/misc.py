import copy
import pickle


def create_list_of_dictionaries(num_items):
    return [{} for _ in range(num_items)]


def clone(obj):
    return copy.deepcopy(obj)


def save(file_name, **kwargs):
    with open(file_name, 'wb') as fp:
        pickle.dump(len(kwargs)+1, fp)
        keys = list(kwargs.keys())
        pickle.dump(keys, fp, protocol=pickle.HIGHEST_PROTOCOL)
        for k, v in kwargs.items():
            pickle.dump(v, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # h5f = h5py.File(file_name, 'w')
    # grp = h5f.create_group('data')
    # for k,v in kwargs.items():
    #     grp.create_dataset(k,data=v)
    # h5f.close()


def load(file_name):
    data = []
    with open(file_name, 'rb') as f:
        for _ in range(pickle.load(f)):
            data.append(pickle.load(f))
    keys = data[0]
    kw_data = {}
    for k, v in zip(keys, data[1:]):
        kw_data[k] = v
    return kw_data
    # h5f = h5py.File(file_name, 'r')
    # print(h5f.keys())
    # data = h5f['data']
    # h5f.close()
    # return data
