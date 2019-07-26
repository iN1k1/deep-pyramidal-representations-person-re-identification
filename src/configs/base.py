import jsonpickle

class BaseConfig(object):
    def __init__(self):
        super(BaseConfig, self).__init__()

    def serialize_json(self):
        return jsonpickle.encode(self)
