from .base import BaseConfig


class GeneralConfig(BaseConfig):
    def __init__(self, main_path):
        super(GeneralConfig, self).__init__()
        self.main_path = main_path