from abc import ABCMeta, abstractmethod

class abcModel(object):
    __meta_class__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def select_action(self, state, user):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def save(self, episode):
        pass

    @abstractmethod
    def load(self, episode):
        pass

    def buffer_add(self, **kwargs):
        pass
