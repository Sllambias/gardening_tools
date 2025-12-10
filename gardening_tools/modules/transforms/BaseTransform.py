import abc


class BaseTransform(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_params(self):
        """This will be called during __call__ and can be used to retrieve arbitrary parameters for the transform applied during __call__."""

    @abc.abstractmethod
    def __call__(self, dict):
        """
        This will be of the form __call__(self, dict):
        which allows calling it as either transform(dict)
        """
