from gardening_tools.modules.transforms.BaseTransform import BaseTransform


class CopyImageToLabel(BaseTransform):
    def __init__(self, copy=False, data_key="image", label_key="label"):
        self.copy = copy
        self.data_key = data_key
        self.label_key = label_key

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __copy__(self, data_dict):
        data_dict[self.label_key] = data_dict[self.data_key].copy()
        return data_dict

    def __call__(self, data_dict):
        if self.copy:
            data_dict = self.__copy__(data_dict)
        return data_dict


class Torch_CopyImageToLabel(BaseTransform):
    def __init__(self, copy=False, data_key="image", label_key="label"):
        self.copy = copy
        self.data_key = data_key
        self.label_key = label_key

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __copy__(self, data_dict):
        data_dict[self.label_key] = data_dict[self.data_key].detach().clone()
        return data_dict

    def __call__(self, data_dict):
        if self.copy:
            data_dict = self.__copy__(data_dict)
        return data_dict
