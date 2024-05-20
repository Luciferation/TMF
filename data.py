import pickle
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, data_path: str = None, dataset: list = None):
        super().__init__()
        """ 有两种方式加载数据集：
            1) 直接传入数据（外部已加载数据集）；
            2) 从文件读取。
            两种方式加载的数据都必须遵循以下python格式：
                [
                    # dict key 表示视角编号；array(...)代表该视角的样本
                    {0:[array(...), array(...), ...], 1:[array(...), array(...), ...], ...},
                    [int, int, ...]  # 每个样本的标签
                ]
        """
        assert data_path != dataset, 'Please specify at least one parameter of `data_path` and `dataset`'
        if data_path is not None:
            self.x, self.y = pickle.load(open(data_path, 'rb'))
        if dataset is not None:
            self.x, self.y = dataset

    def __getitem__(self, index):
        x = dict()
        for v in self.x.keys():
            x[v] = self.x[v][index]
        return {
            'x': x,
            'y': self.y[index],
            'index': index
        }

    def __len__(self):
        return len(self.y)
    