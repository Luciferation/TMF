from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import numpy as np


DIR = Path(__file__).resolve().parent  # 当前文件夹，即dataset/


def process_mat(path=DIR / 'handwritten_6views.mat', train=True, out_file_path=None):
    data = scipy.io.loadmat(path)
    mode = 'train' if train else 'test'
    num_views = int((len(data) - 5) / 2)
    x = dict()
    for k in range(num_views):
        view = data[f'x{k+1}_{mode}']
        x[k] = MinMaxScaler([0, 1]).fit_transform(view).astype(np.float32)
    y = data[f'gt_{mode}'].flatten().astype(np.int64)
    if min(y) > 0:
        y -= 1
    if out_file_path is None:
        out_file_path = path.parent / str(path.stem + f'_{mode}.pkl')
    pickle.dump([x, y], open(out_file_path, 'wb'))


if __name__ == '__main__':
    # handwrtten
    process_mat(DIR / 'handwritten_6views.mat', train=True, out_file_path=DIR / 'handwritten_6views_train.pkl')
    process_mat(DIR / 'handwritten_6views.mat', train=False, out_file_path=DIR / 'handwritten_6views_test.pkl')
