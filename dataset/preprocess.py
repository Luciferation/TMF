from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import numpy as np


DIR = Path(__file__).resolve().parent  # 当前文件夹，即dataset/


def process_toy_example(path=DIR / 'toy_example/ToyExample_2views.pkl', train_rate=0.8):
    x, y = pickle.load(open(path, 'rb'))
    num_train = int(len(y) * 0.8)
    # Train
    train_x = dict()
    for v, k in enumerate(x.keys()):
        train_x[v] = x[k][:num_train]
        train_x[v] = MinMaxScaler([0, 1]).fit_transform(train_x[v]).astype(np.float32)
    train_y = y[:num_train].astype(np.int64)
    # Test
    test_x = dict()
    for v, k in enumerate(x.keys()):
        test_x[v] = x[k][num_train:]
        test_x[v] = MinMaxScaler([0, 1]).fit_transform(test_x[v]).astype(np.float32)
    test_y = y[num_train:].astype(np.int64)
    pickle.dump([train_x, train_y], open(path.parent / 'toy_train.pkl', 'wb'))
    pickle.dump([test_x, test_y], open(path.parent / 'toy_test.pkl', 'wb'))


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
    # toy example
    # process_toy_example(DIR / 'toy_example/ToyExample_2views.pkl')

    # handwrtten
    process_mat(DIR / 'handwritten_6views.mat', train=True, out_file_path=DIR / 'handwritten_6views_train.pkl')
    process_mat(DIR / 'handwritten_6views.mat', train=False, out_file_path=DIR / 'handwritten_6views_test.pkl')
