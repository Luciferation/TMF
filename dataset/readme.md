# 数据集

## 数据处理说明

在`preprocess.py`中编写数据处理代码，其实现的功能应当是：
给定任意数据集的路径，经过处理，将其转化为归一化的多视角数据集。

例如，执行以下命令
```bash
python3 preprocess.py  # 转换数据格式为本实验形式
```
能够获得2个（或3个）文件：`***_train.pkl`,`***_test.pkl`,（`***_valid.pkl`），可供实验使用。这几个文件（`***_train.pkl`）具有相同的数据格式：
```python
[
    # dict key 表示视角编号；array(...)代表该视角的样本列表,形状为(num_samples, ...)
    {0:[array(...), array(...), ...], 1:[array(...), array(...), ...], ...},
    [int, int, ...]  # 每个样本的标签
]
```
实验时，可以使用以下语句读取数据：
```bash
pickle.load(open('./***_train.pkl', 'rb'))
```
