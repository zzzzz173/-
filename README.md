import hashlib
import os
import tarfile
import zipfile
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('..', 'data')):  # @save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):  # @save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():  # @save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB['kaggle_house_train'] = (  # @save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  # @save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
print(train_data.shape)
print(test_data.shape)
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# "Dummy_na=True"将"na"（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values.astype(float), dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values.astype(float), dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.astype(float).reshape(-1, 1), dtype=torch.float32)
loss = nn.MSELoss()
in_features = train_features.shape[1]


def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')

        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')


# 首先定义所有可视化函数
def plot_feature_importance(train_data):
    """绘制特征与房价的相关性热图"""
    try:
        # 只选择数值型列
        numeric_columns = train_data.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix = train_data[numeric_columns].corr()

        # 选择与SalePrice相关性最高的前15个特征
        top_features = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)[:15]

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix.loc[top_features.index, ['SalePrice']],
                    annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('房价与主要特征的相关性分析（前15个特征）')
        plt.tight_layout()
        plt.show()

        # 打印相关性排名
        print("\n特征相关性排名（前15个）：")
        for feature, corr in top_features.items():
            print(f"{feature}: {corr:.3f}")

    except Exception as e:
        print(f"绘制相关性热图时出错: {str(e)}")


def plot_price_distribution(train_data):
    """绘制房价分布图"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_data['SalePrice'], bins=50, kde=True)
    plt.title('房价分布')
    plt.xlabel('价格')
    plt.ylabel('频率')
    plt.show()


def plot_scatter_features(train_data, features, target='SalePrice'):
    """绘制重要特征与房价的散点图"""
    try:
        plt.figure(figsize=(15, 10))
        valid_features = []

        # 只选择存在于数据集中的数值型特征
        for feature in features:
            if feature in train_data.columns and train_data[feature].dtype in ['int64', 'float64']:
                valid_features.append(feature)

        for i, feature in enumerate(valid_features, 1):
            plt.subplot(2, 2, i)
            plt.scatter(train_data[feature], train_data[target], alpha=0.5)
            plt.xlabel(feature)
            plt.ylabel(target)
            plt.title(f'{feature} vs {target}')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"绘制散点图时出错: {str(e)}")


def plot_training_metrics(train_losses, val_losses):
    """绘制训练过程中的损失变化"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练过程损失变化')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.show()


# 然后再定义train_and_pred函数
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    # 在训练前进行数据分析和可视化
    print("开始数据分析和可视化...")

    try:
        # 绘制房价分布
        plot_price_distribution(train_data)

        # 绘制特征相关性
        plot_feature_importance(train_data)

        # 选择数值型特征进行散点图分析
        numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns
        important_features = ['GrLivArea', 'GarageCars', 'TotalBsmtSF', 'OverallQual']
        # 确保特征在数据集中存在
        valid_features = [f for f in important_features if f in numeric_features]
        plot_scatter_features(train_data, valid_features)

    except Exception as e:
        print(f"可视化过程中出错: {str(e)}")
        print("继续执行训练过程...")

    # 继续训练过程
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')

    # 预测和保存结果
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)

    try:
        submission.to_csv('D:\\桌面\\house_price_predictions.csv', index=False)
        print("预测结果已保存到: D:\\桌面\\house_price_predictions.csv")
    except Exception as e:
        print(f"保存预测结果时出错: {str(e)}")


# 最后调用训练和预测函数
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
