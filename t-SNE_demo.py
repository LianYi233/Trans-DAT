import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data_embed_collect = []
label_collect = []


def get_fer_data(data_path=''):
    """
	该函数读取上一步保存的两个npy文件，返回data和label数据
    Args:
        data_path:
        label_path:

    Returns:
        data: 样本特征数据，shape=(BS,embed)
        label: 样本标签数据，shape=(BS,)
        n_samples :样本个数
        n_features：样本的特征维度

    """
    data = np.load(data_path, allow_pickle=True)
    label = [0] * 30 + [1] * 11 + [2] * 46
    # label = np.load(label_path)
    # data_path = sorted(os.listdir('/home/wyn/myData/01-hyperspectral/challenge2023/datasets/training/HSI-NIR-FalseColor/basketball1/'))
    # data = cv2.imread('/home/wyn/myData/01-hyperspectral/challenge2023/datasets/training/HSI-NIR-FalseColor/basketball1/'+data_path[0])
    # data = np.ravel(data)
    # data = np.expand_dims(data, 0)
    #
    # for i in range(1, len(data_path)-392):
    #     data1 = cv2.imread('/home/wyn/myData/01-hyperspectral/challenge2023/datasets/training/HSI-NIR-FalseColor/basketball1/'+data_path[i])
    #     data1 = np.ravel(data1)
    #     data1 = np.expand_dims(data1, 0)
    #     data = np.vstack((data, data1))
    #
    # label = np.zeros(data.shape[0], dtype=int)
    return data, label      #, n_samples, n_features


color_map = ['r', 'b', 'g', 'y', 'k', 'm', 'c']  # 7个类，准备7种颜色


def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # data = data[20:40, :]
    # label = label[20:40]
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='s', markersize=7, color=color_map[label[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    # data, label, n_samples, n_features = get_fer_data()
    epoch_num = '0001'
    data, label = get_fer_data('/data_m1/tmp_data/Trans-DAT_'+epoch_num+'.npy')
    # label = label.tolist()
    print('Beginning......')

    # 调用t-SNE对高维的data进行降维，得到的2维的result_2D，shape=(samples,2)
    tsne_2D = TSNE(n_components=2, init='pca', random_state=87)      # init='pca',random
    result_2D = tsne_2D.fit_transform(data)

    print('Finished......')
    fig1 = plot_embedding_2D(result_2D, label, 'epoch '+epoch_num)  # 't-SNE-'+epoch_num
    # fig1.show()
    plt.savefig('/data_m1/tmp_data/t-SNE-pca_'+epoch_num+'.png')
    # plt.pause(50)


if __name__ == '__main__':
    main()
