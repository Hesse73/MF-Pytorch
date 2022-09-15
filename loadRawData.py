import numpy as np
import pandas as pd
from scipy import sparse as ss
import os

dir_KuaiRand = 'D:/Lab/Rethink/Dataset-KuaiRand-Pure/data/'
dir_YahooR3 = 'D:/Lab/Rethink/Dataset-Yahoo!/R3/'
dir_coat = 'D:/Lab/Rethink/Dataset-coat/'


def load_kuairand():
    print('Loading KuaiRand dataset...')
    standard_file = os.path.join(
        dir_KuaiRand, 'log_standard_4_22_to_5_08_pure.csv')
    rand_file = os.path.join(dir_KuaiRand, 'log_random_4_22_to_5_08_pure.csv')
    #检查重复评价
    std_rating = pd.read_csv(standard_file, header=0)[
        ['user_id', 'video_id', 'play_time_ms', 'duration_ms']].to_numpy()
    rand_rating = pd.read_csv(rand_file, header=0)[
        ['user_id', 'video_id', 'play_time_ms', 'duration_ms']].to_numpy()
    #去掉视频长度为0的数据
    std_rating = std_rating[std_rating[:, 3] != 0]
    rand_rating = rand_rating[rand_rating[:, 3] != 0]

    num_users = max(np.amax(std_rating[:, 0]), np.amax(rand_rating[:, 1]))+1
    num_items = max(np.amax(std_rating[:, 1]), np.amax(rand_rating[:, 1]))+1
    #根据user-itemd对判断重复的评分项
    std_ui_pairs = std_rating[:, :2].copy()
    rand_ui_pairs = rand_rating[:, :2].copy()
    std_mx = ss.csr_matrix((np.ones(len(std_ui_pairs)),
                            (std_ui_pairs[:, 0], std_ui_pairs[:, 1]))).toarray()
    rand_mx = ss.csr_matrix((np.ones(len(rand_ui_pairs)),
                             (rand_ui_pairs[:, 0], rand_ui_pairs[:, 1]))).toarray()
    #补0
    std_mx = np.hstack(
        [std_mx, np.zeros((num_users, rand_mx.shape[1]-std_mx.shape[1]))])
    mask = (std_mx != 0) & (rand_mx != 0)
    #print(mask.sum())   # ---- 输出为136
    users, items = np.where(mask)
    #对于重复项，去除std_rating中的部分
    indexs = []
    for u, i in zip(users, items):
        indexs.append(np.argwhere(
            (std_rating[:, 0] == u) & (std_rating[:, 1] == i))[0][0])
    std_rating = np.delete(std_rating, indexs, 0)
    rating = np.vstack([std_rating, rand_rating]).astype('float64')
    #计算watch_ratio
    rating[:, 2] /= rating[:, 3]
    return np.delete(rating, 3, 1)


def load_yahoo():
    print('Loading yahoo dataset...')
    standard_file = os.path.join(
        dir_YahooR3, 'ydata-ymusic-rating-study-v1_0-train.txt')
    rand_file = os.path.join(
        dir_YahooR3, 'ydata-ymusic-rating-study-v1_0-test.txt')
    std_rating = []
    rand_rating = []
    #yahoo数据集里id从1开始，这里统一减1改为从0开始
    with open(standard_file, 'r') as f:
        for line in f.readlines():
            u, i, r = line.split()
            if u == '5401':
                break
            std_rating.append([int(u)-1, int(i)-1, int(r)])
    std_rating = np.array(std_rating)
    with open(rand_file, 'r') as f:
        for line in f.readlines():
            u, i, r = line.split()
            rand_rating.append([int(u)-1, int(i)-1, int(r)])
    rand_rating = np.array(rand_rating)
    print('standard rating:', len(std_rating),
          'random rating:', len(rand_rating))
    """
    #检查是否存在重复的评分
    std_mx = ss.csr_matrix((std_rating[:,2],(std_rating[:,0],std_rating[:,1]))).toarray()
    rand_mx = ss.csr_matrix((rand_rating[:,2],(rand_rating[:,0],rand_rating[:,1]))).toarray()
    mask = (std_mx!=0)&(rand_mx!=0)
    print(mask.sum()) # ---- 输出为0
    """
    return np.vstack([std_rating, rand_rating])


def load_coat():
    print('Loading coat dataset...')
    standard_file = os.path.join(dir_coat, 'train.ascii')
    random_file = os.path.join(dir_coat, 'test.ascii')
    std_rating = np.loadtxt(standard_file)
    rand_rating = np.loadtxt(random_file)
    print('standard rating:', (std_rating != 0).sum(),
          'random rating:', (rand_rating != 0).sum())
    #print(std_rating.shape,rand_rating.shape)
    """
    #检查是否会存在重复评分
    mark = (std_rating>0)&(rand_rating>0)
    if mark.sum()!=0:
        print('remarked items:',mark.sum())
        print('different items:',(std_rating[mark]!=rand_rating[mark]).sum())
    #输出为：remarked items: 366, different items: 0
    """
    #重复评分的u-i位置是True，其余是False
    mask = (std_rating > 0) & (rand_rating > 0)
    rating = (std_rating + np.where(mask, np.zeros_like(rand_rating),
                                    rand_rating)).astype('int32')
    #print(rating.max(),rating.min())
    #将矩阵转为列表形式
    rate_list = []
    users, items = np.where(rating != 0)
    for x, y in zip(users, items):
        rate_list.append([x, y, rating[x, y]])
    data = np.array(rate_list)
    print('total rating:', len(data))
    return data


if __name__ == '__main__':
    data = load_kuairand()
    print(data)
