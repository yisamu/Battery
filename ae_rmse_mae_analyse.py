import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import json
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# 自编码器性能量化指标

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
font1 = {'family':'Times New Roman','size':28}
# p = 'E:/batteries/data_csv/FastCharge_000002_CH42_structure.csv'
# df = pd.read_csv(p)
# plt.plot(df.iloc[1:-1, 0])
# plt.show()

# a = 500
# c = 2000
# b = np.arctan(a)/np.pi
# d = np.arctan(c)/np.pi
# print(b, d)


# df = pd.read_csv('E:/batteries/data_process/labels/points_log_norm_labels.csv')
# print(len(df), df.head(5))
# from sklearn.utils import shuffle
# df = shuffle(df).reset_index(drop=True)
# print(df.head(5))


# import os
# from glob import glob
# path = '文件夹路径'
# l = len(path)+1
# path_list = glob(os.path.join(path, '*'+'.tif'))
# path_list.sort(key=lambda x: int(x[l:l+6]))
# print(path_list)


# from matplotlib import pyplot as plt
# df = pd.read_csv('E:/batteries/data_clean_csv/FastCharge_000006_CH3_structure.csv')
# plt.plot(df.iloc[:,0])
# plt.show()


# path = 'C:/Users/86199/Desktop/tof/data_pearson2.xlsx'
# df = pd.read_excel(path)
# x = np.array(df.iloc[:, 0])
# y = np.array(df.iloc[:, 1])
# def cal_rate(x, y):
#     p1 = x2 = y2 = 0.0
#     x_ = np.mean(x)
#     y_ = np.mean(y)
#     for i in range(len(x)):
#         p1 += (x[i] - x_) * (y[i] - y_)
#         x2 += (x[i] - x_) ** 2
#         y2 += (y[i] - y_) ** 2
#     rate = p1 / ((x2 ** 0.5) * (y2 ** 0.5))
#     return rate
# r = cal_rate(x, y)
# print(r)
import torch
# a = pd.DataFrame([[0, 1, 2, 3, 4, 5, 6, 7, 8], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
# shuffled_indices = np.random.permutation(len(a))
# train_idx = shuffled_indices[:int(0.8 * len(a))]
# val_idx = shuffled_indices[int(0.8 * len(a)):]
# b = a[train_idx]
# print(a)
# print(shuffled_indices)
# print(train_idx)
# print(val_idx)
# print(b)

# df = pd.read_csv('E:/batteries/data_process/labels/single_label150.csv')
# df1 = df.iloc[:, 1]
# mean_val = df1.mean()
# std_val = df1.std()
# max_val = df1.max()
# min_val = df1.min()
# # print(min_val, max_val)
# # print(mean_val, std_val)
# # df1 = (df1-mean_val)/std_val
# df1 = (df1-min_val)/(max_val-min_val)
# df1.name='norm_label'
# # print(df1.head(10), df.head(10))
# df = pd.concat([df, df1], axis=1)
# # print(df.head(5))
# df.to_csv('E:/batteries/data_process/labels/single_label150.csv', index=False)


# plt.boxplot(x,                # 指定要绘制箱线图的数据
#             notch=None,       # 是否是凹口的形式展现箱线图，默认非凹口；
#             sym=None,         # 指定异常点的形状，默认为+号显示；
#             vert=None,        # 是否需要将箱线图垂直摆放，默认垂直摆放；
#             whis=None,        # 指定上下须与上下四分位的距离，默认为1.5倍的四分位差；
#             positions=None,   # 指定箱线图的位置，默认为[0,1,2…]；
#             widths=None,      #指定箱线图的宽度，默认为0.5；
#             patch_artist=None,# 是否填充箱体的颜色；
#             meanline=None,    # 是否用线的形式表示均值，默认用点来表示；
#             showmeans=None,   # 是否显示均值，默认不显示；
#             showcaps=None,    # 是否显示箱线图顶端和末端的两条线，默认显示；
#             showbox=None,     # 是否显示箱线图的箱体，默认显示；
#             showfliers=None,  # 是否显示异常值，默认显示；
#             boxprops=None,    # 设置箱体的属性，如边框色，填充色等；
#             labels=None,      # 为箱线图添加标签，类似于图例的作用；
#             flierprops=None,  # 设置异常值的属性，如异常点的形状、大小、填充色等；
#             medianprops=None, # 设置中位数的属性，如线的类型、粗细等；
#             meanprops=None,   # 设置均值的属性，如点的大小、颜色等；
#             capprops=None,    # 设置箱线图顶端和末端线条的属性，如颜色、粗细等；
#             whiskerprops=None # 设置须的属性，如颜色、粗细、线的类型等；
#             )


mae_arr = np.load('./save_model/mae_arr.npy')
mse_arr = np.load('./save_model/mse_arr.npy')
rmse_arr = np.load('./save_model/rmse_arr.npy')


mae_mean = mae_arr.mean()
mae_min = mae_arr.min()
mae_max = mae_arr.max()

mse_mean = mse_arr.mean()
mse_min = mse_arr.min()
mse_max = mse_arr.max()


rmse_mean = rmse_arr.mean()
rmse_min = rmse_arr.min()
rmse_max = rmse_arr.max()

plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
print('mae_min, mae_mean, mae_max:', mae_min, mae_mean, mae_max)
print('mse_min, mse_mean, mse_max:', mse_min, mse_mean, mse_max)
print('rmse_min, rmse_mean, rmse_max:', rmse_min, rmse_mean, rmse_max)
print(mae_arr.shape, mse_arr.shape, rmse_arr.shape)
plt.figure(figsize=(12, 5))
plt.tick_params(labelsize=16)


plt.subplot(1, 2, 1)
plt.title('平均绝对误差')
plt.boxplot(x=mae_arr, showfliers=False, patch_artist=True,boxprops={'facecolor': 'lightcyan', 'linewidth': 0.8,'edgecolor': 'black'},showmeans=True)


# plt.subplot(1, 2, 2)
# plt.title('均方误差')
# plt.boxplot(x=mse_arr, showfliers=False,patch_artist=True,boxprops={'facecolor': 'lightcyan', 'linewidth': 0.8,'edgecolor': 'black'},showmeans=True)

plt.subplot(1, 2, 2)
plt.title('均方根误差')
plt.boxplot(x=rmse_arr, showfliers=False, patch_artist=True, boxprops={'facecolor': 'lightcyan', 'linewidth': 0.8,'edgecolor': 'black'}, showmeans=True)
plt.show()

# plt.hist(rmse_arr,bins=10000)
# plt.title('RMSE distribution')
# plt.show()

# k = np.random.random(100)
# y_list = []
# for j in range(len(k)):
#     temp_list = []
#     for i in range(2, 102):
#         y = k[j]*i
#         temp_list.append(y)
#     y_list.append(temp_list)
# y_list = np.array(y_list)
# np.save('./labels/simple_x3.npy', y_list)
# np.save('./labels/simple_y3.npy', k*150)
# print(y_list.shape)
# from sklearn.preprocessing import scale
# import random
# x = np.load('./labels/simple_x3.npy')
# y = np.load('./labels/simple_y3.npy')
# for i in range(x.shape[0]):
#     for j in range(x.shape[1]):
#         x[i][j] += 70 * random.gauss(0, 0.12)
# for i in range(len(x)):
#     plt.plot(x[i])
#     plt.xlabel('t')
#     plt.ylabel('x')
#     plt.show()
# from sklearn.preprocessing import scale
# x_ = scale(x, axis=1)
# print(x, x_)
# for i in range(x.shape[0]):
#     plt.subplot(2, 1, 1)
#     plt.plot(x[i])
#     plt.subplot(2, 1, 2)
#     plt.plot(x_[i])
#     plt.show()
# print(x.shape)
# print('*'*100)
# print(y)


# a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
# b = a[1, 0:5:2]
# print(b)
# a = t.permute(1, 0)
#
# b = a - b
# c = b/torch.std(t, dim=1)
# c = c.permute(1, 0)
# print(c)

# fig = plt.figure(figsize=(10, 8))
# df = pd.read_csv('./labels/cycle100-cycle10.csv')
# cycle10 = df.iloc[:, 0]
# cycle100 = df.iloc[:, 1]
# plt.xlabel('时间(t)', size=22)
# plt.ylabel('容量(Ah)', size=22)
# plt.plot(cycle10, label='第10个周期的放电容量曲线')
# plt.plot(cycle100, label='第100个周期的放电容量曲线')
# plt.legend(prop={'size': 22})
# plt.tick_params(labelsize=22)
# plt.savefig("C:/Users/86199/Desktop/电池中文论文攥写/cycle100-cycle10.svg", format="svg")
# plt.show()


