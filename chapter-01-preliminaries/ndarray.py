import torch

# 使用arange创建一个行向量x
x = torch.arange(12)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# 张量的shape属性来访问张量的形状(沿每个轴的⻓度）
print(x.shape)
# torch.Size([12])

# 张量中元素的总数
print(x.numel())

# 改变一个张量的形状而不改变元素数量和元素值
print(x.reshape(3, 4))

# 全0 初始化矩阵
print(torch.zeros((2, 3, 4)))

# 全1 初始化矩阵
print(torch.ones((2, 3, 4)))

# 随机初始化 均值为0 标准差为1 的标准高斯（正态）分布中随机采样
print(torch.randn(3, 4))
