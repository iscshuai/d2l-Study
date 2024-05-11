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

'''
运算符
'''
# 对于任意具有相同形状的张量，常见的标准算术运算符（+、-、*、/和**）都可以被升级为按元素运算
a = torch.tensor([1.0, 2, 4, 8])
b = torch.tensor([2, 2, 2, 2])
print(a + b, a - b, a * b, a / b, a % b, a ** b)

# 张量连结（concatenate）, dim=0沿行方向拼接，dim=1沿列方向拼接
x = torch.arange(12).reshape(3, 4)
y = torch.ones(3, 4)
print(torch.cat((x, y), dim=0), torch.cat((x, y), dim=1))

'''
广播机制：当形状不同，可以通过 广播机制（broadcasting mechanism）来执行按元素操作
1. 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状；
2. 对生成的数组执行按元素操作。
'''
x = torch.arange(3).reshape(3, -1) # 复制一列
y = torch.arange(2).reshape(1, -1) # 复制两行
print(x + y)

'''
节省内存：可以使用切片表示法将操作的结果分配给先前分配的数组，例如Y[:] = <expression>
'''
x = torch.arange(3).reshape(3, 1)
y = torch.ones((3, 1))
Z = torch.zeros_like(x)
print('id(Z):', id(Z))
Z[:] = x + y
print('id(Z):', id(Z)) # 两次地址一致

'''
转换为其他Python对象:
将深度学习框架定义的张量转换为NumPy张量（ndarray）很容易，反之也同样容易。torch张量和numpy数组将共享它们的底层内存，
就地操作更改一个张量也会同时更改另一个张量。
'''
x = torch.zeros((1, 1))
A = x.numpy()
B = torch.tensor(A)
print(type(A), type(B))
print(B, B.item(), float(B), int(B))
