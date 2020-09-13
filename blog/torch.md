# Torch
>More blogs or fun,see [xiaoxin83121](https://xiaoxin83121.github.io/)  


### Tensors

for more information, click https://pytorch-cn.readthedocs.io/zh/latest/package_references/Tensor/  
torch.Tensor是一种只包含单一类型元素的多维矩阵，默认为FloatTensor

>Tensor的建立、初始化
``` python
	x = torch.empty(5,3)
    x = torch.rand(5,3)
    x = torch.zeros(5, 3, dtype=torch.long)
    x = x.new_ones(5, 3)  # generate ones with the arguments of size
    x = torch.randn_like(x, dtype=torch.float) # generate random metrics
    print(x)
    print(x.size())  # size is tuple
    print(tuple({1:2,3:4}))  # dict convert to tuple; result keeps the keys, is (1, 3)
```
>Tensor的operation    

``` python
# add
    y = torch.zeros(5, 3)
    # print(x + y)
    # print(torch.add(x, y))
    result = torch.empty(5, 3)  # generate one before using
    torch.add(x, y, out=result)
    print(result)
    
# resizing
    x = torch.randn(4, 4)
    y = x.view(16)  # 4*4
    z = x.view(-1, 8) # -1 means x dimension is referred by the other dimensions
    
# item
    print(x[0][0].item())  # item must be an element, not an array

# numpy and tensors
    a = torch.ones(5)
    b = a.numpy()
    print(a)  # tensor object
    print(b)  # array

    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)
    print(b)
```

### Network

> torch.nn, 参考https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/  

##### Container
- Model: torch.nn.Module 基类
- add_module(name, module): 增加子模型，通过name指向
```python
class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.add_module("conv", nn.Conv2d(10, 20, 4))
		# self.conv = nn.Module(10, 20, 4) 等价=_=
```
- children(): 全部子模型的迭代器， 集合，重复只返回一个
- eval(): 模型处于验证阶段，将Dropout与Batch-Normalization层参数固定下来，无Dropout与BN层不影响
- module(): 子模块的迭代器，集合，比children()多一个整体结构
- load_state_dict(): 加载parameters
- state_dict(): 返回parameters
- forward(): 定义计算步骤
- parameters(): 参数的迭代器
- zero_grad(): 所有参数的梯度设置为0
- Sequential(args): 时序容器，按照顺序添加，有forward函数
- ModuleList() : 如其名，可理解为类型为module的list，无forward函数，可以用append，expand

##### Convolution
- conv1d/2d(in_channels, out_channels, kernel_size, stride=1, padding=0,...):
	- kernel_size: interger or tuple
	- stride: 步长，默认与卷积核的大小一致
	- padding：填充
	- in_channels, out_channels: **举个例子，in是1，out是6，kernel_size是5，那么卷积核的规模是$(5 \times 5 \times 1) \times 6$，若输入图片为$32 \times 32 \times 1$，经过$2 \times 2$池化后，大小为$14 \times 14 \times 6$. 再接in=6, out=16, kernel_size=5,卷积核的规模是$(5 \times 5 \times 6) \times 16$，卷积后，形成$10 \times 10 \times 16$的结果**。  
**卷积核的规模由卷积核size、输入的通道数、输出的通道数组成**,i.e. $(kernel\_size \times kernel\_size \times in\_channels) \times out\_channels$， 卷积核中除开输出的通道数的部分，才是在图像中进行卷积的”实际卷积核”

##### Others
- Batch_Normalization: 批标准化， BN层的作用是加速训练，对大learn_rate训练效果更好，并可以替代dropout层。基本思想是对所有隐含层输入做白值处理(whiten)，i.e. 正态分布。若是一次导入一批数据，则E与D都可以通过这批数据获得；若单个，则用数据整体E与D表示。  
公式$$y = \frac{x - mean[x]}{ \sqrt{Var[x]} + \epsilon} * \gamma + \beta$$    
其中$\gamma$与$\beta$是可学习的参数，为了使处理后的结果稍远离激活函数如sigmoid的线性区，防止非线性激活函数退化为线性。  
原论文给出的BN的理论依据是reduce internal covariate shift，i.e. 优化了输入层的数据分布远离了线性区的问题，因此，把偏移给拉回来了。  
Another Paper《How Does Batch Normalization Help Optimization?》 实验ICS与optimization间并没有什么关系，并且BN层并不是所有时候都reduce ICS，But **BN is benefitial adn useful**

---

---

---

---

---

---

> 以下来自jupyter notebook， 比较乱

## item() & grad


```python
import torch
x = torch.tensor([[[[1]]]]) # must be a single value without regrad of shape
print(x.item())
```

    1



```python
x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
out = x.pow(2).sum()
print(out)
out.backward()
print(x.grad) # 反向传播的求导值
```

    tensor(4., grad_fn=<SumBackward0>)
    tensor([[ 2., -2.],
            [ 2.,  2.]])


## create tensor


```python
import numpy as np
a = np.array([[1,2], [3,4]])
b = torch.from_numpy(a)
print(b)
# ndarray和numpy共享内存！
b[1][1] = 8
print('after modify tensor type, array is {}'.format(a))

print('zeros---------')
zeros = torch.zeros((2,3))
print(zeros.shape)
zeros_ = torch.zeros_like(torch.transpose(zeros, 0, 1)) # 接收input为tensor
print(zeros_)

# ones和empty同理， full也可以，多一个fill_value参数
print(torch.full((2,2), 1.2))

print('others---------')
print(torch.eye(3))  # 单位矩阵

arange_ = torch.arange(1, 6)
print('{}, type={}'.format(arange_, arange_.dtype))
print(torch.arange(1,6, step=2))  # [ ) 老左闭右开了

linearity = torch.linspace(start=-10, end=10, steps=5)
linearity1 = torch.linspace(start=-10, end=10, steps=1)
print('step=5,res={}; step=1, res={}'.format(linearity, linearity1))

log_0 = torch.logspace(start=-1, end=1, steps=5) # base 默认为10
log_1= torch.logspace(start=-1, end=1, steps=5, base=2)
print('base=10(default),res={}\nbase=2,res={}'.format(log_0, log_1))
```

    tensor([[1, 2],
            [3, 4]], dtype=torch.int32)
    after modify tensor type, array is [[1 2]
     [3 8]]
    zeros---------
    torch.Size([2, 3])
    tensor([[0., 0.],
            [0., 0.],
            [0., 0.]])
    tensor([[1.2000, 1.2000],
            [1.2000, 1.2000]])
    others---------
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    tensor([1, 2, 3, 4, 5]), type=torch.int64
    tensor([1, 3, 5])
    step=5,res=tensor([-10.,  -5.,   0.,   5.,  10.]); step=1, res=tensor([-10.])
    base=10(default),res=tensor([ 0.1000,  0.3162,  1.0000,  3.1623, 10.0000])
    base=2,res=tensor([0.5000, 0.7071, 1.0000, 1.4142, 2.0000])


## Indexing, Slicing, Joining, Mutating Ops


```python
# cat and split/chunk
x = torch.randn(2, 3)  
y = torch.randn(3, 3)  ## size可以不相同，但在合并的dim需要相同
cat_0 = torch.cat((x,y), 0) # axis=0从0轴即0-th dimension开始合并
cat_1 = torch.cat((x,x),1) # 从1-th dimension开始合并
print('axis=0, res={}'.format(cat_0))
print('axis=1, res={}'.format(cat_1))

x = torch.arange(12).reshape(6,2)
print('split(x,2),res={}'.format(torch.split(x, 2))) # split--para2是单个chunk的大小
print('chunk(x,2),res={}'.format(torch.chunk(x, 2)))  # chunk--para2是由多少组chunk
print('split list,res={}'.format(torch.split(x, [1,2,3])))
print('split another dim, res={}'.format(torch.split(x, 1, dim=1)))
```

    axis=0, res=tensor([[ 1.9117,  0.8733, -0.4337],
            [ 1.4848,  1.6400,  0.3719],
            [-0.9608,  0.1104,  1.5698],
            [-0.9131,  0.7957,  1.4129],
            [-0.4189,  0.8566,  0.6866]])
    axis=1, res=tensor([[ 1.9117,  0.8733, -0.4337,  1.9117,  0.8733, -0.4337],
            [ 1.4848,  1.6400,  0.3719,  1.4848,  1.6400,  0.3719]])
    split(x,2),res=(tensor([[0, 1],
            [2, 3]]), tensor([[4, 5],
            [6, 7]]), tensor([[ 8,  9],
            [10, 11]]))
    chunk(x,2),res=(tensor([[0, 1],
            [2, 3],
            [4, 5]]), tensor([[ 6,  7],
            [ 8,  9],
            [10, 11]]))
    split list,res=(tensor([[0, 1]]), tensor([[2, 3],
            [4, 5]]), tensor([[ 6,  7],
            [ 8,  9],
            [10, 11]]))
    split another dim, res=(tensor([[ 0],
            [ 2],
            [ 4],
            [ 6],
            [ 8],
            [10]]), tensor([[ 1],
            [ 3],
            [ 5],
            [ 7],
            [ 9],
            [11]]))



```python
# gather
t = torch.tensor([[1,2],[3,4]])
print('gather dim0,res={}'.format(torch.gather(input=t, dim=0, index=torch.tensor([[0,0], [1,0]]))))
# dim=0 out[i][j]=t[index[i][j]][j]
# out[0][0] = t[index[0][0]][0]=t[0][0]=1
# out[0][1] = t[0][1] = 2
# out[1][0] = t[1][0] = 3
# out[1][1] = t[0][1] = 2
```

    gather dim0,res=tensor([[1, 2],
            [3, 2]])



```python
# index_select
x = torch.randn(3,4)
print('x={}'.format(x))
print('index dim=0,res={}'.format(torch.index_select(input=x, dim=0, index=torch.tensor([0,2]))))
print('index dim=1,res={}'.format(torch.index_select(input=x, dim=1, index=torch.tensor([0,2]))))
```

    x=tensor([[-0.3982,  0.5520,  0.4301,  0.4231],
            [-0.5398, -0.6979,  0.4298, -0.6340],
            [-0.0932, -0.7609,  0.0489, -0.1882]])
    index dim=0,res=tensor([[-0.3982,  0.5520,  0.4301,  0.4231],
            [-0.0932, -0.7609,  0.0489, -0.1882]])
    index dim=1,res=tensor([[-0.3982,  0.4301],
            [-0.5398,  0.4298],
            [-0.0932,  0.0489]])



```python
# reshape
x = torch.arange(24)
x_0 = x.reshape(2, 3, 4)
print('reshape[2,3,4],res={}'.format(x_0))
x_1 = x_0.reshape(12, 2)
print('reshape[12,2],res={}'.format(x_1))
print('reshape[-1,],res={}'.format(x_0.reshape(-1,)))#  展开到-1的dim
print('reshape[-1,],res={}'.format(x_0.reshape(4,-1))) # -1表示自然填充
```

    reshape[2,3,4],res=tensor([[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]],
    
            [[12, 13, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23]]])
    reshape[12,2],res=tensor([[ 0,  1],
            [ 2,  3],
            [ 4,  5],
            [ 6,  7],
            [ 8,  9],
            [10, 11],
            [12, 13],
            [14, 15],
            [16, 17],
            [18, 19],
            [20, 21],
            [22, 23]])
    reshape[-1,],res=tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23])
    reshape[-1,],res=tensor([[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]])



```python
# t 只包括二阶转置，0和1阶不管
# 0D or 1D return the input
x = torch.arange(6).reshape(2,3)
print(torch.t(x))
# transpose
x_ = torch.randn(2,2,3)
print('multi dim x_={}'.format(x_))
print('multi dim x_ transpose,res={}'.format(torch.transpose(x_, 0, 2))) 
## transpose 一次只能对两个dim进行转置
```

    tensor([[0, 3],
            [1, 4],
            [2, 5]])
    multi dim x_=tensor([[[ 0.8073,  1.5364, -1.2041],
             [-0.1548,  0.6042, -0.8837]],
    
            [[-0.2275,  0.6717, -0.3486],
             [ 0.3815,  0.4738,  0.7966]]])
    multi dim x_ transpose,res=tensor([[[ 0.8073, -0.2275],
             [-0.1548,  0.3815]],
    
            [[ 1.5364,  0.6717],
             [ 0.6042,  0.4738]],
    
            [[-1.2041, -0.3486],
             [-0.8837,  0.7966]]])



```python
# stack  要stack的tensor都必须是相同size
a = torch.arange(6).reshape(2,3)
b = torch.arange(6, 12).reshape(2,3)
c = torch.stack((a,b),dim=0)
print('stack res={}'.format(c))  # 注意和新建一个空list，再list.append差不多；
## cat相当于是拼接两个tensor
```

    stack res=tensor([[[ 0,  1,  2],
             [ 3,  4,  5]],
    
            [[ 6,  7,  8],
             [ 9, 10, 11]]])



```python
## torch.squeeze& torch.unsqueeze
# - torch.squeeze() 更紧致一些，把tensor中shape为1的轴去掉；shape不为1的轴，保持原shape
# - torch.unsqueeze() 更松弛一些，接受的第二个参数in range of (0, shape[0])

a = torch.randn(2,3)
b = torch.unsqueeze(a, 0) # a.shape=[1,2,3]
c = torch.unsqueeze(a, 1) # a.shape=[2,1,3]
d = torch.unsqueeze(a, 2) # a.shape=[2,3,1]

print(a)
print(b)
print(c)
print(d)

print("---------")
print(torch.squeeze(b, 0)) # shape=[2,3]
print(torch.squeeze(c, 0)) # shape=[2,1,3]保持不变
```

    tensor([[ 0.1274,  0.5732, -1.5700],
            [-0.2610, -0.4772,  2.5485]])
    tensor([[[ 0.1274,  0.5732, -1.5700],
             [-0.2610, -0.4772,  2.5485]]])
    tensor([[[ 0.1274,  0.5732, -1.5700]],
    
            [[-0.2610, -0.4772,  2.5485]]])
    tensor([[[ 0.1274],
             [ 0.5732],
             [-1.5700]],
    
            [[-0.2610],
             [-0.4772],
             [ 2.5485]]])
    ---------
    tensor([[ 0.1274,  0.5732, -1.5700],
            [-0.2610, -0.4772,  2.5485]])
    tensor([[[ 0.1274,  0.5732, -1.5700]],
    
            [[-0.2610, -0.4772,  2.5485]]])



```python
# where
## paras(condition, tensor1, tensor2) 
## bitwise, if condition True, choose tensor1; condition False, choose tensor2
x = torch.randn(3, 2)
y = torch.randn(3, 2)  ## 两个tensor size 必须要一致
print('x={}'.format(x))
print('y={}'.format(y))
print('torch.where, res={}'.format(torch.where(x>y, x, y)))

## function: bitwise比较
```

    x=tensor([[ 0.7458,  0.3450],
            [-0.2103,  1.1277],
            [ 0.8660, -0.4983]])
    y=tensor([[-1.6863,  0.4662],
            [-0.0625,  1.2547],
            [-0.3380, -0.2079]])
    torch.where, res=tensor([[ 0.7458,  0.4662],
            [-0.0625,  1.2547],
            [ 0.8660, -0.2079]])


## seed and random


```python
import torch
# seed
## seed用于控制不同情况下生成的随机数能够保持一致
torch.manual_seed(2020)
## Generator 控制seed和random数的生成

# random for distribution
## bernoulli
a = torch.empty(3,3).uniform_(0,1)
print('bernoulli, res={}'.format(torch.bernoulli(a)))  # 按照Bernoulli分布

## poisson
rates = torch.rand(4,4) * 5 # 新用法
print('poisson,res={}'.format(torch.poisson(rates)))

## normal distribution
## paras: mean=(float or tensor), std=(float or tensor)
## if mean==float&& std==float, para3：size tuple
print('paras:tensor,res={}'.format(torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))))
print('paras:float,res={}'.format(torch.normal(mean=2, std=3, size=(2,3))))
```

    bernoulli, res=tensor([[0., 0., 0.],
            [0., 0., 0.],
            [1., 1., 1.]])
    poisson,res=tensor([[1., 3., 0., 2.],
            [3., 4., 3., 6.],
            [0., 3., 2., 1.],
            [4., 6., 2., 1.]])
    paras:tensor,res=tensor([1.6121, 2.0414, 3.5005, 3.1714, 4.1113, 6.1809, 7.0191, 7.8895, 8.8858,
            9.9199])
    paras:float,res=tensor([[6.5937, 3.5000, 2.3347],
            [3.5410, 2.8064, 1.8501]])



```python
# rand class
## rand [0,1)之间的随机数
## randn 满足高斯分布
## randint paras:low, high, size，[low, high)之间的随机数
## randperm [0,n）的n个数（一定是n个）

print('rand, res={}'.format(torch.rand(2,3)))
print('randn, res={}'.format(torch.randn(2,3)))
print('randint, res={}'.format(torch.randint(low=1, high=10, size=(2,3))))
print('randperm, res={}'.format(torch.randperm(5)))

# rand(x)_like函数和zeros_like相仿
```

    rand, res=tensor([[0.6329, 0.9440, 0.7993],
            [0.1589, 0.7711, 0.9412]])
    randn, res=tensor([[-1.6119,  0.1906, -1.3852],
            [-0.4564,  1.5425, -1.3070]])
    randint, res=tensor([[3, 1, 4],
            [5, 8, 2]])
    randperm, res=tensor([4, 2, 3, 1, 0])


## threads
- get_num_threads: number of thread used for parallelizing in CPU
- set_num_threads:

## local gradient computation


```python
x = torch.zeros(1, requires_grad=True) ## 有无grad的设置在于x引入了grad
with torch.no_grad(): # 接下来的计算x不使用grad
    y = x * 2
    with torch.enable_grad(): # 接下来使用grad
        z = x * 2
print('y with no_grad, res={}'.format(y.requires_grad))
print('z with enable_grad,res={}'.format(z.requires_grad))

is_train = False
torch.set_grad_enabled(is_train)
z = x * 2 ## 如果该行注释，res=True
print('z with set fresh,res={}'.format(z.requires_grad))
torch.set_grad_enabled(not is_train)
y = x * 2  ## 注释，res=False
print('y with set fresh,res={}'.format(y.requires_grad))
```

    y with no_grad, res=False
    z with enable_grad,res=True
    z with set fresh,res=False
    y with set fresh,res=True



```python

```