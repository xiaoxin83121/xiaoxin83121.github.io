# Torch

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
	- in_channels, out_channels: 举个例子，in是1，out是6，kernel_size是5，那么卷积核的规模是$(5 \times 5 \times 1) \times 6$，若输入图片为$32 \times 32 \times 1$，经过$2 \times 2$池化后，大小为$14 \times 14 \times 6$. 再接in=6, out=16, kernel_size=5,卷积核的规模是$(5 \times 5 \times 6) \times 16$，卷积后，形成$10 \times 10 \times 16$的结果。  
**卷积核的规模由卷积核size、输入的通道数、输出的通道数组成**,i.e. $(kernel\_size \times kernel\_size \times in\_channels) \times out\_channels$， 卷积核中除开输出的通道数的部分，才是在图像中进行卷积的”实际卷积核”

##### Others
- Batch_Normalization: 批标准化， BN层的作用是加速训练，对大learn_rate训练效果更好，并可以替代dropout层。基本思想是对所有隐含层输入做白值处理(whiten)，i.e. 正态分布。若是一次导入一批数据，则E与D都可以通过这批数据获得；若单个，则用数据整体E与D表示。  
公式$$y = \frac{x - mean[x]}{ \sqrt{Var[x]} + \epsilon} * \gamma + \beta$$    
其中$\gamma$与$\beta$是可学习的参数，为了使处理后的结果稍远离激活函数如sigmoid的线性区，防止非线性激活函数退化为线性。  
原论文给出的BN的理论依据是reduce internal covariate shift，i.e. 优化了输入层的数据分布远离了线性区的问题，因此，把偏移给拉回来了。  
Another Paper《How Does Batch Normalization Help Optimization?》 实验ICS与optimization间并没有什么关系，并且BN层并不是所有时候都reduce ICS，But **BN is benefitial adn useful**

