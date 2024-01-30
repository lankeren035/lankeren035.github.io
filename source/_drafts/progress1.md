## 1. pytorch网课
### Introduction
- format
- calculate
- variable
- activation
### Regression
- 给定x , y（变量）
- 搭建网络
  - init重写（输入，输出，神经元个数）
    - 每一层用Linear
  - forward前向传递（输入数据）
    - x依次经过init里面的层，最后返回
- 定义网络、优化器、误差MSELoss
- 训练（for)
  - x输入网络得到y^
  - 根据y与y^得到误差
  - 优化器梯度为0
  - backward
  - 优化

### Classification
- 定义feature,label
    - x0为y0类点，x1为y1类点
    - 数据形式（[a,b] , 1)
- 搭建网络
    - init
    - forward
- 定义网络、优化器、误差CrossEntropyLoss
- 训练（for)

### quick_setup
- net=torch.nn.Sequential()边定义边实例化

### save and refine
- save 
    - 训练完(for)后
    - torch.save(net1,'net.pkl')#保留整个图
    - torch.save(net1.state_dict(),'net_params.pkl')#保留结点参数
- refine
    - 直接提取：
        - net2=torch.load('net.pkl')
    - 参数提取：
        - 先搭建一模一样的网络n
        - 再n.load_state_dict(torch.load('net_params.pkl'))

### minibize_train(批数据训练)
- 定义batch_size，feature,label
- 定义数据库：torch_dataset=Data.TensorDataset(x,y)
- 定义loader：loader=Data.DataLoader( dataset,batch,shuffle,num_workers)
- 定义网络等
- 训练： for epoch in range(3):  
&emsp;&nbsp;&emsp;&emsp;&emsp;for step,(batch_x,batch_y) in enumerate(loader):

### optimizer
- 用Adam

### cnn
- 下载数据集train_data,定义loader

