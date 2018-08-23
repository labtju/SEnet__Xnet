# SEnet__Xnet
Dian_work
# 经过多天的调试，我发现并比较了以下几种对于SE-ResNet网络防止过拟合并提高准确率的方法
#声明：epoch=30，我保存了训练前24个epoch的网络结构，以节约调试时间，所以图片里只有6个epoch
## 方法一：
### 降低学习率
#### 不过在第几个epoch降低学习率要自己探索
![image](https://github.com/basketballandlearn/SEnet__Xnet/blob/master/learning_rate.png)
![image](https://github.com/basketballandlearn/SEnet__Xnet/blob/master/same.png)
![image](https://github.com/basketballandlearn/SEnet__Xnet/blob/master/down.png)

## 方法二：
### 冻结部份网络
#### 随着训练的深入，冻结部份训练好的网络既提高训练速度，也提高准确率
![image](https://github.com/basketballandlearn/SEnet__Xnet/blob/master/freeze.png)

# Pictures of test acc of net 
## SE_ResNet34
![image](https://github.com/basketballandlearn/SEnet__Xnet/blob/master/down.png)
## ResNet34
![image](https://github.com/basketballandlearn/SEnet__Xnet/blob/master/ResNet.png)

## VGG
![image](https://github.com/basketballandlearn/SEnet__Xnet/blob/master/VGG11.png)
## SE_VGG
![image](https://github.com/basketballandlearn/SEnet__Xnet/blob/master/SE_VGG.png)