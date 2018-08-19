import torchvision as tv
import torch.nn as nn
import torch 
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim

MAX_EPOCH = 5

transforms = transforms.Compose([transforms.ToTensor()])



def getData():                    #  数据
    
    train_set = tv.datasets.CIFAR100(root='./data_cifar100/', train=True, transform=transforms, download=True)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
  
    test_set = tv.datasets.CIFAR100(root='./data_cifar100/', train=False, transform=transforms, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = False)
 
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')           
    return train_loader, test_loader, classes

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        
        # 初始化
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, 1000)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet34():
    return ResNet(BasicBlock, [3,6,3,2])

def train():                                                               #    训练
    net = ResNet34()
    net = net.cuda()
    train_dataloader, test_dataloader, classes = getData()                                          
    ceterion = nn.CrossEntropyLoss()                                                               
    optimizer = optim.Adam(net.parameters(), lr=0.001)                  
    print('Start Training')
    for epoch in range(MAX_EPOCH):
        for step, data in enumerate(train_dataloader):           #   一次循环就是一批batch_size
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
                
            optimizer.zero_grad()
            outputs = net(inputs)                               # 开始训练
            outputs = outputs.cuda()
            loss = ceterion(outputs, labels)                     
            loss.backward()
            optimizer.step()
 
            if step % 600 == 599:
                
                print('Epoch: ', epoch, ' |step: ', step, ' |train_loss: ', loss.item())
    print('Finished Training')
    return net, test_dataloader 
 
def test_net(net, test_dataloader): #  开始测试
    correct, total = .0, .0
    for inputs, labels in test_dataloader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        outputs = outputs.cuda()
        _, predicted = torch.max(outputs, 1)                                                # 获取分类结果 question
        total += labels.size(0) 
        correct += (predicted == labels).sum()   
    return float(correct) / total

net, test_dataloader = train()
acc = test_net(net, test_dataloader)
print('Test Accurency :',acc)



