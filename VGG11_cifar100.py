import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torchvision as tv
import torch.nn as nn
import torch 
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim

MAX_EPOCH = 5
CLASS_NUM = 100
use_cuda = torch.cuda.is_available()

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, CLASS_NUM)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

transforms = transforms.Compose([transforms.ToTensor()
                                 ])

def getData():                  
    
    train_set = tv.datasets.CIFAR100(root='./data_cifar100/', train=True, transform=transforms, download=True)
    train_loader = DataLoader(train_set, batch_size=25, shuffle=True)
  
    test_set = tv.datasets.CIFAR100(root='./data_cifar100/', train=False, transform=transforms, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 25, shuffle = False)
 
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')               #    这个要按顺序直接定义

    return train_loader, test_loader, classes
 
def train():                                                            
    net = VGG('VGG11')
    net = net.cuda()
    train_dataloader, test_dataloader, classes = getData()                                          
    ceterion = nn.CrossEntropyLoss()                                                               
    optimizer = optim.Adam(net.parameters(), lr=0.001)                  
    print('Start Training')
    for epoch in range(MAX_EPOCH):
        for step, data in enumerate(train_dataloader):
            
            inputs, labels = data
            
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = outputs.cuda()
            loss = ceterion(outputs, labels)                      
            loss.backward()
            optimizer.step()
 
            if step % 300 == 299:
                print('Epoch: ', epoch, ' |step: ', step, ' |train_loss: ', loss.item())
    print('Finished Training')
    return net, test_dataloader 
 
def test_net(net, test_dataloader): 
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

