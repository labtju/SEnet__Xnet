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

class SE(nn.Module):
    def __init__(self, deep):
        super(SE, self).__init__()
        self.se = nn.Sequential(
            nn.Conv2d(deep, deep // 16, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.Conv2d(deep // 16, deep, kernel_size=(1, 1), stride=1),
            nn.Sigmoid()
            )

    def forward(self,x, H): 
        #y = nn.functional.avg_pool2d(x, kernel_size=(H, H)).view(batch_size, -1)
        y = F.avg_pool2d(x, kernel_size=(H, H))
        y = self.se(y)
        return x * y
    
class VGG11(torch.nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2,2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2,2),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2,2),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2,2),
        )
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )
        
        self.conv8 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2,2),
            torch.nn.AvgPool2d(kernel_size=1, stride=1)
        )
        
        self.fc = torch.nn.Sequential(         
            torch.nn.Linear(512,100)    
        )
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        senet = SE(128)
        senet = senet.cuda()
        out = senet(out, 8)
        
        out = self.conv3(out)
        out = self.conv4(out)
        
        senet = SE(256)
        senet = senet.cuda()
        out = senet(out, 4)
        
        out = self.conv5(out)
        out = self.conv6(out)
        
        senet = SE(512)
        senet = senet.cuda()
        out = senet(out, 2)
        
        out = self.conv7(out)
        out = self.conv8(out)
        
        senet = SE(512)
        senet = senet.cuda()
        out = senet(out, 1)
        
        out = out.view(out.size()[0], -1)   
        out = self.fc(out)
        return out

transforms = transforms.Compose([
                                 #transforms.Resize(256),
                                 #transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 ])

def getData():                    
    train_set = tv.datasets.CIFAR100(root='./data_cifar100/', train=True, transform=transforms, download=True)
    train_loader = DataLoader(train_set, batch_size=25, shuffle=True)
  
    test_set = tv.datasets.CIFAR100(root='./data_cifar100/', train=False, transform=transforms, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 25, shuffle = False)
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')               #    这个要按顺序直接定义
    return train_loader, test_loader, classes

def train():                                                              
    net = VGG11()
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
            outputs = net(inputs)                               
            outputs = outputs.cuda()
            loss = ceterion(outputs, labels)                      
            loss.backward()
            optimizer.step()
 
            if step % 600 == 599:
                
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
