import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import os

# 定义CNN分类器
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        # 两层转换层
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # 池化层
        x = F.max_pool2d(x,2)
        x=self.dropout1(x)
        # 展开以方便连接
        x=torch.flatten(x,1)
        # 全连接层
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dropout2(x)
        x=self.fc2(x)
        # 输出为一个1维向量，表示取各个分类的概率
        output=F.log_softmax(x,dim=1)
        return output


# 定义训练函数
def train(model, device, train_loader,optimizer):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()


# 定义测试函数
def test(model,device,test_loader):
    model.eval()
    #平均损失函数值
    test_loss = 0
    #总正确样本数
    correct = 0
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            output=model(data)
            test_loss+=F.nll_loss(output,target,reduction='sum').item()
            pred=output.argmax(dim=1,keepdim=True)
            correct+=pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('测试结果：平均损失函数值：{:.4f},正确率：{}/{} ({:.0f}%)'.format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))


def main():
    #测试参数
    parser=argparse.ArgumentParser(description='MNIST CNN分类器')
    parser.add_argument('--batch-size',type=int,default=64,metavar='N',help='输入批次大小，默认为64')
    parser.add_argument('--test-batch-size',type=int,default=1000,metavar='N',help='输入测试批次大小，默认为1000')
    parser.add_argument('--epochs',type=int,default=14,metavar='N',help='训练轮数，默认为14')
    parser.add_argument('--lr',type=float,default=1.0,metavar='LR',help='学习率，默认为1.0')
    parser.add_argument('--gamma',type=float,default=0.7,metavar='M',help='学习率衰减率，默认为0.7')
    parser.add_argument('--seed',type=int,default=1,metavar='S',help='随机种子，默认为1')
    parser.add_argument('--save-model',action='store_true',default=False,help='是否保存模型，默认为False')
    parser.add_argument('--cuda',action='store_true',default=False,help='是否使用cuda，默认为False')
    args = parser.parse_args()
    train_kwargs={'batch_size':args.batch_size}
    test_kwargs={'batch_size':args.test_batch_size}
    if torch.cuda.is_available() and args.cuda:
        device=torch.device('cuda')
        num_cpus=os.cpu_count()
        cuda_kwargs={'num_workers':num_cpus,'pin_memory':True,'shuffle':True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        device=torch.device('cpu')
    torch.manual_seed(args.seed)


    #数据集
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    dataset_train=datasets.MNIST('./data',train=True,download=True,transform=transform)
    dataset_test=datasets.MNIST('./data',train=False,transform=transform)
    train_loader=torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader=torch.utils.data.DataLoader(dataset_test,**test_kwargs)

    #模型定义
    model=CNNClassifier().to(device)
    optimizer=optim.Adadelta(model.parameters(),lr=args.lr)
    scheduler=StepLR(optimizer,step_size=1,gamma=args.gamma)

    #训练
    start=time.time()
    for epoch in range(1,args.epochs+1):
        train(model,device,train_loader,optimizer)
        test(model,device,test_loader)
        scheduler.step()
    end=time.time()
    print('训练时间：{:.2f}s'.format(end-start))

    if args.save_model:
        torch.save(model.state_dict(),'cnn_mnist.pt')

if __name__=='__main__':
    main()