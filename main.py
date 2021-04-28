import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1 , 1 , 1)
        self.conv2 = nn.Conv2d(1 , 1 , 1)
        self.conv3 = nn.Conv2d(1 , 1 , 1)
        self.fc = nn.Linear(512,512)
    def forward(self,x0):
        x0 = F.relu(self.conv1(x0))
        x0 = F.relu(self.conv2(x0))
        x0 = self.conv3(x0)
        x0 = self.fc(x0)
        return x0

net = Net()
for i in range(100):
    running_loss = 0
    x0 = torch.randn(1,1,1,512)
    x1 = torch.rand(1,1,1,512)
    xi = torch.ones(1,1,1,512)

    optimizer = torch.optim.Adam(net.parameters())
    loss_funcation = nn.TripletMarginLoss()
    optimizer.zero_grad(())

    output0 = net(x0)
    output1 = net(x1)
    outputi = net(xi)

    loss = loss_funcation(outputi,output0,output1)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    print(running_loss)

