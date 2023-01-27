
import torch
import torchvision
import matplotlib.pyplot as plt
import csv
import numpy as np
import random
import struct
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(r'train-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))
    

train_data=[]
test_data=[]
for i in range(len(data)):
    train_data.append(data[i])

    
train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=50,shuffle=True)

epochs=50


learning_rate=1


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.conv1=torch.nn.Conv2d(1,3,3,1)
        self.relu1=torch.nn.ReLU()
        self.lin1=torch.nn.Linear(2028,500)
        self.lin2=torch.nn.Linear(500,150)
        self.kl=0
        self.n=torch.distributions.Normal(0,1)
        self.n.loc=self.n.loc.cuda()
        self.n.scale=self.n.scale.cuda()
        self.conv2=torch.nn.Conv2d(1,3,3,1)
        self.batch1=torch.nn.BatchNorm2d(3)
        self.relu2=torch.nn.ReLU()
        self.lin3=torch.nn.Linear(312,400)
        self.lin4=torch.nn.Linear(400,28*28)
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu1(x)
        a,b,c,d=x.shape
        x=x.reshape(a,b*c*d)
        x=self.lin1(x)
        x=self.lin2(x)
        mu=x
        sigma=torch.exp(x)
        z=mu+sigma*self.n.sample(x.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        z=z.unflatten(1,(1,15,10))
        z=self.conv2(z)
        z=self.relu2(z)
        z=self.batch1(z)
        a,b,c,d=z.shape
        z=z.reshape(a,b*c*d)
        z=self.lin3(z)
        z=self.lin4(z)
        z=z.unflatten(1,(1,28,28))
        return(z)

 
    
model=NeuralNetwork().to(device)
X=next(iter(train_loader)).to(device)
X=X.reshape(50,1,28,28)/255
loss=torch.nn.MSELoss()
opti=torch.optim.SGD(model.parameters(),lr=learning_rate)
for epoch in range(0,epochs):
    for i,images in enumerate(train_loader):
        images=images.reshape(50,1,28,28)/255
        images=images.to(device)
        
        output=model(images)
        l=((((output/28)-(images/28)))**2).sum()/50+model.kl/1000
        
        opti.zero_grad()
        l.backward()
        opti.step()
    if epoch%10==0:
        print(epoch/10)
        print(model.kl)
        print(l)

