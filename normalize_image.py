import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#Load data
train_dataset = datasets.CIFAR10(root='dataset/', train=True,
                                 transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

def get_mean_std(loader):
    #VAR[X] = E[X^2] - E[X]^2
    channel_sum, channel_squared_sum, num_batch = 0, 0, 0
    for data, _ in loader:
        channel_sum += torch.mean(data, dim=[0,2,3])
        channel_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batch +=1
        
    mean = channel_sum/num_batch
    std = (channel_squared_sum/num_batch - mean**2)**0.5
    
    return mean,std

mean,std = get_mean_std(train_loader)
print(mean)
print(std)