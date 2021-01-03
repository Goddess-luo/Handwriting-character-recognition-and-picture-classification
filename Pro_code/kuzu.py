# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        # from torch.Size, get the image size is 28 * 28 = 784
        self.Linear_function = nn.Linear(784, 10, bias = False)

    def forward(self, x):
        # get torch.Size([64, 1, 28, 28]) to [64, 1, 28, 28] to view
        view_step = x.view(x.size()[0], -1)
        result = self.Linear_function(view_step)
        return F.log_softmax(result, dim = 1)    # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.tanh = nn.Tanh()
        self.linear_func_1 = nn.Linear(784, 390, bias = False)
        self.linear_func_2 = nn.Linear(390, 10, bias = False)

    def forward(self, x):
        view_step = x.view(x.size()[0], -1)
        step_1 = self.tanh(self.linear_func_1(view_step))
        step_2 = self.linear_func_2(self.tanh(step_1))
        return F.log_softmax(step_2, dim = 1)       # CHANGE CODE HERE
'''
class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.Conv_func_1 = nn.Conv2d(1, 50, 5)
        self.Conv_func_2 = nn.Conv2d(50, 100, 5)
        self.Linear_func_1 = nn.Linear(64 * 5 * 5, 32 * 5 * 5, bias = False)
        self.Linear_func_2 = nn.Linear(32 * 5 * 5, 10, bias = False)

    def forward(self, x):
        #print(1)
        relu_step_1 = F.relu(self.Conv_func_1(x))
        #print(2)
        pool_step_1 = F.max_pool2d(relu_step_1, 2)
        #print(3)
        relu_step_2 = F.relu(self.Conv_func_2(pool_step_1))
        #print(4)
        pool_step_2 = F.max_pool2d(relu_step_2, 2)
        #print(5)
        view_step = pool_step_2.view(pool_step_2.size()[0], -1)
        #print(6)
        linear_step = self.Linear_func_2(self.Linear_func_1(view_step))
        #print(7)
        relu_step_3 = F.relu(linear_step)
        return F.log_softmax(relu_step_3, dim = 1)          # CHANGE CODE HERE
'''
class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.Conv_func_1 = nn.Conv2d(1, 50, 5)
        self.Conv_func_2 = nn.Conv2d(50, 100, 5)
        self.Linear_func = nn.Linear(64 * 5 * 5, 10)

    def forward(self, x):
        #print(1)
        relu_step_1 = F.relu(self.Conv_func_1(x))
        #print(2)
        pool_step_1 = F.max_pool2d(relu_step_1, 2)
        #print(3)
        relu_step_2 = F.relu(self.Conv_func_2(pool_step_1))
        #print(4)
        pool_step_2 = F.max_pool2d(relu_step_2, 2)
        #print(5)
        view_step = pool_step_2.view(pool_step_2.size()[0], -1)
        #print(6)
        linear_step = self.Linear_func(view_step)
        #print(7)
        return F.log_softmax(linear_step, dim = 1)          # CHANGE CODE HERE