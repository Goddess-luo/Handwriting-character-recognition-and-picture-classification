# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.linear_func_1 = nn.Linear(2, num_hid)
        self.linear_func_2 = nn.Linear(num_hid, 1)

    def forward(self, input):
        #print(input.view(input.size()))
        #print(input.shape)
        input_x = input[:, 0]
        input_y = input[:, 1]
        #print(1)
        multi_x = torch.mul(input_x, input_x)
        multi_y = torch.mul(input_y, input_y)
        #print(2)
        sqrt_r = np.sqrt(multi_x + multi_y)
        polar_r = torch.clone(sqrt_r)
        polar_a = torch.atan2(input_y, input_x)
        #print(3)
        combine_co_ord = torch.stack((polar_r, polar_a), dim=1)
        #print(4)
        #print(input_co_ordinates.shape)
        #relu_step = torch.nn.functional.relu(combine_co_ord)
        linear_step_1 = self.linear_func_1(combine_co_ord)
        #print(5)
        self.hid1 = torch.tanh(linear_step_1)
        #print(6)
        linear_step_2 = self.linear_func_2(self.hid1)
        #print(7)
        return torch.sigmoid(linear_step_2)

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.linear_func_1 = nn.Linear(2, num_hid)
        self.linear_func_2 = nn.Linear(num_hid, num_hid)
        self.linear_func_3 = nn.Linear(num_hid, 1)

    def forward(self, input):
        linear_step_1 = self.linear_func_1(input)
        #print(1)
        self.hid1 = torch.relu(torch.tanh(linear_step_1))
        #print(3)
        linear_step_2 = self.linear_func_2(self.hid1)
        #print(4)
        self.hid2 = torch.relu(torch.tanh(linear_step_2))
        #print(6)
        linear_step_3 = self.linear_func_3(self.hid2)
        #print(7)
        return torch.sigmoid(linear_step_3)



def graph_hidden(net, layer, node):
    # This function copied from spiral_main.py -> def graph_output(net)
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)

    with torch.no_grad():  # suppress updating of gradients
        net.eval()  # toggle batch norm, dropout
        output = net(grid)
        net.train()  # toggle batch norm, dropout back again    切换批处理规范，再次退出

        # this part have changed
        if net == 'polar' and layer == 1:
            #print(net)
            #print(1)
            output = net.hid1[:, node]
        elif net != 'polar':
            if layer == 1:
                # print(net)
                # print(2)
                output = net.hid1[:, node]
            else:
                # print(net)
                # print(3)
                output = net.hid2[:, node]


        pred = (output >= 0.5).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]), cmap='Wistia')
