import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from parameterfree import COCOB


class fbsde():
    def __init__(self, x_0, b, sigma, f, g, T, dim_x, dim_y, dim_d, guess_y_0=[0, 1]):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.x_0 = x_0.to(device)
        self.b = b
        self.sigma = sigma
        self.f = f
        self.g = g
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d
        self.guess_y_0 = guess_y_0  # range


class Model(nn.Module):
    def __init__(self, equation, dim_h):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(Model, self).__init__()
        self.equation = equation

        # specifying parameters of NN
        l = torch.rand(equation.dim_y, device=device)
        self.y_0 = nn.Parameter(
            equation.guess_y_0[0]*l + equation.guess_y_0[1]*(1-l))
        # dim_x + 1  the extra 1 for time
        self.linear1 = nn.Linear(equation.dim_x+1, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y*equation.dim_d)
        self.bn1 = nn.BatchNorm1d(dim_h)
        self.bn2 = nn.BatchNorm1d(dim_h)

    def get_z(self, x, t):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tmp = torch.cat((x, t*torch.ones(x.size()[0], 1, device=device)), 1)
        tmp = F.gelu(self.linear1(tmp))
        tmp = self.bn1(F.gelu(self.linear2(tmp)))
        tmp = self.bn2(F.gelu(self.linear3(tmp)))
        return self.linear4(tmp).reshape(-1, self.equation.dim_y, self.equation.dim_d)

    def forward(self, batch_size, N):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dt = self.equation.T / N
        x = self.equation.x_0 + \
            torch.zeros(batch_size, self.equation.dim_x, device=device)
        y = self.y_0+torch.zeros(batch_size,
                                 self.equation.dim_y, device=device)

        for i in range(N):
            t = dt*i
            z = self.get_z(x, t)

            dW = torch.randn(batch_size, self.equation.dim_d,
                             1, device=device) * np.sqrt(dt)
            x = x+self.equation.b(t, x, y)*dt+torch.matmul(
                self.equation.sigma(t, x), dW).reshape(-1, self.equation.dim_x)
            y = y-self.equation.f(t, x, y, z)*dt + torch.matmul(z,
                                                                dW).reshape(-1, self.equation.dim_y)
        return x, y


class BSDEsolver():
    def __init__(self, equation, dim_h):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Model(equation, dim_h).to(device)
        self.equation = equation

    def train(self, batch_size, N, itr, log):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.MSELoss().to(device)
        # optimizer = torch.optim.Adam(self.model.parameters())
        optimizer = COCOB(self.model.parameters())
        loss_data, y0_data = [], []

        for i in range(itr):
            x, y = self.model(batch_size, N)
            loss = criterion(self.equation.g(x), y)
            loss_data.append(float(loss))
            y0_data.append(float(self.model.y_0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log and i % int(itr/20) == 0:
                print(
                    f"loss: {float(loss):7.2f} y0: {float(self.model.y_0):7.2f} done: {i/itr*100:5.2f}% Iteration: {i}")
        return loss_data, y0_data
