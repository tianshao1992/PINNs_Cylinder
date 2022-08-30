import numpy as np
import torch


class data_norm():

    def __init__(self, data, method="min-max"):
        axis = tuple(range(len(data.shape) - 1))
        self.method = method
        if method == "min-max":
            self.max = np.max(data, axis=axis)
            self.min = np.min(data, axis=axis)

        elif method == "mean-std":
            self.mean = np.mean(data, axis=axis)
            self.std = np.std(data, axis=axis)

    def norm(self, x):
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - torch.tensor(self.min, device=x.device)) \
                    / (torch.tensor(self.max, device=x.device) - torch.tensor(self.min, device=x.device)) - 1
            elif self.method == "mean-std":
                x = (x - torch.tensor(self.mean, device=x.device)) / (torch.tensor(self.std, device=x.device))
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std)

        return x

    def back(self, x):
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (torch.tensor(self.max, device=x.device)
                                   - torch.tensor(self.min, device=x.device)) + torch.tensor(self.min, device=x.device)
            elif self.method == "mean-std":
                x = x * (torch.tensor(self.std, device=x.device)) + torch.tensor(self.mean, device=x.device)
        else:
            if self.method == "min-max":
                x = (x + 1) / 2 * (self.max - self.min) + self.min
            elif self.method == "mean-std":
                x = x * (self.std) + self.mean
        return x


class data_sampler():

    def __init__(self, coord, all_coord, time=0):
        self.coord = coord
        self.all_coord = all_coord
        self.time = np.arange(time)

    def sampling(self, Nx, Nt=10):
        index = []
        if self.time.shape[0] > 1:
            for _ in range(Nt):
                index.append(self.mesh_sampling(Nx)+
                             self.all_coord*np.random.choice(self.time, 1, replace=False))
        else:
            index.append(self.mesh_sampling(Nx))

        return np.array(index, dtype=np.int32).flatten()


    def mesh_sampling(self, Nx):
        if Nx == 'all':
            index = self.coord
        else:
            index = np.random.choice(self.coord, Nx, replace=False)
        return index