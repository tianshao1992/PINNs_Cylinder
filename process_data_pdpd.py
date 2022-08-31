import numpy as np
import paddle


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
        if paddle.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - paddle.to_tensor(self.min, place='gpu:0')) \
                    / (paddle.to_tensor(self.max, place='gpu:0') - paddle.to_tensor(self.min, place='gpu:0')) - 1
            elif self.method == "mean-std":
                x = (x - paddle.to_tensor(self.mean, place='gpu:0')) / (paddle.to_tensor(self.std, place='gpu:0'))
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std)

        return x

    def back(self, x):
        if paddle.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (paddle.to_tensor(self.max, place='gpu:0')
                                   - paddle.to_tensor(self.min, place='gpu:0')) + paddle.to_tensor(self.min, place='gpu:0')
            elif self.method == "mean-std":
                x = x * (paddle.to_tensor(self.std, place='gpu:0')) + paddle.to_tensor(self.mean, place='gpu:0')
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