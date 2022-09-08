import numpy as np
import paddle


class data_norm():

    def __init__(self, data, method="min-max"):
        axis = tuple(range(len(data.shape) - 1))
        self.method = method
        if method == "min-max":
            self.max = np.max(data, axis=axis)
            self.min = np.min(data, axis=axis)
            self.max_ = paddle.to_tensor(self.max)
            self.min_ = paddle.to_tensor(self.min)

        elif method == "mean-std":
            self.mean = np.mean(data, axis=axis)
            self.std = np.std(data, axis=axis)
            self.mean_ = paddle.to_tensor(self.mean)
            self.std_ =  paddle.to_tensor(self.std)

    def norm(self, x):
        if paddle.is_tensor(x):
            if self.method == "min-max":
                y = []
                for i in range(x.shape[-1]):
                    y.append(paddle.scale(x[..., i:i+1], 2/(self.max_[i]-self.min_[i]),
                                          -(self.max_[i]+self.min_[i])/(self.max_[i]-self.min_[i])))
            elif self.method == "mean-std":
                y = []
                for i in range(x.shape[-1]):
                    y.append(paddle.scale(x[..., i:i+1], 1/self.std_[i], -self.mean_[i]/self.std[i]))
                x = paddle.concat(y, axis=-1)
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std)

        return x

    def back(self, x):
        if paddle.is_tensor(x):
            if self.method == "min-max":
                y = []
                for i in range(x.shape[-1]):
                    y.append(paddle.scale(x[..., i:i+1], (self.max_[i]-self.min_[i])/2, (self.max_[i]+self.min_[i])/2))
                x = paddle.concat(y, axis=-1)
            elif self.method == "mean-std":
                y = []
                for i in range(x.shape[-1]):
                    y.append(paddle.scale(x[..., i:i+1], self.std_[i], self.mean_[i]))
                x = paddle.concat(y, axis=-1)
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