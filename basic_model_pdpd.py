import paddle
import paddle.nn as nn
import numpy as np


def gradients(y, x, order=1, create=True):
    if order == 1:
        return paddle.autograd.grad(y, x, create_graph=create, retain_graph=True)[0]
    else:
        return paddle.stack([paddle.autograd.grad([y[:, i].sum()], [x], create_graph=True, retain_graph=True)[0]
                             for i in range(y.shape[1])], axis=-1)


class DeepModel_multi(nn.Layer):
    def __init__(self, planes, data_norm, active=nn.GELU()):
        super(DeepModel_multi, self).__init__()
        self.planes = planes
        self.active = active

        self.x_norm = data_norm[0]
        self.f_norm = data_norm[1]
        self.layers = nn.LayerList()

        for j in range(self.planes[-1]):
            layer = []
            for i in range(len(self.planes) - 2):
                layer.append(nn.Linear(self.planes[i], self.planes[i + 1], weight_attr=nn.initializer.XavierNormal()))
                layer.append(self.active)
            layer.append(nn.Linear(self.planes[-2], 1, weight_attr=nn.initializer.XavierNormal()))
            self.layers.append(nn.Sequential(*layer))
            # self.layers[-1].apply(initialize_weights)

    def forward(self, in_var, is_norm=True):
        in_var = self.x_norm.norm(in_var)
        # in_var = in_var * self.input_transform
        y = []
        for i in range(self.planes[-1]):
            y.append(self.layers[i](in_var))
        if is_norm:
            return self.f_norm.back(paddle.concat(y, axis=-1))
        else:
            return paddle.concat(y, axis=-1)

    def loadmodel(self, File):
        try:
            checkpoint = paddle.load(File)
            self.set_state_dict(checkpoint['model'])  # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print("load start epoch at " + str(start_epoch))
            log_loss = checkpoint['log_loss']  # .tolist()
            return start_epoch, log_loss
        except:
            print("load model failed！ start a new model.")
            return 0, []

    def equation(self, inv_var, out_var):
        return 0


class DeepModel_single(nn.Layer):
    def __init__(self, planes, data_norm, active=nn.GELU()):
        super(DeepModel_single, self).__init__()
        self.planes = planes
        self.active = active

        self.x_norm = data_norm[0]
        self.f_norm = data_norm[1]
        self.layers = nn.LayerList()
        for i in range(len(self.planes) - 2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1], weight_attr=nn.initializer.XavierNormal()))
            self.layers.append(self.active)
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1], weight_attr=nn.initializer.XavierNormal()))

        self.layers = nn.Sequential(*self.layers)
        # self.apply(initialize_weights)

    def forward(self, inn_var, is_norm=True):
        inn_var = self.x_norm.norm(inn_var)
        out_var = self.layers(inn_var)

        if is_norm:
            return self.f_norm.back(out_var)
        else:
            return out_var

    def loadmodel(self, File):
        try:
            checkpoint = paddle.load(File)
            self.set_state_dict(checkpoint['model'])  # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print("load start epoch at " + str(start_epoch))
            log_loss = checkpoint['log_loss']  # .tolist()
            return start_epoch, log_loss
        except:
            print("load model failed！ start a new model.")
            return 0, []

    def equation(self, **kwargs):
        return 0


class Dynamicor(nn.Layer):

    def __init__(self, device, coords):
        super(Dynamicor, self).__init__()
        self.c, self.rho, self.uinf = 1.0, 1.0, 1.0
        self.Re = 250
        self.miu = self.c * self.rho * self.uinf / self.Re
        self.device = device
        self.coords = paddle.to_tensor(coords, dtype='float32', place=device)

        cir_num = self.coords.shape[0]
        self.ind1 = [i for i in range(cir_num)]
        self.ind2 = [i for i in range(1, cir_num)]
        self.ind2.append(0)

        self.T_vector = paddle.to_tensor(self.coords.numpy()[self.ind2, 0, :]) - paddle.to_tensor(self.coords.numpy()[self.ind1, 0, :])

        self.N_vector = paddle.matmul(self.T_vector, paddle.to_tensor(np.array([[0, -1], [1, 0]]),
                                                                      dtype='float32', place=self.device))
        self.T_norm = paddle.norm(self.T_vector, axis=-1)
        self.N_norm = paddle.norm(self.N_vector, axis=-1)
        self.delta = self.coords[:, 0] - self.coords[:, 1]
        self.delta = paddle.norm(self.delta, axis=-1)



    def cal_force(self, fields):

        p =fields[:, :, 0, 0]
        p_ave = paddle.to_tensor(p.numpy()[:, self.ind1]) + paddle.to_tensor(p.numpy()[:, self.ind2]) / 2.
        Ft_n = p_ave * self.T_norm
        Fx = - Ft_n * self.N_vector[:, 0] / self.N_norm
        Fy = - Ft_n * self.N_vector[:, 1] / self.N_norm

        # tao = -miu * du/dn at grid point
        # fields : [96, 198, (p,u,v)]

        u = fields[:, :, 1, 1:]
        du = (u[:, :, 0] * self.T_vector[:, 0] + u[:, :, 1] * self.T_vector[:, 1]) / self.T_norm
        tau = du / self.delta * self.miu
        tau_ave = (paddle.to_tensor(tau.numpy()[:, self.ind1]) + paddle.to_tensor(tau.numpy()[:, self.ind2])) * 0.5
        # tau_ave = (tau[:, self.ind1] + tau[:, self.ind2]) * 0.5
        T_n = tau_ave * self.T_norm
        Tx = T_n * self.T_vector[:, 0] / self.T_norm
        Ty = T_n * self.T_vector[:, 1] / self.T_norm

        Fx = paddle.sum(Fx, axis=1)
        Fy = paddle.sum(Fy, axis=1)
        Tx = paddle.sum(Tx, axis=1)
        Ty = paddle.sum(Ty, axis=1)

        return Fx, Fy, Tx, Ty


    def forward(self, fields):

        Fx, Fy, Tx, Ty = self.cal_force(fields)

        Fx += Tx
        Fy += Ty
        CL = Fy/(0.5*self.rho*self.uinf**2)
        CD = Fx / (0.5 * self.rho * self.uinf ** 2)
        return paddle.stack((Fy, Fx, CL, CD), axis=-1)
