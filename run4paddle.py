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
    def __init__(self, planes, active=nn.Tanh()):
        super(DeepModel_multi, self).__init__()
        self.planes = planes
        self.active = active
        self.Re = 250.
        self.layers = nn.LayerList()

        for j in range(self.planes[-1]):
            layer = []
            for i in range(len(self.planes) - 2):
                layer.append(nn.Linear(self.planes[i], self.planes[i + 1], weight_attr=nn.initializer.XavierNormal()))
                layer.append(self.active)
            layer.append(nn.Linear(self.planes[-2], 1, weight_attr=nn.initializer.XavierNormal()))
            self.layers.append(nn.Sequential(*layer))
            # self.layers[-1].apply(initialize_weights)

    def forward(self, in_var):
        # in_var = self.x_norm.norm(in_var)
        # in_var = in_var * self.input_transform
        y = []
        for i in range(self.planes[-1]):
            y.append(self.layers[i](in_var))
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



    def equation(self, inn_var, out_var):
        # a = grad(psi.sum(), in_var, create_graph=True, retain_graph=True)[0]
        p, u, v = out_var[:, 0:1], out_var[:, 1:2], out_var[:, 2:3]

        duda = gradients(u, inn_var)
        dudx, dudy, dudt = duda[:, 0:1], duda[:, 1:2], duda[:, 2:3]
        dvda = gradients(v, inn_var)
        dvdx, dvdy, dvdt = dvda[:, 0:1], dvda[:, 1:2], dvda[:, 2:3]
        d2udx2 = gradients(dudx, inn_var)[:, 0:1]
        d2udy2 = gradients(dudy, inn_var)[:, 1:2]
        d2vdx2 = gradients(dvdx, inn_var)[:, 0:1]
        d2vdy2 = gradients(dvdy, inn_var)[:, 1:2]
        dpda = gradients(p, inn_var)
        dpdx, dpdy = dpda[:, 0:1], dpda[:, 1:2]

        eq1 = dudt + (u * dudx + v * dudy) + dpdx - 1 / self.Re * (d2udx2 + d2udy2)
        eq2 = dvdt + (u * dvdx + v * dvdy) + dpdy - 1 / self.Re * (d2vdx2 + d2vdy2)
        eq3 = dudx + dvdy
        eqs = paddle.concat((eq1, eq2, eq3), axis=1)
        return eqs


class DeepModel_single(nn.Layer):
    def __init__(self, planes, active=nn.Tanh()):
        super(DeepModel_single, self).__init__()
        self.planes = planes
        self.active = active
        self.Re = 250.
        self.layers = nn.LayerList()
        for i in range(len(self.planes) - 2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1], weight_attr=nn.initializer.XavierNormal()))
            self.layers.append(self.active)
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1], weight_attr=nn.initializer.XavierNormal()))

        self.layers = nn.Sequential(*self.layers)
        # self.apply(initialize_weights)

    def forward(self, inn_var, is_norm=True):
        # inn_var = self.x_norm.norm(inn_var)
        out_var = self.layers(inn_var)
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

    def equation(self, inn_var, out_var):
        # a = grad(psi.sum(), in_var, create_graph=True, retain_graph=True)[0]
        p, u, v = out_var[:, 0:1], out_var[:, 1:2], out_var[:, 2:3]

        duda = gradients(u, inn_var)
        dudx, dudy, dudt = duda[:, 0:1], duda[:, 1:2], duda[:, 2:3]
        dvda = gradients(v, inn_var)
        dvdx, dvdy, dvdt = dvda[:, 0:1], dvda[:, 1:2], dvda[:, 2:3]
        d2udx2 = gradients(dudx, inn_var)[:, 0:1]
        d2udy2 = gradients(dudy, inn_var)[:, 1:2]
        d2vdx2 = gradients(dvdx, inn_var)[:, 0:1]
        d2vdy2 = gradients(dvdy, inn_var)[:, 1:2]
        dpda = gradients(p, inn_var)
        dpdx, dpdy = dpda[:, 0:1], dpda[:, 1:2]

        eq1 = dudt + (u * dudx + v * dudy) + dpdx - 1 / self.Re * (d2udx2 + d2udy2)
        eq2 = dvdt + (u * dvdx + v * dvdy) + dpdy - 1 / self.Re * (d2vdx2 + d2vdy2)
        eq3 = dudx + dvdy
        eqs = paddle.concat((eq1, eq2, eq3), axis=1)
        return eqs

if __name__ == '__main__':

    paddle.set_device("gpu:" + str(0))  # 指定第一块gpu
    device = "gpu:" + str(0)

    planes = [3, ] + [64] * 3 + [3, ]
    Net_model_single = DeepModel_single(planes=planes).to(device)
    Net_model_multi = DeepModel_multi(planes=planes).to(device)


    xyt = np.random.rand(100, 3)
    xyt = paddle.to_tensor(xyt, dtype='float32')
    xyt.stop_gradient = False

    puv = Net_model_multi(xyt)
    eqs = Net_model_multi.equation(xyt, puv)
    loss = (eqs**2).mean()
    loss.backward()

    puv = Net_model_single(xyt)
    eqs = Net_model_single.equation(xyt, puv)
    loss = (eqs**2).mean()
    loss.backward()