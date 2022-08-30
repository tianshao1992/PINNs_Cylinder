import torch
import torch.nn as nn
import numpy as np

def gradients(y, x, order=1):
    if order == 1:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
    else:
        return gradients(gradients(y, x), x, order=order - 1)


def jacobians(u, x, order=1):
    if order == 1:
        return torch.autograd.functional.jacobian(u, x, create_graph=True)[0]


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1)
            # nn.init.xavier_uniform(m.weight, gain=1)
            m.bias.data.zero_()

class DeepModel_multi(nn.Module):
    def __init__(self, planes, data_norm, active=nn.GELU()):
        super(DeepModel_multi, self).__init__()
        self.planes = planes
        self.active = active

        self.x_norm = data_norm[0]
        self.f_norm = data_norm[1]
        self.g_norm = data_norm[2]
        self.layers = nn.ModuleList()
        for j in range(self.planes[-1]):
            layer = []
            for i in range(len(self.planes) - 2):
                layer.append(nn.Linear(self.planes[i], self.planes[i + 1]))
                layer.append(self.active)
            layer.append(nn.Linear(self.planes[-2], 1))
            self.layers.append(nn.Sequential(*layer))
        self.apply(initialize_weights)

    def forward(self, in_var):
        in_var = self.x_norm.norm(in_var)
        y = []
        for i in range(self.planes[-1]):
            y.append(self.layers[i](in_var))
        return self.f_norm.back(torch.cat(y, dim=-1))

    def loadmodel(self, File):
        try:
            checkpoint = torch.load(File)
            self.load_state_dict(checkpoint['model'])        # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print("load start epoch at " + str(start_epoch))
            try:
                log_loss = checkpoint['log_loss']
            except:
                log_loss= []
            return start_epoch, log_loss
        except:
            print("load model failed, start a new model!")
            return 0, []

    def equation(self, inv_var, out_var):
        return 0

class DeepModel_single(nn.Module):
    def __init__(self, planes, data_norm, active=nn.GELU()):
        super(DeepModel_single, self).__init__()
        self.planes = planes
        self.active = active

        self.x_norm = data_norm[0]
        self.f_norm = data_norm[1]
        if len(data_norm) > 2:
            self.g_norm = data_norm[2]
        self.layers = nn.ModuleList()
        for i in range(len(self.planes)-2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1]))
            self.layers.append(self.active)
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1]))

        self.layers = nn.Sequential(*self.layers)
        self.apply(initialize_weights)

    def forward(self, inn_var, is_norm=True):
        inn_var = self.x_norm.norm(inn_var)
        out_var = self.layers(inn_var)

        if is_norm:
            return self.f_norm.back(out_var)
        else:
            return out_var


    def loadmodel(self, File):
        try:
            checkpoint = torch.load(File)
            self.load_state_dict(checkpoint['model'])        # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print("load start epoch at" + str(start_epoch))
            log_loss = checkpoint['log_loss']
            return start_epoch, log_loss
        except:
            print("load model failed, start a new model!")
            return 0, []

    def equation(self, **kwargs):
        return 0

    def dirichlet(self, ind, out_var, value, is_norm=False):
        if is_norm:
            res = self.f_norm.norm(out_var[ind, :]) - self.f_norm.norm(value[ind].unsqueeze(-1))
        else:
            res = out_var[ind, :] - value[ind, :]
        return res

    def neumann(self, ind, out_var, in_var, value, norm):
        duda = gradients(out_var.sum(), in_var)
        dudx = duda[ind, :1]
        dudy = duda[ind, 1:]
        res = norm[ind, :1] * dudx + norm[ind, 1:] * dudy - value[ind, :]
        return res


class Dynamicor(nn.Module):

    def __init__(self, device, coords):
        super(Dynamicor, self).__init__()
        self.c, self.rho, self.uinf = 1.0, 1.0, 1.0
        self.Re = 250
        self.miu = self.c * self.rho * self.uinf / self.Re
        self.device = device
        self.coords = torch.tensor(coords, dtype=torch.float, device=device)

        cir_num = self.coords.shape[0]
        self.ind1 = [i for i in range(cir_num)]
        self.ind2 = [i for i in range(1, cir_num)]
        self.ind2.append(0)

        self.T_vector = self.coords[self.ind2, 0, :] - self.coords[self.ind1, 0, :]
        self.N_vector = torch.matmul(self.T_vector, torch.tensor(np.array([[0, -1], [1, 0]]), dtype=torch.float, device=self.device))
        self.T_norm = torch.norm(self.T_vector, dim=-1)
        self.N_norm = torch.norm(self.N_vector, dim=-1)
        self.delta = self.coords[:, 0] - self.coords[:, 1]
        self.delta = torch.norm(self.delta, dim=-1)



    def cal_force(self, fields):

        p = fields[:, :, 0, 0]
        p_ave = (p[:, self.ind1] + p[:, self.ind2]) / 2.
        Ft_n = p_ave * self.T_norm
        Fx = - Ft_n * self.N_vector[:, 0] / self.N_norm
        Fy = - Ft_n * self.N_vector[:, 1] / self.N_norm

        # tao = -miu * du/dn at grid point
        # fields : [96, 198, (p,u,v)]

        u = fields[:, :, 1, 1:]
        du = (u[:, :, 0] * self.T_vector[:, 0] + u[:, :, 1] * self.T_vector[:, 1]) / self.T_norm
        tau = du / self.delta * self.miu
        tau_ave = (tau[:, self.ind1] + tau[:, self.ind2]) * 0.5
        T_n = tau_ave * self.T_norm
        Tx = T_n * self.T_vector[:, 0] / self.T_norm
        Ty = T_n * self.T_vector[:, 1] / self.T_norm

        Fx = torch.sum(Fx, dim=1)
        Fy = torch.sum(Fy, dim=1)
        Tx = torch.sum(Tx, dim=1)
        Ty = torch.sum(Ty, dim=1)

        return Fx, Fy, Tx, Ty


    def forward(self, fields):

        Fx, Fy, Tx, Ty = self.cal_force(fields)

        Fx += Tx
        Fy += Ty
        CL = Fy/(0.5*self.rho*self.uinf**2)
        CD = Fx / (0.5 * self.rho * self.uinf ** 2)
        return torch.stack((Fy, Fx, CL, CD), dim=-1)

