import torch
import torch.nn as nn

def gradients(y, x, order=1):
    if order == 1:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
    else:
        return gradients(gradients(y, x), x, order=order - 1)


def jacobians(u, x, order=1):
    if order == 1:
        return torch.autograd.functional.jacobian(u, x, create_graph=True)[0]

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
            print("load start epoch at " + str(start_epoch))
            try:
                log_loss = checkpoint['log_loss']
            except:
                log_loss= []
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

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1)
            # nn.init.xavier_uniform(m.weight, gain=1)
            m.bias.data.zero_()