import h5py
import numpy as np
import paddle
import paddle.nn as nn
from process_data_pdpd import data_norm, data_sampler
from basic_model_pdpd import gradients, DeepModel_single, DeepModel_multi
import visual_data
import matplotlib.pyplot as plt
import time
import os
import argparse
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_args():

    parser = argparse.ArgumentParser('PINNs for naiver-stokes cylinder with Karman Vortex', add_help=False)
    parser.add_argument('-f', type=str, default="external parameters")
    parser.add_argument('--Layer_depth', default=5, type=int, help="Number of Layers depth")
    parser.add_argument('--Layer_width', default=64, type=int, help="Number of Layers width")
    parser.add_argument('--points_name', default="24+6+4", type=str, help="distribution of supervised points")
    parser.add_argument('--Net_pattern', default='single', type=str, help="single or multi networks")
    parser.add_argument('--epochs_adam', default=400000, type=int, help='total training epoch')
    parser.add_argument('--save_freq', default=5000, type=int, help="frequency to save model and image")
    parser.add_argument('--print_freq', default=1000, type=int, help="frequency to print loss")
    parser.add_argument('--device', default=0, type=int, help="gpu id")
    parser.add_argument('--work_name', default='', type=str, help="work path to save files")

    parser.add_argument('--Nx_EQs', default=30000, type=int, help="xy sampling in for equation loss")
    parser.add_argument('--Nt_EQs', default=10, type=int, help="time sampling in for equation loss")
    parser.add_argument('--Nt_BCs', default=200, type=int, help="time sampling in for boundary loss")

    return parser.parse_args()

def read_data():
    data = h5py.File('./data/cyl_Re250.mat', 'r')

    nodes = np.array(data['grids_']).squeeze().transpose((3, 2, 1, 0)) # [Nx, Ny, Nf]
    field = np.array(data['fields_']).squeeze().transpose((3, 2, 1, 0)) # [Nt, Nx, Ny, Nf]
    times = np.array(data['dynamics_']).squeeze().transpose((1, 0))[3::4, (0,)] # (800, 3) -> (200, 1)
    nodes = nodes[0]
    times = times - times[0, 0]

    return times[:120], nodes[:, :, 1:], field[:120, :, :, :]   # Nx / 2


def BCS_ICS(nodes, points):
    BCS = []
    ICS = []
    Num_Nodes = nodes.shape[0] * nodes.shape[1]
    Index = np.arange(Num_Nodes).reshape((nodes.shape[0], nodes.shape[1]))
    BCS.append(np.concatenate((Index[:93, -1], Index[284:, -1]), axis=0)) #in
    BCS.append(Index[93:284, -1]) #out
    BCS.append(Index[:, 0]) #wall

    ######监督测点生成
    if points == "24+6+4":
        point = np.concatenate((Index[175:220:15, 0:64:8].reshape(-1, 1),
                                Index[378:360:-15, 0:16:8].reshape(-1, 1),
                                Index[14, 0:16:8].reshape(-1, 1)), axis=0)[:, 0]
        BCS.append(np.concatenate((Index[::96, 1], point), axis=0))  # 24+6+4  尾迹+前缘+圆周
    elif points == "30+4":
        point = np.concatenate((Index[175:220:15, 0:80:8].reshape(-1, 1)), axis=0)
        BCS.append(np.concatenate((Index[::96, 1], point), axis=0))  # 30+4
    elif points == "48+12+8":
        point = np.concatenate((Index[175:220:15, 0:64:4].reshape(-1, 1),
                                Index[378:360:-15, 0:32:8].reshape(-1, 1),
                                Index[14, 0:32:8].reshape(-1, 1)), axis=0)[:, 0]
        BCS.append(np.concatenate((Index[::48, 1], point), axis=0))  # 48+12+8  尾迹+前缘+圆周
    elif points == "60+8":
        point = np.concatenate((Index[175:220:15, 0:80:4].reshape(-1, 1)), axis=0)
        BCS.append(np.concatenate((Index[::48, 1], point), axis=0))  # 60+8
    elif points == "96+24+16":
        point = np.concatenate((Index[175:220:15, 0:64:2].reshape(-1, 1),
                                Index[378:360:-15, 0:32:4].reshape(-1, 1),
                                Index[14, 0:32:4].reshape(-1, 1)), axis=0)[:, 0]
        BCS.append(np.concatenate((Index[::24, 1], point), axis=0))  # 96+24+16
    elif points == "120+16":
        point = np.concatenate((Index[175:220:15, 0:80:2].reshape(-1, 1)), axis=0)
        BCS.append(np.concatenate((Index[::24, 1], point), axis=0))  # 120+16
    elif points == "192+48+32":
        point = np.concatenate((Index[175:220:15, 0:128:2].reshape(-1, 1),
                                Index[378:360:-15, 0:64:4].reshape(-1, 1),
                                Index[14, 0:64:4].reshape(-1, 1)), axis=0)[:, 0]
        BCS.append(np.concatenate((Index[::12, 1], point), axis=0))  # 192+48+32
    elif points == "240+32":
        point = np.concatenate((Index[175:220:15, 0:160:2].reshape(-1, 1)), axis=0)
        BCS.append(np.concatenate((Index[::12, 1], point), axis=0))  # 240+32
    elif points == "13":
        BCS.append(Index[::30, 1])  # 13
    elif points == "12+4":
        point = np.concatenate((Index[175:220:30, 0:48:8].reshape(-1, 1)), axis=0)
        BCS.append(np.concatenate((Index[::96, 1], point), axis=0))  # 12+4
    else:
        BCS.append(Index[::60, 1])  # 6

    ICS.append(Index.reshape(-1))
    INN = np.setdiff1d(ICS[0], np.concatenate(BCS[:-1], axis=0)) #其余点

    return INN, BCS, ICS

class Net_single(DeepModel_single):
    def __init__(self, planes, data_norm):
        super(Net_single, self).__init__(planes, data_norm, active=nn.Tanh())
        self.Re = 250.

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

class Net_multi(DeepModel_multi):
    def __init__(self, planes, data_norm):
        super(Net_multi, self).__init__(planes, data_norm, active=nn.Tanh())
        self.Re = 250.

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



def train(inn_var, BCs, ICs, out_true, model, Loss, optimizer, scheduler, log_loss, opts):


    inn = BCs[0].sampling(Nx=opts.Nx_EQs, Nt=opts.Nt_EQs)  #随机抽取守恒损失计算点
    BC_in = BCs[1].sampling(Nx='all', Nt=opts.Nt_BCs); ind_BC_in = BC_in.shape[0]  #入口
    BC_out = BCs[2].sampling(Nx='all', Nt=opts.Nt_BCs); ind_BC_out = BC_out.shape[0] + ind_BC_in #出口
    BC_wall = BCs[3].sampling(Nx='all', Nt=opts.Nt_BCs); ind_BC_wall = BC_wall.shape[0] + ind_BC_out #圆柱
    BC_meas = BCs[4].sampling(Nx='all', Nt=opts.Nt_BCs); ind_BC_meas = BC_meas.shape[0] + ind_BC_wall

    IC_0 = ICs[0].sampling(Nx='all')  #初始场

    inn_BCs = paddle.concat((inn_var[BC_in], inn_var[BC_out], inn_var[BC_wall], inn_var[BC_meas], inn_var[IC_0]), axis=0)
    out_BCs = paddle.concat((out_true[BC_in], out_true[BC_out], out_true[BC_wall], out_true[BC_meas], out_true[IC_0]), axis=0)

    inn_EQs = inn_var[inn]
    out_EQs = out_true[inn]


    optimizer.clear_grad()

    inn_EQs.stop_gradient = False
    out_EQs_ = model(inn_EQs)
    res_EQs = model.equation(inn_EQs, out_EQs_)
    inn_BCs.stop_gradient = True
    out_BCs_ = model(inn_BCs)

    bcs_loss_1 = Loss(out_BCs_[:ind_BC_in, 1:], out_BCs[:ind_BC_in, 1:])  #进口速度
    bcs_loss_2 = Loss(out_BCs_[ind_BC_in:ind_BC_out, 0], out_BCs[ind_BC_in:ind_BC_out, 0])  #出口压力
    bcs_loss_3 = Loss(out_BCs_[ind_BC_out:ind_BC_wall, 1:], out_BCs[ind_BC_out:ind_BC_wall, 1:])  #壁面速度
    bcs_loss_4 = Loss(out_BCs_[ind_BC_wall:ind_BC_meas, :], out_BCs[ind_BC_wall:ind_BC_meas, :])  #监督测点
    ics_loss_0 = Loss(out_BCs_[ind_BC_meas:, :], out_BCs[ind_BC_meas:, :])   #初始条件损失
    eqs_loss = (res_EQs**2).mean()   #方程损失

    loss_batch = bcs_loss_1 + bcs_loss_2 + bcs_loss_3 + bcs_loss_4 + ics_loss_0 + eqs_loss
    loss_batch.backward()

    data_loss = Loss(out_EQs_, out_EQs)   #全部点的data loss  没有用来训练
    log_loss.append([eqs_loss.item(), bcs_loss_1.item(), bcs_loss_2.item(), bcs_loss_3.item(), bcs_loss_4.item(),
                     ics_loss_0.item(), data_loss.item()])

        # return loss_batch

    # optimizer.step(closure)
    optimizer.step()
    scheduler.step()

def inference(inn_var, model):

    with paddle.no_grad():

        out_pred = model(inn_var)

    return out_pred


if __name__ == '__main__':

    opts = get_args()
    print(opts)

    if paddle.fluid.is_compiled_with_cuda():
        paddle.set_device("gpu:" + str(opts.device)) # 指定第一块gpu
        device = "gpu:" + str(opts.device)
    else:
        paddle.set_device('cpu')
        device = 'cpu'

    points_name = opts.points_name
    work_name = 'NS-cylinder-2d-t_pdpd_' + points_name + '-' + opts.work_name
    work_path = os.path.join('work', work_name)
    tran_path = os.path.join('work', work_name, 'train')
    isCreated = os.path.exists(tran_path)
    if not isCreated:
        os.makedirs(tran_path)

    # 将控制台的结果输出到a.log文件，可以改成a.txt
    sys.stdout = visual_data.Logger(os.path.join(work_path, 'train.log'), sys.stdout)


    times, nodes, field = read_data()
    INN, BCS, ICS = BCS_ICS(nodes, points_name)

    Nt, Nx, Ny, Nf = field.shape[0], field.shape[1], field.shape[2], field.shape[3]
    times = np.tile(times[:, None, None, :], (1, Nx, Ny, 1))
    nodes = np.tile(nodes[None, :, :, :], (Nt, 1, 1, 1))
    times = times.reshape(-1, 1)
    nodes = nodes.reshape(-1, 2)
    field = field.reshape(-1, Nf)
    input = np.concatenate((nodes, times), axis=-1)
    input_norm = data_norm(input, method='mean-std')
    field_norm = data_norm(field, method='mean-std')

    input_visual = input.reshape((Nt, Nx, Ny, 3))
    add_input = input_visual[:, 0, :, :].reshape((Nt, -1, Ny, 3))
    input_visual = np.concatenate((input_visual, add_input), axis=1)
    field_visual = field.reshape((Nt, Nx, Ny, Nf))
    add_field = field_visual[:, 0, :, :].reshape((Nt, -1, Ny, Nf))
    field_visual = np.concatenate((field_visual, add_field), axis=1)

    # Training Data
    input = paddle.to_tensor(input, dtype='float32')
    field = paddle.to_tensor(field, dtype='float32')

    NumNodes = Nx * Ny
    BC_in = data_sampler(BCS[0], NumNodes, time=Nt)
    BC_out = data_sampler(BCS[1], NumNodes, time=Nt)
    BC_cyl = data_sampler(BCS[2], NumNodes, time=Nt)
    BC_meas = data_sampler(BCS[3], NumNodes, time=Nt)
    IC_cyl = data_sampler(ICS[0], NumNodes, time=0)
    IN_cyl = data_sampler(INN, NumNodes, time=Nt)
    BCs = [IN_cyl, BC_in, BC_out, BC_cyl,  BC_meas]
    ICs = [IC_cyl,]

    L1Loss = nn.L1Loss()
    HBLoss = nn.SmoothL1Loss()
    L2Loss = nn.MSELoss()

    planes = [3,] + [opts.Layer_width] * opts.Layer_depth + [3,]
    if opts.Net_pattern == "single":
        Net_model = Net_single(planes=planes, data_norm=(input_norm, field_norm)).to(device)
    elif opts.Net_pattern == "multi":
        Net_model = Net_multi(planes=planes, data_norm=(input_norm, field_norm)).to(device)

    Boundary_epoch1 = [opts.epochs_adam*8/10, opts.epochs_adam*9/10]
    Boundary_epoch2 = [opts.epochs_adam*11/10, opts.epochs_adam*12/10]
    Scheduler1 = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.001, milestones=Boundary_epoch1, gamma=0.1)
    Optimizer1 = paddle.optimizer.Adam(parameters=Net_model.parameters(), learning_rate=Scheduler1, beta1=0.8,
                                        beta2=0.9)
    # Optimizer2 = paddle.incubate.optimizer.functional.minimize_lbfgs(parameters=Net_model.parameters(), learning_rate=.1, max_iter=100)
    # Scheduler2 = paddle.optimizer.lr.MultiStepDecay(Optimizer2, milestones=Boundary_epoch2, gamma=0.1)

    Visual = visual_data.matplotlib_vision('/', field_name=('p', 'u', 'v'), input_name=('x', 'y'))
    Visual.font['size'] = 20

    star_time = time.time()
    start_epoch=0
    log_loss=[]
    """load a pre-trained model"""
    start_epoch, log_loss = Net_model.loadmodel(os.path.join(work_path, 'latest_model.pth'))
    for i in range(start_epoch):
        #  update the learning rate for start_epoch times
        Scheduler1.step()

    # Training
    for iter in range(start_epoch, opts.epochs_adam):

        if iter < opts.epochs_adam:
            train(input, BCs, ICs, field, Net_model, L2Loss, Optimizer1, Scheduler1, log_loss, opts)
            learning_rate = Optimizer1.get_lr()

        if iter > 0 and iter % opts.print_freq == 0:

            print('iter: {:6d}/{:6d}, lr: {:.1e}, cost: {:.2f}, dat_loss: {:.2e} \n'
                  'eqs_loss: {:.2e}, BCS_loss_in: {:.2e}, BCS_loss_out: {:.2e}, '
                  'BCS_loss_wall: {:.2e}, BCS_loss_meas: {:.2e}, ICS_loss_0: {:.2e}'.
                  format(iter, opts.epochs_adam, learning_rate, time.time() - star_time, log_loss[-1][-1],
                         log_loss[-1][0], log_loss[-1][1], log_loss[-1][2],
                         log_loss[-1][3], log_loss[-1][4], log_loss[-1][5]))

            plt.figure(1, figsize=(20, 15))
            plt.clf()
            plt.subplot(2, 1, 1)
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, -1], 'dat_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'eqs_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 5], 'ICS_loss_0')
            plt.subplot(2, 1, 2)
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'BCS_loss_in')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 2], 'BCS_loss_out')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 3], 'BCS_loss_wall')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 4], 'BCS_loss_meas')
            plt.savefig(os.path.join(tran_path, 'log_loss.svg'))

            star_time = time.time()

        if iter > 0 and iter % opts.save_freq == 0:
            input_visual_p = paddle.to_tensor(input_visual[:100:10], dtype='float32', place='gpu:0')
            field_visual_p = inference(input_visual_p, Net_model)
            field_visual_t = field_visual[:100:10]
            field_visual_p = field_visual_p.cpu().numpy()

            for t in range(field_visual_p.shape[0]):
                plt.figure(2, figsize=(20, 10))
                plt.clf()
                Visual.plot_fields_ms(field_visual_t[t], field_visual_p[t], input_visual_p[0, :, :, :2].numpy(),
                                      cmin_max=[[-4, -4], [10, 4]], field_name=['p', 'u', 'v'])
                plt.subplots_adjust(wspace=0.2, hspace=0.3)  # left=0.05, bottom=0.05, right=0.95, top=0.95
                plt.savefig(os.path.join(tran_path, 'loca_' + str(t) + '.jpg'))

                plt.figure(3, figsize=(30, 20))
                plt.clf()
                Visual.plot_fields_ms(field_visual_t[t], field_visual_p[t], input_visual_p[0, :, :, :2].numpy())
                plt.subplots_adjust(wspace=0.2, hspace=0.3)
                plt.savefig(os.path.join(tran_path, 'full_' + str(t) + '.jpg'))

            paddle.save({'epoch': iter, 'model': Net_model.state_dict(), 'log_loss': log_loss},
                        os.path.join(work_path, 'latest_model.pth'))