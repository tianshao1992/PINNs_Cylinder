import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
from process_data import data_norm, data_sampler
from basic_model import gradients, DeepModel_single, Dynamicor
import visual_data
import matplotlib.pyplot as plt
import time
import os
import argparse
from run_train import read_data, Net_multi, Net_single, inference

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_args():

    parser = argparse.ArgumentParser('PINNs for naiver-stokes cylinder with Karman Vortex', add_help=False)
    parser.add_argument('--points_name', default="30+4", type=str)
    parser.add_argument('--Layer_depth', default=6, type=int, help="Number of Layers depth")
    parser.add_argument('--Layer_width', default=64, type=int, help="Number of Layers width")
    parser.add_argument('--Net_pattern', default='single', type=str, help="single or multi networks")
    parser.add_argument('--epochs_adam', default=400000, type=int)
    parser.add_argument('--save_freq', default=2000, type=int, help="frequency to save model and image")
    parser.add_argument('--print_freq', default=500, type=int, help="frequency to print loss")
    parser.add_argument('--device', default=0, type=int, help="gpu id")
    parser.add_argument('--work_name', default='', type=str, help="work path to save files")

    parser.add_argument('--Nx_EQs', default=30000, type=int, help="xy sampling in for equation loss")
    parser.add_argument('--Nt_EQs', default=15, type=int, help="time sampling in for equation loss")
    parser.add_argument('--Nt_BCs', default=120, type=int, help="time sampling in for boundary loss")

    return parser.parse_args()

# def cal_force():


if __name__ == '__main__':

    opts = get_args()

    np.random.seed(2022)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.device)  # 指定第一块gpu

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    points_name = opts.points_name
    work_name = 'NS-cylinder-2d-t_' + points_name + opts.work_name
    work_path = os.path.join('work', work_name,)
    vald_path = os.path.join('work', work_name, 'validation')
    isCreated = os.path.exists(vald_path)
    if not isCreated:
        os.makedirs(vald_path)

    # 将控制台的结果输出到a.log文件，可以改成a.txt
    sys.stdout = visual_data.Logger(os.path.join(work_path, 'valid.log'), sys.stdout)


    print(opts)
    times, nodes, field = read_data()
    Dyn_model = Dynamicor(device, nodes[:, (0,1), :])

    Nt, Nx, Ny, Nf = field.shape[0], field.shape[1], field.shape[2], field.shape[3]

    times = np.tile(times[:, None, None, :], (1, Nx, Ny, 1))
    nodes = np.tile(nodes[None, :, :, :], (Nt, 1, 1, 1))
    times = times.reshape(-1, 1)
    nodes = nodes.reshape(-1, 2)
    # field = field.reshape(-1, Nf)
    input = np.concatenate((nodes, times), axis=-1)
    input_norm = data_norm(input, method='mean-std')
    field_norm = data_norm(field, method='mean-std')

    input_visual = input.reshape((Nt, Nx, Ny, 3))
    add_input = input_visual[:, 0, :, :].reshape((Nt, -1, Ny, 3))
    input_visual = np.concatenate((input_visual, add_input), axis=1)
    field_visual = field.reshape((Nt, Nx, Ny, Nf))
    add_field = field_visual[:, 0, :, :].reshape((Nt, -1, Ny, Nf))
    field_visual = np.concatenate((field_visual, add_field), axis=1)

    planes = [3,] + [opts.Layer_width] * opts.Layer_depth + [3,]
    if opts.Net_pattern == "single":
        Net_model = Net_single(planes=planes, data_norm=(input_norm, field_norm)).to(device)
    elif opts.Net_pattern == "multi":
        Net_model = Net_multi(planes=planes, data_norm=(input_norm, field_norm)).to(device)
    Visual = visual_data.matplotlib_vision(vald_path, field_name=('p', 'u', 'v'), input_name=('x', 'y'))
    Visual.font['size'] = 20
    start_epoch, log_loss = Net_model.loadmodel(os.path.join(work_path, 'latest_model.pth'))

####################################### plot loss #################################################################################
    try:
        print("plot training loss")
        plt.figure(1, figsize=(15, 10))
        plt.clf()
        Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, -1], 'unsupervised loss')
        Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'conversation loss')
        plt.savefig(os.path.join(vald_path, 'loss_eqs_data.jpg'), dpi=300)

        plt.figure(2, figsize=(15, 10))
        plt.clf()
        Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'inlet boundary loss')
        Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 2], 'outlet boundary loss')
        Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 3], 'cylinder boundary loss')
        Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 4], 'measurement loss')
        Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 5], 'initial boundary loss')
        plt.savefig(os.path.join(vald_path, 'loss_boundary.jpg'), dpi=300)
    except:
        print("No loss log")

####################################### plot several fields #################################################################################
    print("plot several true and predicted fields")
    inds = np.concatenate((np.zeros((1,), dtype=np.int32), np.linspace(0, 100, 11, dtype=np.int32)))
    input_visual_p = torch.tensor(input_visual[inds], dtype=torch.float32)
    field_visual_p = inference(input_visual_p.to(device), Net_model)
    field_visual_t = field_visual[inds]
    field_visual_p = field_visual_p.cpu().numpy()
    input_visual_p = input_visual_p.cpu().numpy()
    ori_input = input_norm.back(input_visual_p[:, 0, 0, :])
    for t in range(len(inds)):
        plt.figure(3, figsize=(20, 10))
        plt.clf()
        Visual.plot_fields_ms(field_visual_t[t], field_visual_p[t], input_visual_p[0, :, :, :2],
                              cmin_max=[[-4, -4], [10, 4]], field_name=['p', 'u', 'v'])
        # plt.suptitle('$t$ = ' + str(ori_input[t, 0]) + ' T', )
        plt.subplots_adjust(wspace=0.2, hspace=0.3)#left=0.05, bottom=0.05, right=0.95, top=0.95
        plt.savefig(os.path.join(vald_path, 'loca_' + str(inds[t]) + '.jpg'))

        plt.figure(4, figsize=(15, 12))
        plt.clf()
        Visual.plot_fields_ms(field_visual_t[t], field_visual_p[t], input_visual_p[0, :, :, :2])
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.savefig(os.path.join(vald_path, 'full_' + str(inds[t]) + '.jpg'))

####################################### plot continous fields #################################################################################
    input_visual_p = torch.tensor(input_visual[:100:5], dtype=torch.float32)
    field_visual_p = inference(input_visual_p.to(device), Net_model)
    field_visual_t = field_visual[:100:5]
    field_visual_p = field_visual_p.cpu().numpy()
    input_visual_p = input_visual_p.cpu().numpy()
    ori_input = input_norm.back(input_visual_p[:, 0, 0, :])
    non_time = (ori_input[:, -1] - ori_input[0, -1]) / (ori_input[-1, -1] - ori_input[0, -1])

    # plot continous fields at points
    ind_xs = np.random.choice(np.arange(0, Nx), 4)
    ind_ys = np.random.choice(np.arange(0, Ny), 4)
    plt.figure(6, figsize=(15, 10))
    plt.clf()
    lims = [[-1, 1], [-0.5, 1.5], [-1, 1]]
    tits = ['p', 'u', 'v']
    for j in range(3):
        plt.subplot(2, 2, j+1)
        plt.ylim(lims[j])
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 20
        for i, (ind_x, ind_y) in enumerate(zip(ind_xs, ind_ys)):
            plt.plot(non_time, field_visual_t[:, ind_x, ind_y, j])
            plt.scatter(non_time, field_visual_p[:, ind_x, ind_y, j])
        plt.legend(['original', 'predicted'])
        plt.grid()
        plt.xlabel('Time t/T')
        plt.ylabel('Physical field of ' + tits[j])
    plt.savefig(os.path.join(vald_path, 'continuous_fields_point.jpg'))

    ####################################### plot dynamic coefficient ############################################################################
    print("plot dynamic coefficient")
    ori_forces = Dyn_model(torch.tensor(field_visual_t[:, :-1], device=device))
    pre_forces = Dyn_model(torch.tensor(field_visual_p[:, :-1], device=device))
    ori_forces = ori_forces.cpu().numpy()
    pre_forces = pre_forces.cpu().numpy()

    plt.figure(6, figsize=(8, 6))
    plt.clf()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    plt.plot(ori_input[:, -1], ori_forces[:, 0], 'r-')
    plt.plot(ori_input[:, -1], pre_forces[:, 0], 'ro')
    plt.plot(ori_input[:, -1], ori_forces[:, 1], 'b-')
    plt.plot(ori_input[:, -1], pre_forces[:, 1], 'bo')
    plt.ylim([-0.5, 1.0])
    plt.xlabel('Time t/T')
    plt.ylabel('Force F/N')
    plt.grid()
    plt.legend(['original lift', 'predicted lift', 'original drag', 'predicted drag'])
    plt.savefig(os.path.join(vald_path, 'forces.jpg'))

    ori_drag_mean = np.mean(ori_forces[:, 1])
    pre_drag_mean = np.mean(pre_forces[:, 1])
    err_drag_mean = np.abs(pre_drag_mean - ori_drag_mean)/ori_drag_mean

    ori_lift_delt = np.max(ori_forces[:, 0]) - np.min(ori_forces[:, 0])
    pre_lift_delt = np.max(pre_forces[:, 0]) - np.min(pre_forces[:, 0])
    err_lift_delt = np.abs(pre_lift_delt - ori_lift_delt)/ori_lift_delt

    print('origin drag mean: {:.3f}, predicted drag mean: {:.3f}, relative error: {:.3f}%, \n'
          'origin lift delta: {:.3f}, predicted lift delta: {:.3f}, relative error: {:.3f}%, \n'.format
          (ori_drag_mean, pre_drag_mean, err_drag_mean*100, ori_lift_delt, pre_lift_delt, err_lift_delt*100,))

####################################### L2 error ##############################################################################
    print("plot fields L2 error ")
    err = field_visual_p - field_visual_t
    err_L2 = np.linalg.norm(err, axis=(1, 2))/np.linalg.norm(field_visual_t[:, :, :, (0, 1, 1)], axis=(1, 2))
    plt.figure(10, figsize=(15, 10))
    plt.clf()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    for i in range(3):
        plt.plot(non_time, err_L2[:, i], '-', linewidth=2.0)
    plt.legend(['p', 'u', 'v'])
    plt.xlabel('Time t/T')
    plt.ylabel('Relative $L_2$ error')
    plt.grid()
    plt.savefig(os.path.join(vald_path, 'L2.jpg'))
    print('Relative L2 error: {:.3f}'.format(np.mean(err_L2),))

####################################### plot fields gif #################################################################################
    # plot continous fields
    print("plot several true and predicted fields gif")
    fig = plt.figure(100, figsize=(15, 10))
    Visual.plot_fields_am(np.concatenate([field_visual_t, field_visual_t[(0,),]], axis=0),
                          np.concatenate([field_visual_p, field_visual_p[(0,),]], axis=0),
                          np.concatenate([input_visual_p, input_visual_p[(0,),]], axis=0)[:, :, :, :2],
                          0, fig, cmin_max=[[-4, -4], [10, 4]],)
