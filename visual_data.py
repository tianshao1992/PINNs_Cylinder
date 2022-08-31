import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sbn
from scipy import stats
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri
import matplotlib.cm as cm
from matplotlib import ticker, rcParams
from matplotlib.ticker import MultipleLocator

import sys


class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class matplotlib_vision(object):

    def __init__(self, log_dir, input_name=('x'), field_name=('f',)):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        # sbn.set_style('ticks')
        # sbn.set()

        self.field_name = field_name
        self.input_name = input_name
        self._cbs = [None] * len(self.field_name) * 3

        gs = gridspec.GridSpec(1, 1)
        gs.update(top=0.95, bottom=0.07, left=0.1, right=0.9, wspace=0.5, hspace=0.7)
        gs_dict = {key: value for key, value in gs.__dict__.items() if key in gs._AllowedKeys}
        self.fig, self.axes = plt.subplots(len(self.field_name), 3, gridspec_kw=gs_dict, num=100, figsize=(30, 20))
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30}

    def plot_loss(self, x, y, label, title=None, color=None):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)

        plt.plot(x, y, label=label, color=color)
        plt.semilogy()
        plt.grid(True)  # 添加网格
        plt.legend(loc="upper right", prop=self.font)
        plt.xlabel('iterations', self.font)
        plt.ylabel('loss value', self.font)
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.title(title, self.font)
        # plt.pause(0.001)

    def plot_value(self, x, y, label, title=None):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)

        plt.plot(x, y, label=label)
        plt.grid(True)  # 添加网格
        plt.legend(loc="upper right", prop=self.font)
        plt.xlabel('iterations', self.font)
        plt.ylabel('pred value', self.font)
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.title(title, self.font)
        # plt.pause(0.001)

    def plot_scatter(self, true, pred, axis=0, title=None):
        # sbn.set(color_codes=True)

        plt.scatter(np.arange(true.shape[0]), true, marker='*')
        plt.scatter(np.arange(true.shape[0]), pred, marker='.')

        plt.ylabel('target value', self.font)
        plt.xlabel('samples', self.font)
        plt.xticks(fontproperties='Times New Roman', size=25)
        plt.yticks(fontproperties='Times New Roman', size=25)
        plt.grid(True)  # 添加网格
        plt.title(title, self.font)


    def plot_fields_tri(self, out_true, out_pred, coord, cell, cmin_max=None, fmin_max=None, cmap='jet', field_name=None):

        plt.clf()
        Num_fields = out_true.shape[-1]
        if fmin_max == None:
            fmin, fmax = out_true.min(axis=(0,)), out_true.max(axis=(0,))
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if cmin_max == None:
            cmin, cmax = coord.min(axis=(0, 1)), coord.max(axis=(0, 1))
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        if field_name == None:
            field_name = self.field_name

        x_pos = coord[:, 0]
        y_pos = coord[:, 1]
        ############################# Plotting ###############################
        for fi in range(Num_fields):
            plt.rcParams['font.size'] = 20
            triObj = tri.Triangulation(x_pos, y_pos, triangles=cell)  # 生成指定拓扑结构的三角形剖分.

            Num_levels = 20
            # plt.triplot(triObj, lw=0.5, color='white')

            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            ########      Exact f(t,x,y)     ###########
            plt.subplot(3, Num_fields, 0 * Num_fields + fi + 1)
            levels = np.arange(out_true.min(), out_true.max(), 0.05)
            plt.tricontourf(triObj, out_true[:, fi], Num_levels, cmap=cmap)
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            plt.rcParams['font.family'] = 'Times New Roman'
            # cb.set_label('value', rotation=0, fontdict=self.font, y=1.08)
            plt.rcParams['font.size'] = 20
            # plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('True field $' + field_name[fi] + '$' + '', fontsize=30)

            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            ########     Learned f(t,x,y)     ###########
            plt.subplot(3, Num_fields, 1 * Num_fields + fi + 1)
            # levels = np.arange(out_true.min(), out_true.max(), 0.05)
            plt.tricontourf(triObj, out_pred[:, fi], Num_levels, cmap=cmap)
            cb = plt.colorbar()
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 20
            # plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('Pred field $' + field_name[fi] + '$' + '', fontsize=30)

            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            ########     Error f(t,x,y)     ###########
            plt.subplot(3, Num_fields, 2 * Num_fields + fi + 1)
            err = out_pred[:, fi] - out_true[:, fi]
            plt.tricontourf(triObj, err, Num_levels, cmap='coolwarm')
            cb = plt.colorbar()
            plt.clim(vmin=-max(abs(fmin[fi]), abs(fmax[fi])), vmax=max(abs(fmin[fi]), abs(fmax[fi])))
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 20
            plt.xlabel('$' + self.input_name[0] + '$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('field error$' + field_name[fi] + '$' + '', fontsize=30)

    def plot_fields_ms(self, out_true, out_pred, coord, cmin_max=None, fmin_max=None, field_name=None):

        plt.clf()
        Num_fields = out_true.shape[-1]
        if fmin_max == None:
            fmin, fmax = out_true.min(axis=(0, 1)), out_true.max(axis=(0, 1))
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if cmin_max == None:
            cmin, cmax = coord.min(axis=(0, 1)), coord.max(axis=(0, 1))
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        if field_name == None:
            field_name = self.field_name

        x_pos = coord[:, :, 0]
        y_pos = coord[:, :, 1]
        ############################# Plotting ###############################
        for fi in range(Num_fields):
            plt.rcParams['font.size'] = 20

            ########      Exact f(t,x,y)     ###########
            plt.subplot(3, Num_fields, 0 * Num_fields + fi + 1)
            # plt.axis('equal')
            f_true = out_true[:, :, fi]
            plt.pcolormesh(x_pos, y_pos, f_true, cmap='RdYlBu_r', shading='gouraud', antialiased=True, snap=True)

            # plt.contourf(x_pos, y_pos, f_true, cmap='jet',)
            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            tick_locator = ticker.MaxNLocator(nbins=3)  # colorbar上的刻度值个数
            cb.locator = tick_locator
            cb.update_ticks()
            plt.rcParams['font.family'] = 'Times New Roman'
            # cb.set_label('value', rotation=0, fontdict=self.font, y=1.08)
            plt.rcParams['font.size'] = 20
            # plt.xlabel('$x$', fontdict=self.font)
            # plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('Original field $' + field_name[fi] + '$' + '', fontdict=self.font)


            ########     Learned f(t,x,y)     ###########
            plt.subplot(3, Num_fields, 1 * Num_fields + fi + 1)
            # plt.axis('equal')
            f_pred = out_pred[:, :, fi]
            plt.pcolormesh(x_pos, y_pos, f_pred, cmap='RdYlBu_r', shading='gouraud', antialiased=True, snap=True)
            # plt.contourf(x_pos, y_pos, f_pred, cmap='jet',)
            cb = plt.colorbar()
            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            tick_locator = ticker.MaxNLocator(nbins=3)  # colorbar上的刻度值个数
            cb.locator = tick_locator
            cb.update_ticks()
            plt.rcParams['font.size'] = 20
            # plt.xlabel('$x$', fontdict=self.font)
            # plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('Predicted field $' + field_name[fi] + '$' + '', fontdict=self.font)

            ########     Error f(t,x,y)     ###########
            plt.subplot(3, Num_fields, 2 * Num_fields + fi + 1)
            # plt.axis('equal')
            err = f_true - f_pred
            plt.pcolormesh(x_pos, y_pos, err, cmap='coolwarm', shading='gouraud', antialiased=True, snap=True)
            # plt.contourf(x_pos, y_pos, err, cmap='coolwarm', )
            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            cb = plt.colorbar()
            err_bar = np.abs(err).max()
            plt.clim(vmin=-err_bar, vmax=err_bar)

            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 20
            # plt.xlabel('$' + self.input_name[0] + '$', fontdict=self.font)
            # plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('field error $\it{' + field_name[fi] + '}$' + '', fontdict=self.font)

    def plot_fields_am(self, out_true, out_pred, coord, p_id, fig, cmin_max=None):

        fmax = out_true.max(axis=(0, 1, 2))  # 云图标尺
        fmin = out_true.min(axis=(0, 1, 2))  # 云图标尺

        def anim_update(t_id):
            print('para:   ' + str(p_id) + ',   time:   ' + str(t_id))
            axes = self.plot_fields_ms(out_true[t_id], out_pred[t_id], coord[t_id], cmin_max=cmin_max, fmin_max=(fmin, fmax))
            return axes

        anim = FuncAnimation(fig, anim_update,
                             frames=np.arange(0, out_true.shape[0]).astype(np.int64), interval=200)

        anim.save(self.log_dir + "\\" + str(p_id) + ".gif", writer='pillow', dpi=300)
