import h5py
import numpy as np
import matplotlib.pyplot as plt

data = h5py.File('data\\cyl_Re250.mat', 'r')
nodes = np.array(data['grids_']).squeeze().transpose((3, 2, 1, 0))  # [Nx, Ny, Nf]
nodes = nodes[0]

plt.figure(1)
plt.plot(nodes[:93, -1, 1], nodes[:93, -1, 2], 'k.')
plt.plot(nodes[284:, -1, 1], nodes[284:, -1, 2], 'k.')
plt.plot(nodes[93:284, -1, 1], nodes[93:284, -1, 2], 'b.')
plt.plot(nodes[175:220:15, 0:64:8, 1], nodes[175:220:15, 0:64:8, 2], 'c.') #测点
plt.plot(nodes[14, 0:16:8, 1], nodes[14, 0:16:8, 2], 'c.')
plt.plot(nodes[378:360:-15, 0:16:8, 1], nodes[378:360:-15, 0:16:8, 2], 'c.')
plt.plot(nodes[::96, 1, 1], nodes[::96, 1, 2], 'c.')
plt.savefig('24+6+4.jpg', dpi=500)

plt.figure(2)
plt.plot(nodes[:93, -1, 1], nodes[:93, -1, 2], 'k.')
plt.plot(nodes[284:, -1, 1], nodes[284:, -1, 2], 'k.')
plt.plot(nodes[93:284, -1, 1], nodes[93:284, -1, 2], 'b.')
plt.plot(nodes[175:220:15, 0:80:8, 1], nodes[175:220:15, 0:80:8, 2], 'c.') #测点
plt.plot(nodes[::96, 1, 1], nodes[::96, 1, 2], 'c.')
plt.savefig('30+4.jpg', dpi=500)

plt.figure(3)
plt.plot(nodes[:93, -1, 1], nodes[:93, -1, 2], 'k.')
plt.plot(nodes[284:, -1, 1], nodes[284:, -1, 2], 'k.')
plt.plot(nodes[93:284, -1, 1], nodes[93:284, -1, 2], 'b.')
plt.plot(nodes[175:220:15, 0:64:4, 1], nodes[175:220:15, 0:64:4, 2], 'c.') #测点
plt.plot(nodes[14, 0:32:8, 1], nodes[14, 0:32:8, 2], 'c.')
plt.plot(nodes[378:360:-15, 0:32:8, 1], nodes[378:360:-15, 0:32:8, 2], 'c.')
plt.plot(nodes[::48, 1, 1], nodes[::48, 1, 2], 'c.')
plt.savefig('48+12+8.jpg', dpi=500)

plt.figure(4)
plt.plot(nodes[:93, -1, 1], nodes[:93, -1, 2], 'k.')
plt.plot(nodes[284:, -1, 1], nodes[284:, -1, 2], 'k.')
plt.plot(nodes[93:284, -1, 1], nodes[93:284, -1, 2], 'b.')
plt.plot(nodes[175:220:15, 0:80:4, 1], nodes[175:220:15, 0:80:4, 2], 'c.') #测点
plt.plot(nodes[::48, 1, 1], nodes[::48, 1, 2], 'c.')
plt.savefig('60+8.jpg', dpi=500)

plt.figure(5)
plt.plot(nodes[:93, -1, 1], nodes[:93, -1, 2], 'k.')
plt.plot(nodes[284:, -1, 1], nodes[284:, -1, 2], 'k.')
plt.plot(nodes[93:284, -1, 1], nodes[93:284, -1, 2], 'b.')
plt.plot(nodes[175:220:15, 0:64:2, 1], nodes[175:220:15, 0:64:2, 2], 'c.') #测点
plt.plot(nodes[14, 0:32:4, 1], nodes[14, 0:32:4, 2], 'c.')
plt.plot(nodes[378:360:-15, 0:32:4, 1], nodes[378:360:-15, 0:32:4, 2], 'c.')
plt.plot(nodes[::24, 1, 1], nodes[::24, 1, 2], 'c.')
plt.savefig('96+24+16.jpg', dpi=500)

plt.figure(6)
plt.plot(nodes[:93, -1, 1], nodes[:93, -1, 2], 'k.')
plt.plot(nodes[284:, -1, 1], nodes[284:, -1, 2], 'k.')
plt.plot(nodes[93:284, -1, 1], nodes[93:284, -1, 2], 'b.')
plt.plot(nodes[175:220:15, 0:80:2, 1], nodes[175:220:15, 0:80:2, 2], 'c.') #测点
plt.plot(nodes[::24, 1, 1], nodes[::24, 1, 2], 'c.')
plt.savefig('120+16.jpg', dpi=500)

plt.figure(7)
plt.plot(nodes[:93, -1, 1], nodes[:93, -1, 2], 'k.')
plt.plot(nodes[284:, -1, 1], nodes[284:, -1, 2], 'k.')
plt.plot(nodes[93:284, -1, 1], nodes[93:284, -1, 2], 'b.')
plt.plot(nodes[175:220:15, 0:128:2, 1], nodes[175:220:15, 0:128:2, 2], 'c.') #测点
plt.plot(nodes[14, 0:64:4, 1], nodes[14, 0:64:4, 2], 'c.')
plt.plot(nodes[378:360:-15, 0:64:4, 1], nodes[378:360:-15, 0:64:4, 2], 'c.')
plt.plot(nodes[::12, 1, 1], nodes[::12, 1, 2], 'c.')
plt.savefig('192+48+32.jpg', dpi=500)

plt.figure(8)
plt.plot(nodes[:93, -1, 1], nodes[:93, -1, 2], 'k.')
plt.plot(nodes[284:, -1, 1], nodes[284:, -1, 2], 'k.')
plt.plot(nodes[93:284, -1, 1], nodes[93:284, -1, 2], 'b.')
plt.plot(nodes[175:220:15, 0:160:2, 1], nodes[175:220:15, 0:160:2, 2], 'c.') #测点
plt.plot(nodes[::12, 1, 1], nodes[::12, 1, 2], 'c.')
plt.savefig('240+32.jpg', dpi=500)

plt.figure(9)
plt.plot(nodes[:93, -1, 1], nodes[:93, -1, 2], 'k.')
plt.plot(nodes[284:, -1, 1], nodes[284:, -1, 2], 'k.')
plt.plot(nodes[93:284, -1, 1], nodes[93:284, -1, 2], 'b.')
plt.plot(nodes[::30, 1, 1], nodes[::30, 1, 2], 'c.')
plt.savefig('13.jpg', dpi=500)
