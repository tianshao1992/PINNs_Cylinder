# Paddle Paddle Hackathon（第三期 65）
## **1.任务描述**




## 2.代码说明
### 2.1.环境依赖

  > numpy == 1.22.3 \
  > scipy == 1.8.0  \
  > scikit-optimize == 0.9.0 \
  > paddlepaddle-gpu == 2.3.0 \
  > paddle==1.0.2 \
  > matplotlib==3.5.1 \
  > seaborn==0.11.2 


## 3.数据集说明
### 3.1 数值计算方法

该数据集为Re=250时的二维层流圆柱绕流数值计算结果，包括了压力场 *p*、*x*方向速度场 *u* 和*y*方向速度场 *v* 在一个周期内共120个时间切片，该问题的Navier-Stokes控制方程可表示为

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 
$$

$$
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial x} + \frac{\mu}{\rho}(\frac{\partial ^2 u}{\partial x ^2} + \frac{\partial ^2 u}{\partial y ^2})
$$

$$
\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial y} + \frac{\mu}{\rho}(\frac{\partial ^2 v}{\partial x ^2} + \frac{\partial ^2 v}{\partial y ^2})
$$

该圆柱（直径为c=1.0m）绕流问题的计算域以及相应边界条件如下图所示：本模型所文献[1]采用的直径约64c的同心圆形域，入口边界为速度入口（ $u=U_0,v=0$ ）,其中 $U_0=1m/s$ ，出口边界为出流边界，即速度在边界上符合（ $\partial u/ \partial\vec{\bf{n}}=0,\partial v/ \partial\vec{\bf{n}}=0$ ），圆柱表面为无滑移边界条件。在数值计算中采用标准的物理场初始化方法（ $u=U_0,v=0,p=0$ ），采用SIMPLE算法耦合压力-速度项，二阶和二阶迎风算法空间离散压力和动量项，二阶隐式进行时间离散。工质物性设置为 $\rho=1,U_0=1$ ，而 $\mu$ 由Re=250决定，此时，该工况的控制方程可简便的无量纲化，无量纲方法为：

$$
x^*=x/c, y^*=y/c, u^*=u/U_0, v^*=v/U_0, p^*=\frac{p}{\rho c U_0^2}
$$

但是为了方便表示仍采用原本的符号表示，无量纲控制方程为：

$$
eq_1 :\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 
$$

$$
eq_2 :\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \frac{1}{Re}(\frac{\partial ^2 u}{\partial x ^2} + \frac{\partial ^2 u}{\partial y ^2}) = 0
$$

$$
eq_3 :\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} +\frac{\partial p}{\partial y} -\frac{1}{Re}(\frac{\partial ^2 v}{\partial x ^2} + \frac{\partial ^2 v}{\partial y ^2})
$$

边界条件的数学表达式为

| 边界条件 | 公式 |
| :----------------------------------------------------------: | :---------------------------------------------------------: |
| 入口边界    |   $bq_1: u-U_0=0; bq_2: v=0$ |
| 出口边界    |  $bq_3: \frac{\partial u}{\partial\vec{\bf{n}}}=0; bq_4: \frac{\partial v}{\partial\vec{\bf{n}}}=0$ |
| 圆柱壁面边界 |  $bq_5:  u=0; bq_6:  v=0$ |
| 初始边界条件 | $in_1:  {u- u_{initial}}=0;in_2:  v- v_{initial}=0;in_3:  p- p_{initial}=0$ |


![conputational domain](figs%20for%20md/computational%20domain.png)

考虑的气动性能参数包括刚体表面的受力，即y方向上的升力和x方向上的阻力，这些性能参数可以表示为物理场在圆柱表面的积分形式：

$$
\vec{F}(t)= [F_x(t), F_y(t)]^T= \oint{(-p \vec{\bf{n}}+\frac{\mu}{\rho}(\nabla \vec{\bf{u}}+ \nabla \vec{\bf{u}}^T) \cdot \vec{\bf{n}})ds}
$$

对应的升力和阻力系数可表示为：

$$
C_l=\frac{F_x}{\frac{1}{2}\rho c U_0^2}, C_d=\frac{F_y}{\frac{1}{2}\rho c U_0^2}
$$

为了验证圆柱绕流数值计算的准确性，下图展示了升力系数和阻力系数与其他研究的对比。由图可知，在 Re=60～500范围内，本文所采用的数值计算方法得到的系数与文献[2-5]的研究基本吻合，因此，本次所采用的Re = 250 工况结果较为准确。

| 阻力系数曲线 | 升力系数曲线 |
| ---- | -----|
| ![Cd](figs%20for%20md/Cd.JPG) | ![Cl](figs%20for%20md/Cl.JPG) |


## 4.模型描述

本模型采用常规的 Physics informed neural network (PINN) 模型， 以每个采样点的时空坐标为输入，物理场为输出，该模型的数学表达式为

$$
f(t,x,y)= [p,u,v]^T
$$

综合的损失函数为监测点损失、控制方程损失和所有的边界条件损失：

$$
综合损失： L_{tol}= w_1 L_{sup} +  w_2 L_{eq} +  w_3 L_{bq} +  w_4 L_{in}
$$

$$
监督点损失： L_{sup}=\frac{1}{n_{sup}}\sum_{i=1}^{n_{sup}}{sqrt(({\bf{f}}_{sup}^i - \hat{\bf{f}}_{sup}^i)^2)}
$$

$$
控制方程损失： L_{eq}=\frac{1}{n_{eq}}\sum_{i,j=1}^{i,j=n_{eq},3}{sqrt({{\bf{eq}}_{j}^{i}}^2)}
$$

$$
边界条件损失： L_{bq}=\frac{1}{n_{bq}}\sum_{i,j=1}^{i,j=n_{bq},6}{sqrt({{\bf{bq}}_{j}^{i}}^2)}
$$

$$
初始条件损失： L_{in}=\frac{1}{n_{in}}\sum_{i,j=1}^{i,j=n_{in},3}{sqrt({{\bf{in}}_{j}^{i}}^2)}
$$

需要注意的是，由于该数据集仅采用了稳定后的一个周期内的物理场，因此初始边界损失中的物理场应为周期中初始时刻的物理场。

**模型的详细参数和训练方法总结如下**：
* 本模型共采用6层全连接层，每层各由64个神经元
* 输入和输出分别采用最大最小方法进行归一化：  $\hat y=\frac{y-min(y)}{max(y)-min(y)}$
* 损失函数权重固定为 $w_1=1.0, w_2=1.0, w_3=1.0, w_4=1.0$
* 采用Adam优化器，初始学习率为0.001
* 共训练400,000个迭代步，学习率分别在300,000和350,000步时衰减为之前的0.1

**各损失函数的采样点总结如下**：
* 控制方程的采样点直接采用网格节点

  训练过程中每步计算随机选取10个时间步上各30000个网格节点进行控制方程损失计算。

* 边界条件采样点

  训练过程中每步计算随机选取20个时间步上进口、出口以及圆柱壁面的网格节点进行边界条件损失计算。

* 初始条件采样点

  训练过程中选取初始时刻所有网格节点进行初始条件损失计算。

* 监测点采样：

  本模型监测点布置于尾迹区域、来流区域以及圆柱壁面圆周，其中圆柱壁面圆周监测点为靠近壁面的第一个网格节点。设置了10组不同个数监测点进行学习对比，具体监测点布置个数及位置总结如下：

| ![](figs%20for%20md/192+48+32.JPG)**192+48+32（尾迹区+来流区+圆柱壁面)** | ![](figs%20for%20md/240+32.JPG)**240+32（尾迹区+圆柱壁面)**  |
| :----------------------------------------------------------: | :---------------------------------------------------------: |
| ![](figs%20for%20md/96+24+16.JPG)**96+24+16（尾迹区+来流区+圆柱壁面)** | ![](figs%20for%20md/120+16.JPG)**120+16（尾迹区+圆柱壁面)** |
| ![](figs%20for%20md/48+12+8.JPG)**48+12+8（尾迹区+来流区+圆柱壁面)** |   ![](figs%20for%20md/60+8.JPG)**60+8（尾迹区+圆柱壁面)**   |
| ![](figs%20for%20md/24+6+4.JPG)**24+6+4（尾迹区+来流区+圆柱壁面)** |   ![](figs%20for%20md/30+4.JPG)**30+4（尾迹区+圆柱壁面)**   |
|   ![](figs%20for%20md/12+4.JPG)**12+4（尾迹区+圆柱壁面)**    |        ![](figs%20for%20md/4.JPG)**4（圆柱壁面)**         |


## 5.结果

### 5.1 监测点位置及数量
在本模型中，




|                                  | 尾迹区+来流区+圆柱壁面 (196+48+32)  | 尾迹区+圆柱壁面 (240+32) |
| -------------------------------- | -------------------------------- | ---------------------------- |
| 预测物理场                         | ![Cd](figs%20for%20md/196+48+32/0.gif) | ![Cd](figs%20for%20md/240+32/0.gif) |
| 相对 $L_2$ 误差                    | ![Cd](figs%20for%20md/196+48+32/L2.jpg) | ![Cd](figs%20for%20md/240+32/L2.jpg) |
| 升力 $C_l$ & 阻力 $C_d$            | ![Cd](figs%20for%20md/196+48+32/forces.jpg) | ![Cd](figs%20for%20md/240+32/forces.jpg) |




|                                  | 尾迹区+来流区+圆柱壁面 (24+6+4)  | 尾迹区+圆柱壁面 (30+4) |
| -------------------------------- | -------------------------------- | ---------------------------- |
| 预测物理场                         | ![Cd](figs%20for%20md/24+6+4/0.gif) | ![Cd](figs%20for%20md/30+4/0.gif) |
| 相对 $L_2$ 误差                    | ![Cd](figs%20for%20md/24+6+4/L2.jpg) | ![Cd](figs%20for%20md/30+4/L2.jpg) |
| 升力 $C_l$ & 阻力 $C_d$            | ![Cd](figs%20for%20md/24+6+4/forces.jpg) | ![Cd](figs%20for%20md/30+4/forces.jpg) |




### 5.2目标工况（少量监测点）


|                                  | 尾迹区+来流区+圆柱壁面 (24+6+4)  | 尾迹区+圆柱壁面 (12+4) |
| -------------------------------- | -------------------------------- | ---------------------------- |
| 预测物理场                         | ![Cd](figs%20for%20md/24+6+4/0.gif) | ![Cd](figs%20for%20md/12+4/0.gif) |
| 相对 $L_2$ 误差                    | ![Cd](figs%20for%20md/24+6+4/L2.jpg) | ![Cd](figs%20for%20md/12+4/L2.jpg) |
| 升力 $C_l$ & 阻力 $C_d$            | ![Cd](figs%20for%20md/24+6+4/forces.jpg) | ![Cd](figs%20for%20md/12+4/forces.jpg) |


### 5.3 极端情况（极少圆柱壁面监测点）

|                                  | 圆柱壁面 (4)  | 
| -------------------------------- | -------------------------------- |
| 预测物理场                         | ![Cd](figs%20for%20md/4/0.gif) | 
| 相对 $L_2$ 误差                    | ![Cd](figs%20for%20md/4/L2.jpg) | 
| 升力 $C_l$ & 阻力 $C_d$            | ![Cd](figs%20for%20md/4/forces.jpg) |




[1]: R. Franke, W. Rodi, and B. Schönung, “Numerical calculation of laminar vortex-shedding flow past cylinders,” _J. Wind Eng. Ind. Aerodyn._, vol. 35, pp. 237–257, Jan. 1990, doi: 10.1016/0167-6105(90)90219-3

[2]: R. D. Henderson, “Details of the drag curve near the onset of vortex shedding,” _Phys. Fluids_, vol. 7, no. 9, pp. 2102–2104, 1995, doi: 10.1063/1.868459.

[3]: C. Y. Wen, C. L. Yeh, M. J. Wang, and C. Y. Lin, “On the drag of two-dimensional flow about a circular cylinder,” _Phys. Fluids_, vol. 16, no. 10, pp. 3828–3831, 2004, doi: 10.1063/1.1789071.

[4]:   O. Posdziech and R. Grundmann, “A systematic approach to the numerical calculation of fundamental quantities of the two-dimensional flow over a circular cylinder,” _J. Fluids Struct._, vol. 23, no. 3, pp. 479–499, 2007, doi: 10.1016/j.jfluidstructs.2006.09.004.

[5]:   J. Park and H. Choi, “Numerical solutions of flow past a circular cylinder at reynolds numbers up to 160,” _KSME Int. J._, vol. 12, no. 6, pp. 1200–1205, 1998, doi: 10.1007/BF02942594.

