import numpy as np  # 导入NumPy库，用于数值计算
from scipy.special import lpmv  # 引入勒让德函数
import dedalus.public as d3  # 导入Dedalus库，用于解决偏微分方程
import logging  # 导入日志库
logger = logging.getLogger(__name__)  # 创建一个日志记录器

# Simulation units
meter = 1 / 6.37122e6  # 定义单位米，这里的值是地球半径的倒数
hour = 1  # 定义单位小时
second = hour / 3600  # 定义单位秒，1小时等于3600秒

# Parameters
Nphi = 256  # 定义经度方向的网格点数
Ntheta = 128  # 定义纬度方向的网格点数
dealias = 3/2  # 定义去混叠因子，用于在谱方法中去除混叠误差
R = 6.37122e6 * meter  # 定义地球半径
Omega = 7.292e-5 / second  # 定义地球自转角速度
nu = 1e5 * meter**2 / second / 32**2 # 定义超扩散系数，这里的值是在ell=32处匹配的
g = 9.80616 * meter / second**2  # 定义重力加速度
H = 1e4 * meter  # 定义大气厚度
timestep = 600 * second  # 定义模拟的时间步长
stop_sim_time = 360 * hour  # 定义模拟的停止时间
dtype = np.float64  # 定义数据类型为64位浮点数

# Bases
coords = d3.S2Coordinates('phi', 'theta')  # 定义球面坐标系，坐标变量为经度phi和纬度theta
dist = d3.Distributor(coords, dtype=dtype)  # 创建一个分布器，用于在多个处理器之间分配数据
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)  # 创建一个球面基函数，用于在球面上进行谱分析

# Fields
u = dist.VectorField(coords, name='u', bases=basis)  # 创建一个矢量场u，用于表示风速
h = dist.Field(name='h', bases=basis)  # 创建一个标量场h，用于表示大气厚度
V_tid = dist.Field(name='V_tid', bases=basis)  # 创建一个标量场V_tid，用于表示潮汐势

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))  # 定义一个函数zcross，用于计算A的偏斜和余弦的乘积

# Initial conditions: zonal jet
phi, theta = dist.local_grids(basis)  # 获取本地的经度和纬度网格
lat = np.pi / 2 - theta + 0*phi  # 计算纬度
umax = 0 * meter / second  # 定义最大风速
lat0 = np.pi / 7  # 定义纬度的下限
lat1 = np.pi / 2 - lat0  # 定义纬度的上限
en = np.exp(-4 / (lat1 - lat0)**2)  # 计算指数函数的值
jet = (lat0 <= lat) * (lat <= lat1)  # 创建一个布尔数组，表示纬度是否在lat0和lat1之间
u_jet = umax / en * np.exp(1 / (lat[jet] - lat0) / (lat[jet] - lat1))  # 计算风速的初始值
u['g'][0][jet]  = u_jet  # 将计算得到的风速赋值给u的第0个分量

# Initial conditions: balanced height
c = dist.Field(name='c')  # 创建一个新的标量场c
problem = d3.LBVP([h, c], namespace=locals())  # 创建一个线性边界值问题，变量为h和c
problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")  # 添加一个方程，描述h和c的关系
problem.add_equation("ave(h) = 0")  # 添加一个方程，要求h的平均值为0
solver = problem.build_solver()  # 构建一个求解器，用于求解这个问题
solver.solve()  # 求解这个问题

# Initial conditions: perturbation
lat2 = np.pi / 4  # 定义纬度的中心值
hpert = 0 * meter  # 定义扰动的幅度
alpha = 1 / 3  # 定义扰动在经度方向的宽度
beta = 1 / 15  # 定义扰动在纬度方向的宽度
h['g'] += hpert * np.cos(lat) * np.exp(-(phi/alpha)**2) * np.exp(-((lat2-lat)/beta)**2)  # 在h的初始值上添加一个扰动

# Tidal potential
theta_m = np.pi / 2  # 计算 theta_m 的值
phi_m = 0  # 计算 phi_m 的值
V_tid['g'] = 0  # 初始化场值为零
for n in range(2, 40):
    for m in range(0, n+1):
        A = 1 if m == 0 else 2  # 根据 m 的值选择 A 的值
        N_nm = ((-1)**m * ((2*n+1)/(4*np.pi) * np.math.factorial(n-m) / np.math.factorial(n+m))**0.5)  # 计算 N_n^m 的值
        P_nm = lpmv(m, n, np.cos(theta))  # 使用 theta 计算 P_n^m 的值
        P_nm_m = lpmv(m, n, np.cos(theta_m))  # 使用 theta_m 计算 P_n^m 的值
        K_n = R * 1/81 * (6.37122e6/384403000)**(n+1)  # 计算 K_n 的值
        a_nm = (-1)**m * A * 4*np.pi/(2*n+1) * K_n * N_nm * P_nm_m * np.sin(m * phi_m)  # 使用 phi_m 计算 b_nm 的值
        b_nm = (-1)**m * A * 4*np.pi/(2*n+1) * K_n * N_nm * P_nm_m * np.sin(m * phi_m)  # 使用 phi_m 计算 b_nm 的值
        V_tid['g'] += N_nm * P_nm * (a_nm * np.cos(m * phi) + b_nm * np.sin(m * phi))  # 更新场值


# Problem
problem = d3.IVP([u, h, V_tid], namespace=locals())  # 创建一个初始值问题，变量为u和h
problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h) + g*grad(V_tid) + 2*Omega*zcross(u)  = - u@grad(u)")  # 添加一个方程，描述u的时间演化
problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")  # 添加一个方程，描述h的时间演化
problem.add_equation("dt(V_tid) = 0")  # 添加一个方程，描述V_tid的时间演化

# Solver
solver = problem.build_solver(d3.RK222)  # 使用二阶Runge-Kutta方法构建求解器
solver.stop_sim_time = stop_sim_time  # 设置模拟的停止时间

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots_test', sim_dt=1*hour, max_writes=10)  # 添加一个文件处理器，用于保存模拟的快照，每小时保存一次，最多保存10次
snapshots.add_task(h, name='height')  # 添加一个任务，保存大气厚度h的值
snapshots.add_task(V_tid, name='V_tid')  # 添加一个任务，保存潮汐势V_tid的值
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')  # 添加一个任务，保存涡度的值，涡度定义为u的偏斜的散度的负值

# Main loop
try:  # 尝试执行以下代码
    logger.info('Starting main loop')  # 记录一条信息，表示主循环开始
    while solver.proceed:  # 当求解器可以继续进行时
        V_tid['g'] = 0
        for ntheta in range(0, 64):
            for nphi in range(0, 128):
                thetax = ntheta * np.pi / Ntheta
                phix = nphi * 2 * np.pi / Nphi
                for n in range(2, 20):
                    for m in range(0, n+1):
                        A = 1 if m == 0 else 2
                        N_nm = ((-1)**m * ((2*n+1)/(4*np.pi) * np.math.factorial(n-m) / np.math.factorial(n+m))**0.5)
                        P_nm = lpmv(m, n, np.cos(thetax))
                        P_nm_m = lpmv(m, n, np.cos(0.25 * np.pi - 0.6 * m * solver.iteration * Omega))
                        K_n = R * 1/81 * (6.37122e6/384403000)**(n+1)
                        a_nm = (-1)**m * A * 4*np.pi/(2*n+1) * K_n * N_nm * P_nm_m * np.sin(0.8 * m * solver.iteration * Omega)
                        b_nm = (-1)**m * A * 4*np.pi/(2*n+1) * K_n * N_nm * P_nm_m * np.sin(0.8 * m * solver.iteration * Omega)
                        if ntheta < Ntheta and nphi < Nphi:  # 确保索引不会超出范围
                            V_tid['g'][ntheta, nphi] += N_nm * P_nm * (a_nm * np.cos(m*phix) + b_nm * np.sin(m*phix)) * 1000
        solver.step(timestep)  # 让求解器前进一个时间步长
        if (solver.iteration-1) % 10 == 0:  # 如果当前的迭代次数减1后可以被10整除
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))  # 记录一条信息，包含当前的迭代次数、模拟时间和时间步长
except:  # 如果上面的代码出现异常
    logger.error('Exception raised, triggering end of main loop.')  # 记录一条错误信息，表示异常被抛出，主循环结束
    raise  # 抛出异常
finally:  # 无论上面的代码是否出现异常，都会执行以下代码
    solver.log_stats()  # 记录求解器的统计信息

