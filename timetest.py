#

import numpy as np
from main import run_single_case
import math

def time_convergence_test(nx=401, dt_list=None, T=0.1):
    if dt_list is None:
        nsteps_list = [10,20,50,100,200, 400, 800, 1600]
        dt_list = [T / n for n in nsteps_list] # 时间步长
    errs = []
    for dt in dt_list:

        x = np.linspace(0,1,nx)
        dx = x[1]-x[0]

        tsteps = int(np.ceil(T/dt)) # 时间步数
        dt_adj = T / tsteps # 依赖于时间步数的时间步长
        _, u, u_ex, l2, mx = run_single_case(nx=nx, dt_factor=dt_adj/(dx**2), T=T)
        errs.append(l2)
        print(f"时间步长约为：{dt_adj:.3e}, L2 误差 = {l2:.5e}")

    pts = []
    for i in range(len(errs)-1):
        E1, E2 = errs[i], errs[i+1]
        dt1 = dt_list[i]; dt2 = dt_list[i+1]
        p = math.log(E1/E2) / math.log(dt1/dt2)
        pts.append(p)
        print(f"时间收敛阶dt[{i}]和dt[{i+1}] ≈ {p:.4f}")
    return dt_list, errs, pts

if __name__ == "__main__":
    dt_list, errs, pts = time_convergence_test(nx=401)