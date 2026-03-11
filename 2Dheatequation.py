#-*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt


def solve_pentadiagonal(n, r, d, u_init, w=1.0, max_iter=100, tol=1e-6):
    """
    使用 SOR 迭代法解五对角线性方程组.
    n: 网格每行的内部节点数
    r: 系数
    d: RHS 右端项
    u_init: 初始猜测值 (通常取上一时刻的值)
    """
    u = np.array(u_init)
    N = len(d)
    coeff_main = 4 * r + 1

    for _ in range(max_iter):
        err = 0.0
        for k in range(N):
            row, col = divmod(k, n)

            if col > 0:
                val_l = u[k - 1]
            else:
                val_l = 0.0

            if col < n - 1:
                val_r = u[k + 1]
            else:
                val_r = 0.0

            if row > 0:
                val_u = u[k - n]
            else:
                val_u = 0.0

            if row < n - 1:
                val_d = u[k + n]
            else:
                val_d = 0.0

            sigma = val_l + val_r + val_u + val_d
            u_new = (d[k] + r * sigma) / coeff_main

            diff = u_new - u[k]
            if abs(diff) > err: err = abs(diff)
            u[k] += w * diff

        if err < tol: break
    return u


def crank_nicolson_step_2d(u, n, dx, dt):
    """
    给定当前时刻的 u (一维展开), 做一步 2D Crank–Nicolson 推进.
    """
    r = dt / (dx ** 2)
    N = len(u)

    # 计算 RHS: d = (2E - A_old) * u = (1-4r)u + r*(neighbors)
    d = np.zeros(N)
    for k in range(N):
        row, col = divmod(k, n)

        # 内部点贡献
        term = (1 - 4 * r) * u[k]

        if col > 0: term += r * u[k - 1]
        if col < n - 1: term += r * u[k + 1]
        if row > 0: term += r * u[k - n]
        if row < n - 1: term += r * u[k + n]

        d[k] = term

    u_new = solve_pentadiagonal(n, r, d, u, w=1.5)

    return u_new


def analytic_solution_2d(x, y, t):
    """ 2D 解析解 """
    return np.exp(-5 * np.pi ** 2 * t) * np.sin(2 * np.pi * x) * np.cos(np.pi * y)

def run_single_case_2d(nx=51, dt_factor=0.4, T=0.01):
    """
    2D 情况运行函数
    """
    # 网格
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, nx)
    dx = x[1] - x[0]
    dt = dt_factor * dx ** 2
    tsteps = int(T / dt)
    dt = T / tsteps

    n_inner = nx - 2

    # 初始场 (仅内部节点)
    X, Y = np.meshgrid(x[1:-1], y[1:-1])
    u0_flat = analytic_solution_2d(X, Y, 0).flatten()

    u = u0_flat
    num = 0
    while num < tsteps:
        u = crank_nicolson_step_2d(u, n_inner, dx, dt)
        num += 1

    # 重塑为 2D 以便计算误差
    u_numeric_2d = u.reshape((n_inner, n_inner))
    u_exact_2d = analytic_solution_2d(X, Y, T)

    l2_err = np.sqrt(np.sum((u_numeric_2d - u_exact_2d) ** 2) * dx ** 2)
    max_err = np.max(np.abs(u_numeric_2d - u_exact_2d))

    return X, Y, u_numeric_2d, u_exact_2d, l2_err, max_err


if __name__ == "__main__":
    # 2维使用SOR迭代法
    try:
        X, Y, u_num, u_ex, l2, mx = run_single_case_2d(nx=21, dt_factor=0.4, T=0.005)
        print("2D CN: L2 = {:.4e}, max = {:.4e}".format(l2, mx))
    except Exception as e:
        print(e)
