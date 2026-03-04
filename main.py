# -*- coding: utf-8 -*-

# main_skeleton.py  —— 请按顺序实现 TODO
import numpy as np
import math

def thomas(a, b, c, d):
    """
    参数说明：
      a: sub-diagonal 长度 n-1
      b: diagonal 长度 n
      c: super-diagonal 长度 n-1
      d: RHS 长度 n
    要点：注意数值稳定性，避免分母为0。
    """
    n = len(b)
    l = [b[0]]
    u = []
    for i in range(n-1):
        u.append(c[i]/l[i])
        l.append(b[i+1]-a[i]*u[i])
    y = [d[0]/l[0]]

    for i in range(n-1):
        y.append((d[i+1]-y[i]*a[i])/l[i+1])

    x = [0 for i in range(n)]
    x[n-1] = y[n-1]
    for i in range(n-2,-1,-1):
        x[i] = y[i]-u[i]*x[i+1]
    return x


def crank_nicolson_step(u, dx, dt):
    """
    TODO: 给定当前时刻的 u（包含边界），做一步 Crank–Nicolson 时间推进并返回下一时刻的 u_new（包含边界）。
    说明：边界点 u[0] 和 u[-1] 视作 Dirichlet 已知，不更新。
    提示：内部构造三对角系数 a,b,c 和 RHS d，然后调用 thomas 解出内部节点。
    """
    # ====== START TODO ======
    r = dt/pow(dx,2)
    n = len(u)
    a = [-r/2 for _ in range(n-3)]

    b = [1+r for _ in range(n-2)]

    c = [-r/2 for _ in range(n-3)]

    d = [r/2*u[j+1]+(1-r)*u[j]+r/2*u[j-1] for j in range(1,n-1)]
    d[0] = d[0] + r/2*u[0]
    d[-1] = d[-1] + r/2*u[-1]
    u_new = thomas(a,b,c,d)
    u_new.append(u[-1])
    u_new.insert(0,u[0])
    return u_new
    # ====== END TODO ======

def analytic_solution(x, t):
    # 解析解：初值 u(x,0)=sin(pi x)，零 Dirichlet
    return np.sin(np.pi * x) * np.exp(-(np.pi**2) * t)

def run_single_case(nx=101, dt_factor=0.4, T=0.1):
    """
    TODO: 用 crank_nicolson_step 反复推进到时间 T 并返回 (x, u_numeric, u_exact, l2_err, max_err)
    要点：
      - 构造网格 x 在 [0,1] 上 nx 点
      - dx = x[1]-x[0]
      - dt = dt_factor * dx**2, 然后计算 tsteps = ceil(T/dt) 并调整 dt = T/tsteps（使得整步到达 T）
      - 初值 u0 = sin(pi x)
      - 通过循环调用 crank_nicolson_step 得到 u_T
      - 计算 L2 误差近似 sqrt(sum((u-u_ex)**2)*dx) 与 max abs error
    """
    # ====== START TODO ======
    x = np.linspace(0,1,nx)
    dx = x[1] - x[0]
    dt = dt_factor * dx**2
    tsteps = int(T/dt)
    dt = T/tsteps
    u0 =np.sin(np.pi*x)
    num = 0
    u = u0
    while num<tsteps:
        u_new = crank_nicolson_step(u, dx, dt)
        u = u_new
        num += 1
    u_numeric = u
    u_exact = analytic_solution(x,T)
    l2_err = np.sqrt(sum((u_numeric - u_exact) ** 2) * dx)
    max_err = np.max(np.abs(u_numeric - u_exact))
    return (x,u_numeric,u_exact,l2_err,max_err)
    # ====== END TODO ======

# ------------------------------
# 单元测试与验证（你运行本文件查看输出）
# ------------------------------
if __name__ == "__main__":
    # 1) 测试 Thomas：解已知小系统 Ax = d（A 为 tridiag）
    A_n = 5
    a = -1.0 * np.ones(A_n-1)
    b = 2.0 * np.ones(A_n)
    c = -1.0 * np.ones(A_n-1)
    # 令真实解 x_true 全为 1，则 d = A @ x_true
    x_true = np.ones(A_n)
    # 构造 d
    d = np.zeros(A_n)
    d[0] = b[0]*1 + c[0]*1
    for i in range(1, A_n-1):
        d[i] = a[i-1]*1 + b[i]*1 + c[i]*1
    d[-1] = a[-1]*1 + b[-1]*1
    # 运行 thomas 并验证残差
    try:
        x_sol = thomas(a, b, c, d)
        res = np.linalg.norm(x_sol - x_true)
        print("Thomas test: residual norm =", res)
    except NotImplementedError as e:
        print(e)
        print("请先实现 thomas 函数")

    # 2) 测试 Crank-Nicolson 完整流程（若你已实现）
    try:
        x, u_num, u_ex, l2, mx = run_single_case(nx=101, dt_factor=0.4, T=0.1)
        print("CN demo: L2 =", l2, "max =", mx)
        # 展示两点误差范围提示
        print("示例: 若实现正确，L2 应在大约 1e-3 ~ 1e-2 范围（与参数有关）")
    except NotImplementedError as e:
        print(e)
        print("先实现 run_single_case（以及上游函数）再运行完整测试")