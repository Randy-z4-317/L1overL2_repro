# -*- coding: utf-8 -*-
"""
Fig.2：算法收敛行为的经验验证。
  左列：L1/L2（Algorithm 3.1，无盒约束），过采样 DCT，F=10，s=12；||x-y||、||x-z|| 与 ||x||_1/||x||_2 随迭代变化
  右列：L1/L2-grad（Algorithm 3.2，含 [0,1] 盒约束），MRI 7 条径向线；||∇u-d||、||∇u-h|| 与 ||∇u||_1/||∇u||_2
用法：
    python exp_fig2_convergence.py --part left
    python exp_fig2_convergence.py --part right
    python exp_fig2_convergence.py --plot
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_rng, dct_matrix, sparse_signal, l1_linprog, rel_err
import solvers_sparse as S
import mri

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "figures")

# radioL1/L2
def run_left(maxit=3000):
    rng = make_rng(2, 10, 12)
    A = dct_matrix(64, 1024, 10, rng) # m x n
    x = sparse_signal(1024, 12, rng) # n x 1
    b = A @ x # m x 1

    x1 = l1_linprog(A, b) # n x 1
    xr, info = S.l1l2_admm(A, b, x1, rho1=100.0, rho2=100.0, maxit=maxit, record=True,
                           rng=make_rng(2, 10, 12, 99))
    h = info["hist"]
    os.makedirs(RES_DIR, exist_ok=True)
    np.savez(os.path.join(RES_DIR, "fig2_left.npz"), xy=h["xy"], xz=h["xz"], obj=h["obj"])
    print(f"left: iters={info['iters']}  L1 相对误差={rel_err(x1, x):.2e}  L1/L2 相对误差={rel_err(xr, x):.2e}")

# radioL1/L2-grad
def run_right(lines=7, maxit=1500):
    u_true = mri.shepp_logan(256)
    M = mri.radial_mask(256, lines)
    f = mri.fwd(u_true, M) # 对真实数据做傅里叶变换并采样
    u, info = mri.l1dl2_grad(f, M, lam=100.0, rho1=1.0, rho2=1.0, rho3=1.0, maxit=maxit,
                             record=True, u_true=u_true)
    h = info["hist"]
    os.makedirs(RES_DIR, exist_ok=True)
    np.savez(os.path.join(RES_DIR, "fig2_right.npz"), gd=h["gd"], gh=h["gh"], obj=h["obj"], re=h["re"])
    print(f"right: lines={lines} iters={info['iters']}  RE={100 * mri.rel_err(u, u_true):.3f}%")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    pl = os.path.join(RES_DIR, "fig2_left.npz")
    pr = os.path.join(RES_DIR, "fig2_right.npz")
    if os.path.exists(pl):
        d = np.load(pl); k = np.arange(1, len(d["xy"]) + 1)
        ax[0, 0].semilogy(k, d["xy"], label="distance y"); ax[0, 0].semilogy(k, d["xz"], label="distance z")
        ax[0, 0].legend(); ax[0, 0].set_xlabel("iteration")
        ax[1, 0].plot(k, d["obj"]); ax[1, 0].set_xlabel("Iteration")
    ax[0, 0].set_title("(a) Residual errors in L1/L2", y=-0.25)
    ax[1, 0].set_title("(c) Objective functions of L1/L2", y=-0.25)
    if os.path.exists(pr):
        d = np.load(pr); k = np.arange(1, len(d["gd"]) + 1)
        ax[0, 1].semilogy(k, d["gd"], label="distance y"); ax[0, 1].semilogy(k, d["gh"], label="distance z")
        ax[0, 1].legend(); ax[0, 1].set_xlabel("iteration")
        ax[1, 1].plot(k, d["obj"]); ax[1, 1].set_xlabel("iteration")
    # 图例沿用论文原图的写法（distance y / distance z），右列分别对应 ||∇u-d|| 与 ||∇u-h||
    ax[0, 1].set_title("(b) Residual errors in L1/L2-grad", y=-0.25)
    ax[1, 1].set_title("(d) Objective functions of L1/L2-grad", y=-0.25)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig2_convergence.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["left", "right", "all"])
    ap.add_argument("--lines", type=int, default=7)
    ap.add_argument("--maxit", type=int, default=3000)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.part in ("left", "all"):
        run_left(a.maxit)
    if a.part in ("right", "all"):
        run_right(a.lines)
    if a.plot:
        plot()
