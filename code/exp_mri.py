# -*- coding: utf-8 -*-
"""
Fig.5：MRI 重建（Shepp-Logan 256x256，6 条径向线，无噪声，约束形式）。
方法：FBP（零填充逆 FFT）、L1（TV, split Bregman [15]）、L1-0.5L2（[27]）、L1/L2-grad（Algorithm 3.2）。
另检验 8 条线时三种方法是否都能精确恢复（论文：误差 ~1e-8）。

用法（每个方法可单独运行，结果缓存在 results/mri_<lines>_<method>.npz）：
    python exp_mri.py --lines 6 --method all
    python exp_mri.py --lines 6 --method l1dl2      # 也可逐个：fbp / tv / l1ml2 / l1dl2
    python exp_mri.py --lines 6 --plot              # 生成 figures/fig5_mri.png（Fig.5 的 2x3 版式）
    python exp_mri.py --lines 8 --method all        # 8 条线验证
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mri

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "figures")
N = 256
METHODS = ["fbp", "tv", "l1ml2", "l1dl2"]
LABELS = {"fbp": "FBP", "tv": r"$L_1$", "l1ml2": r"$L_1$-0.5$L_2$", "l1dl2": r"$L_1/L_2$"}


def setup(lines):
    u_true = mri.shepp_logan(N)
    M = mri.radial_mask(N, lines)
    f = mri.fwd(u_true, M)
    return u_true, M, f


def run_method(method, lines, args):
    u_true, M, f = setup(lines)
    t = time.time()
    if method == "fbp":
        u = mri.fbp_zero_filled(f, M); info = {"iters": 0}
    elif method == "tv":
        u, info = mri.tv_grad(f, M, mu=args.lam, lam=args.rho, maxit=args.maxit)
    elif method == "l1ml2":
        u, info = mri.l1_minus_al2_grad(f, M, alpha=0.5, mu=args.lam, lam=args.rho,
                                        max_dca=args.max_dca, max_breg_outer=args.maxit // args.max_dca)
    elif method == "l1dl2":
        u, info = mri.l1dl2_grad(f, M, lam=args.lam, rho1=args.rho, rho2=args.rho, rho3=args.rho,
                                 maxit=args.maxit)
    else:
        raise ValueError(method)
    re = mri.rel_err(u, u_true)
    os.makedirs(RES_DIR, exist_ok=True)
    np.savez(os.path.join(RES_DIR, f"mri_{lines}_{method}.npz"), u=u, re=re, iters=info["iters"],
             ratio=M.mean())
    print(f"lines={lines} {method:6s} RE = {100 * re:.4f}%  iters={info['iters']}  "
          f"采样率={100 * M.mean():.2f}%  用时 {time.time() - t:.1f}s", flush=True)


def plot(lines, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u_true, M, _ = setup(lines)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.6))
    panels = [("(a) Original", u_true), ("(b) Sampling mask", np.fft.fftshift(M))]
    for m in ["fbp", "tv", "l1ml2", "l1dl2"]:
        p = os.path.join(RES_DIR, f"mri_{lines}_{m}.npz")
        if os.path.exists(p):
            d = np.load(p)
            panels.append((f"{LABELS[m]} (RE = {100 * float(d['re']):.2f}%)", d["u"]))
        else:
            panels.append((f"{LABELS[m]} (未运行)", np.zeros_like(u_true)))
    letters = "abcdef"
    for k, (ax, (title, img)) in enumerate(zip(axes.ravel(), panels)):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if k >= 2:
            title = f"({letters[k]}) " + title
        ax.set_xlabel(title, fontsize=12)
    fig.suptitle(f"MRI reconstruction results from {lines} radial lines in the frequency domain "
                 f"({100 * M.mean():.2f}% measurements)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"fig5_mri_{lines}lines.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=6)
    ap.add_argument("--method", default="all", choices=METHODS + ["all"])
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--maxit", type=int, default=3000)
    ap.add_argument("--max-dca", type=int, default=10)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        plot(a.lines, a)
    else:
        for m in (METHODS if a.method == "all" else [a.method]):
            run_method(m, a.lines, a)
