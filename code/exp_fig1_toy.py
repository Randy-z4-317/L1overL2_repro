# -*- coding: utf-8 -*-
"""Fig.1：玩具例子 (2.1)。x(t) = (t, t, t, 20-2t, 40-4t, 2(t-9))，t in [-30, 30]。
绘制 L1、Lp(p=1/2)、TL1(a=1)、L1-L2、L1/L2 五个目标函数随 t 的变化。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
A = np.array([[1, -1, 0, 0, 0, 0], [1, 0, -1, 0, 0, 0], [0, 1, 1, 1, 0, 0],
              [2, 2, 0, 0, 1, 0], [1, 1, 0, 0, 0, -1]], dtype=float)
b = np.array([0, 0, 20, 40, 18], dtype=float)


def x_of_t(t):
    t = np.atleast_1d(t)[:, None]
    return np.hstack([t, t, t, 20 - 2 * t, 40 - 4 * t, 2 * (t - 9)])


def main():
    t = np.union1d(np.linspace(-30, 30, 6001), [0.0, 9.0, 10.0])
    X = x_of_t(t)
    assert np.allclose(X @ A.T, b)          # 检验通解满足 Ax=b
    aX = np.abs(X)
    l1 = aX.sum(1); l2 = np.linalg.norm(X, axis=1)
    funcs = [("(a) $L_1$", l1), ("(b) $L_p$ (p=1/2)", np.sqrt(aX).sum(1)),
             ("(c) TL1", (2 * aX / (1 + aX)).sum(1)), ("(d) $L_1$-$L_2$", l1 - l2),
             ("(e) $L_1/L_2$", l1 / l2)]
    fig = plt.figure(figsize=(15, 8))
    pos = [(0, 0, 3), (0, 3, 3), (1, 0, 2), (1, 2, 2), (1, 4, 2)]
    for (title, f), (r, c, w) in zip(funcs, pos):
        ax = plt.subplot2grid((2, 6), (r, c), colspan=w, fig=fig)
        ax.plot(t, f, "b-", lw=1.2)
        ax.set_xlim(-30, 30)
        ax.set_title(title, y=-0.2)
        k = np.argmin(f)
        print(f"{title:22s} 全局最小 t = {t[k]:.3f}，值 = {f[k]:.4f}")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", "fig1_toy.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
