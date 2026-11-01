# -*- coding: utf-8 -*-
"""
公共工具：感知矩阵 / 稀疏信号生成、L1 初值（linprog）、各模型目标函数、收缩算子、随机种子。

复现论文：Y. Rahimi, C. Wang, H. Dong, Y. Lou,
          "A scale invariant approach for sparse signal recovery", arXiv:1812.08852v4 (SISC 2019)
"""
import os
import warnings
import numpy as np

BASE_SEED = 20190816          # 全局固定随机种子（所有实验由此派生）


def make_rng(*keys):
    """由 (BASE_SEED, *keys) 派生独立且可复现的随机数发生器；keys 为非负整数。"""
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *[int(k) for k in keys]]))


# ----------------------------------------------------------------------------------------
# 感知矩阵
# ----------------------------------------------------------------------------------------
def dct_matrix(m, n, F, rng):
    """过采样 DCT 矩阵，论文式 (4.1)：
        a_j = cos(2*pi*w*j/F) / sqrt(m),  j = 1..n,
    w ~ U[0,1]^m 为整个矩阵共用的随机向量；F 越大，相干性越高。不做额外的列归一化。"""
    w = rng.random(m)
    j = np.arange(1, n + 1)
    return np.cos(2.0 * np.pi * np.outer(w, j) / F) / np.sqrt(m)


def gaussian_matrix(m, n, r, rng):
    """列相关高斯矩阵：每行 ~ N(0, Sigma)，Sigma = (1-r) I + r 11^T（论文第 4 节 / TL1 论文 [47]）。
    等价构造：row = sqrt(1-r) g + sqrt(r) xi 1，g ~ N(0, I_n)，xi ~ N(0,1)。
    【注意】按约定不做列归一化（论文未提及列归一化）。"""
    return np.sqrt(1.0 - r) * rng.standard_normal((m, n)) + np.sqrt(r) * rng.standard_normal((m, 1))


# ----------------------------------------------------------------------------------------
# 稀疏信号
# ----------------------------------------------------------------------------------------
def sparse_signal(n, s, rng, min_sep=None):
    """s-稀疏信号：随机支撑集，非零元 ~ N(0,1)，再归一化使 max|x_i| = 1（以便检验 [-1,1] 盒约束）。
    min_sep：相邻非零元下标的最小间隔（仅 Fig.6b 使用，MS=40）；None 表示不限制。
    带间隔的支撑集采样：从 {0..n-(s-1)(MS-1)-1} 中无放回取 s 个排序后，第 i 个加上 i*(MS-1)，
    得到在所有合法支撑集上均匀分布的样本。"""
    if min_sep is None or min_sep <= 1:
        supp = rng.choice(n, s, replace=False)
    else:
        L = n - (s - 1) * (min_sep - 1)
        if L < s:
            raise ValueError("稀疏度过大，无法满足最小间隔约束")
        supp = np.sort(rng.choice(L, s, replace=False)) + np.arange(s) * (min_sep - 1)
    x = np.zeros(n)
    x[supp] = rng.standard_normal(s)
    x /= np.max(np.abs(x))
    return x


# ----------------------------------------------------------------------------------------
# 基本算子
# ----------------------------------------------------------------------------------------
def shrink(v, mu):
    """软阈值，式 (3.11)：sign(v) * max(|v| - mu, 0)。"""
    return np.sign(v) * np.maximum(np.abs(v) - mu, 0.0) # maximum逐元素比较大小


def rel_change(x_new, x_old):
    nx = np.linalg.norm(x_new)
    return np.linalg.norm(x_new - x_old) / (nx if nx > 0 else 1.0)


def rel_err(x_rec, x_true):
    return np.linalg.norm(x_rec - x_true) / np.linalg.norm(x_true)


# ----------------------------------------------------------------------------------------
# L1 最小化（初值）：min ||x||_1 s.t. Ax = b
# ----------------------------------------------------------------------------------------
def l1_linprog(A, b):
    """论文使用 Gurobi 线性规划；此处按约定使用 scipy.optimize.linprog（HiGHS）。
    LP：x = u - v，u, v >= 0，min 1^T(u+v) s.t. [A, -A][u; v] = b。

    【仅供无 scipy 环境调试】若设置环境变量 REPRO_L1_FALLBACK=1，则改用 ADMM 基追踪近似求解
    （_l1_admm_fallback），其精度/速度均不如 LP，正式实验请勿使用。"""
    try:
        from scipy.optimize import linprog
    except ImportError:
        if os.environ.get("REPRO_L1_FALLBACK", "0") == "1":
            return _l1_admm_fallback(A, b)
        raise ImportError("需要 scipy（scipy.optimize.linprog）。调试时可设置 REPRO_L1_FALLBACK=1。")
    m, n = A.shape
    res = linprog(np.ones(2 * n), A_eq=np.hstack([A, -A]), b_eq=b,
                  bounds=(0, None), method="highs")
    if res.status != 0:
        warnings.warn("linprog 未正常结束：" + str(res.message))
    return res.x[:n] - res.x[n:]


def _l1_admm_fallback(A, b, rho=None, maxit=20000, tol=1e-12):
    """调试用：min ||z||_1 s.t. x = z, Ax = b 的 ADMM（x 步为仿射投影）。"""
    M = A.T @ np.linalg.inv(A @ A.T)
    proj = lambda f: f - M @ (A @ f - b)
    x = proj(np.zeros(A.shape[1]))
    rho = rho if rho is not None else 10.0 / max(np.abs(x).max(), 1e-12)
    z = x.copy(); u = np.zeros_like(x)
    for _ in range(maxit):
        x = proj(z - u)
        z_old = z
        z = shrink(x + u, 1.0 / rho)
        u += x - z
        if np.linalg.norm(z - z_old) <= tol * max(np.linalg.norm(z), 1e-12) and \
           np.linalg.norm(x - z) <= tol * max(np.linalg.norm(x), 1e-12):
            break
    return x


# ----------------------------------------------------------------------------------------
# 各模型目标函数（用于"模型失败 / 算法失败"判定，均为各模型自身的正则项）
# ----------------------------------------------------------------------------------------
def obj_l1(x):
    return np.sum(np.abs(x))


def obj_lp(x, p=0.5):
    return np.sum(np.abs(x) ** p)


def obj_l1ml2(x):
    return np.sum(np.abs(x)) - np.linalg.norm(x)


def obj_tl1(x, a=1.0):
    ax = np.abs(x)
    return np.sum((a + 1.0) * ax / (a + ax))


def obj_ratio(x):
    n2 = np.linalg.norm(x)
    return np.sum(np.abs(x)) / n2 if n2 > 0 else np.inf # 正无穷 若L2范数为零则比值返回值退化成无穷大


# 【说明】L1/L2-box 的目标函数取 L1/L2 比值本身，不额外加入盒约束指示函数
#         （真值满足 [-1,1]；重建结果 x 由 ADMM 返回，可能有 1e-6 量级的越界）。
OBJECTIVES = {
    "L1": obj_l1,
    "TL1": obj_tl1,
    "Lp": obj_lp,
    "L1-L2": obj_l1ml2,
    "L1/L2": obj_ratio,
    "L1/L2-box": obj_ratio,
}
