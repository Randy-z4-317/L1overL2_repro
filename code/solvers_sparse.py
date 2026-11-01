# -*- coding: utf-8 -*-
"""
稀疏信号恢复的各求解器（约束形式，Ax = b）：
    l1l2_admm      : L1/L2 及 L1/L2-box，论文 Algorithm 3.1 / 式 (3.15)
    lp_chartrand   : Lp (p=0.5)，按 Chartrand [9] 原文的投影梯度法
    dca_l1ml2      : L1-L2，DCA + 内层 ADMM（Lou et al. [26] Algorithm 2）
    dca_tl1        : TL1 (a=1)，DCA + 内层 ADMM（Zhang & Xin [47] Algorithm 3）
所有非凸方法均以 L1 解为初值（论文 4.2 节）。
"""
import numpy as np
from common import shrink, rel_change, obj_ratio


class AffineProjector:
    """到 {x : Ax = b} 的正交投影：P(f) = f - A^T (AA^T)^{-1} (A f - b)。
    预计算 M = A^T (AA^T)^{-1}（n x m），每次投影仅需 O(mn)，与论文 Remark 1 的预计算思想一致。"""

    def __init__(self, A, b):
        self.A, self.b = A, b
        self.M = A.T @ np.linalg.inv(A @ A.T)

    def __call__(self, f):
        return f - self.M @ (self.A @ f - self.b)

    def null(self, d):
        """投影到 ker(A)：(I - A^T(AA^T)^{-1}A) d。"""
        return d - self.M @ (self.A @ d)


# ========================================================================================
# L1/L2 (+box) via ADMM —— Algorithm 3.1
# ========================================================================================
def _tau_root(D):
    """tau^3 - tau^2 - D = 0 的唯一实根，闭式解 (3.8)。"""
    q = 27.0 * D + 2.0
    C = np.cbrt((q + np.sqrt(q * q - 4.0)) / 2.0)
    return 1.0 / 3.0 + (C + 1.0 / C) / 3.0


def l1l2_admm(A, b, x0, rho1=100.0, rho2=100.0, box=None, maxit=3000, tol=1e-8, min_iter=50,
              rng=None, record=False, proj=None):
    """min ||z||_1/||y||_2 + I_0(Ax-b)  s.t. x = y, x = z  （式 (3.3)），可选 z in [c,d]（式 (3.13)）。

    参数
      x0     : 初值（L1 解）；y, z 初始化为 x0，对偶变量 v, w 初始化为 0（论文未说明，此为约定）
      box    : None 或 (c, d)；L1/L2-box 取 (-1, 1)
      min_iter : 迭代下限 50，前 50 次不做停止判断（避免迭代初期的停滞平台被误判为收敛）
      maxit  : 最大迭代 3000（按约定）；tol：||x^k - x^{k-1}|| / ||x^k|| < 1e-8 停止
      rng    : d^(k)=0 时生成随机向量 e^(k) 用的随机数发生器（固定种子）
      record : 记录 ||x-y||, ||x-z||, ||x||_1/||x||_2（Fig.2 用）
      proj   : 投影到可行集{x:Ax=b}的算子，解决ADMM的x子问题时将某个中间量f投影回满足Ax=b的超平面
    """
    n = A.shape[1]
    P = proj if proj is not None else AffineProjector(A, b)
    rng = rng if rng is not None else np.random.default_rng(0)
    x = P(np.asarray(x0, dtype=float).copy())
    y = x.copy(); 
    z = x.copy()
    v = np.zeros(n); 
    w = np.zeros(n)
    hist = {"xy": [], "xz": [], "obj": []} if record else None
    s = rho1 + rho2
    it = 0
    for it in range(1, maxit + 1):
        x_old = x

        # ---- x 步：投影到 Ax=b，f 见 (3.6)
        f = (rho1 * (y - v / rho1) + rho2 * (z - w / rho2)) / s
        x = P(f) 

        # ---- y 步：(3.7)-(3.9)
        c = np.sum(np.abs(z))
        d = x + v / rho1
        eta = np.linalg.norm(d)
        if eta == 0.0:  # d = 0
            e = rng.standard_normal(n)
            y = (e / np.linalg.norm(e)) * np.cbrt(c / rho1) # 归一化 x 模长
        elif c == 0.0:  # c = 0
            y = d
        else:           # d != 0, c != 0
            y = _tau_root(c / (rho1 * eta ** 3)) * d

        # ---- z 步：(3.10)；带盒约束时为 shrink 后截断 (3.15)
        z = shrink(x + w / rho2, 1.0 / (rho2 * np.linalg.norm(y)))
        if box is not None:
            z = np.minimum(np.maximum(z, box[0]), box[1])

        # ---- 对偶更新
        v = v + rho1 * (x - y)
        w = w + rho2 * (x - z)
        
        if record:
            hist["xy"].append(np.linalg.norm(x - y))
            hist["xz"].append(np.linalg.norm(x - z))
            hist["obj"].append(obj_ratio(x))
        # 论文写作 "while k < Max or ..."，此处按约定理解为"任一条件满足即停止"。
        # 迭代下限 min_iter=50：前 50 次不判停（例如第 1 次迭代时 y=z=x0、v=w=0，x 步必然返回 x0，
        # 相对变化恒为 0）。
        if it >= min_iter and rel_change(x, x_old) < tol:
            break
    info = {"iters": it}
    if record:
        info["hist"] = {k: np.array(vv) for k, vv in hist.items()}
    return x, info


# ========================================================================================
# Lp —— Chartrand [9]
# ========================================================================================
def _exact_line_search(u, pd, p, n_grid=121, n_golden=60):
    """g(t) = sum |u + t*pd|^p 在 t>=0 上的"精确"线搜索：
    先在对数网格 t = T0*10^[-10, 2] 上粗搜，再在最优点相邻区间内用黄金分割细化。
    （[9] 未给出线搜索的具体实现；此处为避免依赖 scipy 的自实现。）"""
    npd = np.linalg.norm(pd)
    if npd == 0.0:
        return 0.0
    g = lambda t: np.sum(np.abs(u + t * pd) ** p)
    T0 = max(np.linalg.norm(u), 1e-12) / npd
    ts = np.concatenate([[0.0], T0 * np.logspace(-10, 2, n_grid)])
    vals = np.array([g(t) for t in ts])
    i = int(np.argmin(vals))
    if i == 0:
        return 0.0
    lo, hi = ts[i - 1], ts[min(i + 1, len(ts) - 1)]
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, bb = lo, hi
    c1, c2 = bb - gr * (bb - a), a + gr * (bb - a)
    g1, g2 = g(c1), g(c2)
    for _ in range(n_golden):
        if g1 < g2:
            bb, c2, g2 = c2, c1, g1
            c1 = bb - gr * (bb - a); g1 = g(c1)
        else:
            a, c1, g1 = c1, c2, g2
            c2 = a + gr * (bb - a); g2 = g(c2)
    cand = [(vals[i], ts[i]), (g1, c1), (g2, c2)]
    return min(cand)[1]


def lp_chartrand(A, b, x0, p=0.5, n_iter=40000, eps0=1e-2, eps_decay=0.95, block=100, proj=None):
    """Lp 最小化（p=0.5），严格按 Chartrand [9] 原文：
        d_n = -u_n * (u_n^2 + eps^2)^{(p-2)/2}          （|u| 用 sqrt(u^2+eps^2) 光滑化）
        u_{n+1} = P_{Au=b}(u_n + t_n d_n)
      * 步长 t_n 每 100 次迭代做一次精确线搜索（最小化 ||P(u + t d)||_p），其余迭代沿用；
      * eps 从 0.01 开始，每 100 次迭代衰减 5%；
      * 【注意】按约定保留 [9] 原文的固定 40000 次迭代（不提前停止，不受 3000 次上限约束），
        最终 eps = 0.01*0.95^400 ≈ 1.2e-11，与 [9] 报告的 1.23e-11 一致。
      * 初值：[9] 原文使用最小 L2 范数解；本复现按论文 4.2 节"所有算法以 L1 解为初值"处理。
      * 【备注】论文 Table 1 中 Lp 平均耗时仅约 0.3 s，与 40000 步投影梯度不符，
        作者实际可能使用的是 IRLS 类 Lp 算法（如 Lai-Xu-Yin [23] / Chartrand-Yin 的迭代重加权最小二乘）。
    """
    P = proj if proj is not None else AffineProjector(A, b)
    u = P(np.asarray(x0, dtype=float).copy())
    eps = eps0
    t = 0.0
    for k in range(n_iter):
        d = -u * (u * u + eps * eps) ** ((p - 2.0) / 2.0)
        pd = P.null(d)                       # u 可行时 P(u + t d) = u + t * P_ker(d)
        if k % block == 0:
            t = _exact_line_search(u, pd, p)
        u = u + t * pd
        if (k + 1) % block == 0:
            eps *= eps_decay
            u = P(u)                         # 周期性重新投影，消除累积的数值漂移
    return P(u), {"iters": n_iter, "eps_final": eps}


# ========================================================================================
# DCA for L1-L2 / TL1 （内层 ADMM）
# ========================================================================================
class _InvATAplusI:
    """(A^T A + I)^{-1} r = r - A^T (I + AA^T)^{-1} A r （Woodbury 恒等式，O(mn)）。"""

    def __init__(self, A):
        self.A = A
        self.K = A.T @ np.linalg.inv(np.eye(A.shape[0]) + A @ A.T)

    def __call__(self, r):
        return r - self.K @ (self.A @ r)


def _dca_inner_admm(A, b, inv, Atb, z, x_start, lam, delta, inner_max, inner_tol, inner_min=50):
    """求解凸子问题  min lam*||x||_1 - <z, x>  s.t. Ax = b  的 ADMM（[26] Alg.2 / [47] Alg.3 内层）：
        L = lam||w||_1 - z^T x + u^T(x-w) + v^T(Ax-b) + delta/2||x-w||^2 + delta/2||Ax-b||^2
        x <- (A^TA + I)^{-1} (A^T b + w + (z - u - A^T v)/delta)
        w <- shrink(x + u/delta, lam/delta)
        u <- u + delta (x - w);   v <- v + delta (Ax - b)
    内层初值：x = w = x^n，u = v = 0（与 [26]/[47] 一致，每次外层迭代重置对偶变量）。
    停止：||x^{j+1}-x^j|| / ||x^{j+1}|| < inner_tol（1e-5）或 inner_max（1000）次；迭代下限 inner_min=50。
    （仅内层设迭代下限；DCA 外层不设，外层通常几次即收敛，强制 50 次会成倍增加耗时。）
    【补充】仅用相对变化判停时，delta=1 下 ADMM 初期会出现 w≡0 的"停滞平台"（相对变化可达 1e-15），
    导致内层被错误提前终止；因此同时要求原始残差 max(||x-w||, ||Ax-b||)/||x|| < inner_tol。"""
    x = x_start.copy(); w = x_start.copy()
    u = np.zeros_like(x); v = np.zeros(A.shape[0])
    for j in range(1, inner_max + 1):
        x_new = inv(Atb + w + (z - u - A.T @ v) / delta)
        w = shrink(x_new + u / delta, lam / delta)
        u = u + delta * (x_new - w)
        v = v + delta * (A @ x_new - b)
        nx = max(np.linalg.norm(x_new), 1e-300)
        resid = max(np.linalg.norm(x_new - w), np.linalg.norm(A @ x_new - b)) / nx
        # 迭代下限 inner_min=50：前 50 次不判停
        done = (j >= inner_min) and (rel_change(x_new, x) < inner_tol) and (resid < inner_tol)
        x = x_new
        if done:
            break
    return x, j


def _dca_inner_lp(A, b, z, lam):
    """内层子问题 min lam||x||_1 - <z,x> s.t. Ax=b 的精确 LP 解（可选方案，inner='lp'）：
    x = p - q, p,q >= 0, min (lam-z)^T p + (lam+z)^T q。L1-L2 中 |z|<=1=lam，TL1 中 |z|<2=lam，问题有界。"""
    from scipy.optimize import linprog
    n = A.shape[1]
    res = linprog(np.concatenate([lam - z, lam + z]), A_eq=np.hstack([A, -A]), b_eq=b,
                  bounds=(0, None), method="highs")
    return res.x[:n] - res.x[n:], 1


def _dca(A, b, x0, grad_h, lam, delta, outer_max, outer_tol, inner_max, inner_tol, inner="admm",
         inner_min=50):
    inv = _InvATAplusI(A)
    Atb = A.T @ b
    x = np.asarray(x0, dtype=float).copy()
    n_inner = 0
    k = 0
    for k in range(1, outer_max + 1):
        z = grad_h(x)
        if inner == "lp":
            x_new, j = _dca_inner_lp(A, b, z, lam)
        else:
            x_new, j = _dca_inner_admm(A, b, inv, Atb, z, x, lam, delta, inner_max, inner_tol, inner_min)
        n_inner += j
        done = rel_change(x_new, x) < outer_tol
        x = x_new
        if done:
            break
    return x, {"outer": k, "inner_total": n_inner}


def dca_l1ml2(A, b, x0, delta=100.0, outer_max=3000, outer_tol=1e-8, inner_max=1000, inner_tol=1e-5,
              inner="admm", inner_min=50):
    """L1-L2：||x||_1 - ||x||_2，DC 分解 g = ||x||_1 + I(Ax=b)，h = ||x||_2，z^n = x^n/||x^n||_2。
    外层：相对变化 < 1e-8 或 3000 次；内层：delta=100，相对变化 < 1e-5 或 1000 次（按约定）。
    【说明】delta 默认值由 1 改为 100：delta=1 时内层几乎每次跑满 1000 次，单次试验最长达数分钟；
            delta=100 时单次试验约 0.1–4 s，且小稀疏度下同样精确恢复。"""
    def grad_h(x):
        nx = np.linalg.norm(x)
        return x / nx if nx > 0 else np.zeros_like(x)
    return _dca(A, b, x0, grad_h, 1.0, delta, outer_max, outer_tol, inner_max, inner_tol, inner, inner_min)


def dca_tl1(A, b, x0, a=1.0, delta=100.0, outer_max=3000, outer_tol=1e-8, inner_max=1000, inner_tol=1e-5,
            inner="admm", inner_min=50):
    """TL1：sum (a+1)|x_i|/(a+|x_i|)，DC 分解 ([47] 式 (3.4),(3.11))：
        g = (a+1)/a ||x||_1 + I(Ax=b),  h = phi_a(x) = sum (a+1)|x_i|^2 / (a(a+|x_i|)),
        z = grad phi_a(x) = sign(x)(a+1)(|x|^2 + 2a|x|) / (a(a+|x|)^2)。
    参数同 dca_l1ml2；a = 1。"""
    def grad_h(x):
        ax = np.abs(x)
        return np.sign(x) * (a + 1.0) * (ax * ax + 2.0 * a * ax) / (a * (a + ax) ** 2)
    return _dca(A, b, x0, grad_h, (a + 1.0) / a, delta, outer_max, outer_tol, inner_max, inner_tol, inner,
                inner_min)
