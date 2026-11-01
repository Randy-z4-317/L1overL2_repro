# -*- coding: utf-8 -*-
"""
MRI 重建（论文 3.3 节 / 4.3 节）：
    shepp_logan       : MATLAB phantom(N)（默认 Modified Shepp-Logan，取值 [0,1]）的复刻
    radial_mask       : 径向线采样掩膜（scipy.ndimage.rotate 旋转中心线后取并集）
    l1dl2_grad        : L1/L2-grad，论文 Algorithm 3.2（含 [0,1] 盒约束）
    l1_minus_al2_grad : L1 - alpha*L2 梯度模型（Lou-Zeng-Osher-Xin [27] Algorithm 2，DCA + split Bregman）
                        alpha=0 时即为 TV / L1 on gradient（Goldstein-Osher split Bregman [15]）
    fbp_zero_filled   : "FBP"——零填充逆 FFT
测量算子：A u = M .* fft2(u, norm="ortho")（复数测量，u 为实数图像）。
梯度：∇u = (u_ij - u_(i+1)j, u_ij - u_i(j+1))，周期边界（论文 3.3 节）。
"""
import numpy as np
from scipy.ndimage import rotate


# ----------------------------------------------------------------------------------------
# 体模与掩膜
# ----------------------------------------------------------------------------------------
_MOD_SHEPP_LOGAN = np.array([
    #   A      a       b      x0      y0    phi(deg)
    [1.0,  .69,   .92,    0.0,   0.0,     0],
    [-.8,  .6624, .8740,  0.0,  -.0184,   0],
    [-.2,  .1100, .3100,  .22,   0.0,   -18],
    [-.2,  .1600, .4100, -.22,   0.0,    18],
    [.1,   .2100, .2500,  0.0,   .35,     0],
    [.1,   .0460, .0460,  0.0,   .1,      0],
    [.1,   .0460, .0460,  0.0,  -.1,      0],
    [.1,   .0460, .0230, -.08,  -.605,    0],
    [.1,   .0230, .0230,  0.0,  -.606,    0],
    [.1,   .0230, .0460,  .06,  -.605,    0]])


def shepp_logan(N=256):
    """复刻 MATLAB phantom(N)（默认 'Modified Shepp-Logan'，像素值位于 [0,1]，满足论文的盒约束假设）。
    注：skimage.data.shepp_logan_phantom() 为 400x400 且栅格化方式不同，故不使用。"""
    xax = (np.arange(N) - (N - 1) / 2.0) / ((N - 1) / 2.0)
    xg = np.tile(xax, (N, 1))            # x 坐标（列方向）
    yg = np.rot90(xg)                    # y 坐标（与 MATLAB rot90 一致：上方为 +1）
    p = np.zeros((N, N))
    for A, a, b, x0, y0, phi in _MOD_SHEPP_LOGAN:
        ph = np.deg2rad(phi)
        x = xg - x0; y = yg - y0
        cp, sp = np.cos(ph), np.sin(ph)
        idx = ((x * cp + y * sp) ** 2) / a ** 2 + ((y * cp - x * sp) ** 2) / b ** 2 <= 1.0
        p[idx] += A
    return p


def radial_mask(n, lines):
    """径向线采样掩膜（返回 bool 数组，已 ifftshift：零频位于 [0,0]，可直接与 fft2 结果相乘）。

    做法：在 m x m（m = ceil(sqrt(2) n)）的大画布上画一条水平中心线，按 180/lines 度间隔旋转
    （scipy.ndimage.rotate，order=0 最近邻）后取并集，再裁剪中心 n x n。
    【注意】旋转中心为 (m-1)/2，而水平线画在 centre 行，二者相差 1 个像素，故除 0° 外的线不严格过零频，
    掩膜也不是严格共轭对称（与 MATLAB imrotate 版本同样的特征）；N=256 时 6/7/8 条线的采样率
    约为 2.57% / 3.03% / 3.98%（与论文 Fig.5 及 Guo-Yin [18] 报告值吻合）。"""
    if n < 8 or n % 2 or not 1 <= lines <= 180:
        raise ValueError("请使用偶数且不小于 8 的图像尺寸，以及 1~180 条线")
    m = int(np.ceil(np.sqrt(2) * n))

    centre = int(np.floor(m / 2 + 1 + .5)) - 1

    aux = np.zeros((m, m))
    aux[centre, :] = 1

    image = np.zeros((m, m), dtype=bool)
    for angle in np.arange(lines) * 180 / lines:
        image |= rotate(aux, angle, reshape=False, order=0,
                        mode="constant", prefilter=False) > 0

    start = centre - n // 2
    return np.fft.ifftshift(image[start:start + n, start:start + n])


# ----------------------------------------------------------------------------------------
# 算子
# ----------------------------------------------------------------------------------------
def fwd(u, M):
    return M * np.fft.fft2(u, norm="ortho")


def grad(u):
    """离散梯度算子，∇u = (u_ij - u_(i+1)j, u_ij - u_i(j+1))，周期边界。返回 (2, N, N)。"""
    return np.stack([u - np.roll(u, -1, axis=0), u - np.roll(u, -1, axis=1)])


def grad_T(p):
    """∇ 的伴随：∇^T p。"""
    return (p[0] - np.roll(p[0], 1, axis=0)) + (p[1] - np.roll(p[1], 1, axis=1))


def sym_mask(M):
    """对称化掩膜 Ms(k) = (M(k) + M(-k)) / 2（-k 为模 N 的频率取反）。
    对实数图像 u，Re(A^H A u) = ifft2(Ms * fft2(u))：当 M 不严格共轭对称时（radial_mask 即如此），
    u 步 FFT 对角化的分母必须用 Ms，而不是 M；右端项无需另行对称化——分母 Ms 是偶函数，
    对 U = V/denom 取实部（np.real(ifft2(U))）等价于先把 V 对称化。
    测量 Au = M * fft2(u) 与数据残差、w/z 的 Bregman 更新仍使用原掩膜 M。"""
    return 0.5 * (M + np.roll(np.flip(M), 1, axis=(0, 1)))


def lap_eig(N1, N2):
    """∇^T∇ = -Δ（周期边界）在 FFT 下的特征值：(2-2cos(2πk/N1)) + (2-2cos(2πl/N2))。"""
    k = np.arange(N1)[:, None]; l = np.arange(N2)[None, :]
    return (2 - 2 * np.cos(2 * np.pi * k / N1)) + (2 - 2 * np.cos(2 * np.pi * l / N2))


def rel_err(u, u_true):
    return np.linalg.norm(u - u_true) / np.linalg.norm(u_true)


def _tau_root(D):
    q = 27.0 * D + 2.0
    C = np.cbrt((q + np.sqrt(q * q - 4.0)) / 2.0)
    return 1.0 / 3.0 + (C + 1.0 / C) / 3.0


def _shrink(v, mu):
    return np.sign(v) * np.maximum(np.abs(v) - mu, 0.0)


# ----------------------------------------------------------------------------------------
# FBP（零填充）
# ----------------------------------------------------------------------------------------
def fbp_zero_filled(f, M):
    """按约定，以零填充逆 FFT 作为 "FBP"（径向采样下的直接反变换，无密度补偿）。"""
    return np.real(np.fft.ifft2(M * f, norm="ortho"))


# ----------------------------------------------------------------------------------------
# L1/L2-grad —— Algorithm 3.2
# ----------------------------------------------------------------------------------------
def l1dl2_grad(f, M, lam=100.0, rho1=1.0, rho2=1.0, rho3=1.0, box=(0.0, 1.0),
               maxit=3000, tol=1e-8, res_tol=1e-6, min_iter=50, u0=None, u_true=None, record=False, rng=None):
    """min ||∇u||_1/||∇u||_2  s.t. Au = f, u in [0,1]（式 (3.17)），增广拉格朗日 (3.19)：
        u <- (λA^TA - (ρ1+ρ2)Δ + ρ3 I)^{-1} [λA^T(f+w) + ρ1∇^T(d-b) + ρ2∇^T(h-g) + ρ3(v-e)]   (3.20)，FFT 求逆
            （A^TA 的频域对角元用对称化掩膜 Ms，见 sym_mask）
        v <- min(max(u+e, 0), 1)                                                               (3.21)
        h <- τ(∇u+g)，τ 为 τ^3-τ^2-D=0 的根，D = ||d||_1/(ρ2||∇u+g||_2^3) 当 ||∇u+g||_2=0, h <- e
        d <- shrink(∇u+b, 1/(ρ1||h||_2))
        b <- b + ∇u - d;  
        g <- g + ∇u - h;  
        w <- w + f - Au;  
        e <- e + u - v
    默认参数（按约定）：λ=100，ρ1=ρ2=ρ3=1，最多 3000 次，||u^k-u^{k-1}||/||u^k|| < 1e-8 停止。
    初值：论文未说明，此处所有变量初始化为 0（u0 可另行指定）。
    停止：迭代次数 >= min_iter（50），且相对变化 < tol 且数据残差 ||Au-f||/||f|| < res_tol（1e-6）；仅用相对变化时，
          迭代初期 d≡0 的停滞阶段会被误判为收敛（Bregman 迭代的典型现象）。
    注：||·||_2 为整个梯度场（两个分量合并）的全局 L2 范数。"""
    N1, N2 = M.shape
    rng = rng if rng is not None else np.random.default_rng(0)
    Ms = sym_mask(M)                       # 对称化掩膜（见 sym_mask），仅用于 u 步分母

    denom = lam * Ms + (rho1 + rho2) * lap_eig(N1, N2) + rho3
    denom_nobox = np.where(denom - rho3 == 0, 1.0, denom - rho3)   # 无盒约束时不含 ρ3 I 项
    
    # 设置初值
    u = np.zeros((N1, N2)) if u0 is None else u0.copy()
    gu = grad(u)
    d = gu.copy(); 
    h = gu.copy(); 
    b = np.zeros_like(gu); 
    g = np.zeros_like(gu)
    v = np.clip(u, *box) if box is not None else u.copy()
    e = np.zeros_like(u); 
    w = np.zeros_like(f)

    hist = {"gd": [], "gh": [], "obj": [], "re": []} if record else None
    it = 0
    for it in range(1, maxit + 1):
        u_old = u
        # u 步（FFT 对角化）。A^T y = Re(ifft2(M*y))，合并到频域中一次求解
        rhs_img = rho1 * grad_T(d - b) + rho2 * grad_T(h - g)
        if box is not None:
            rhs_img = rhs_img + rho3 * (v - e)
            U = (lam * M * (f + w) + np.fft.fft2(rhs_img, norm="ortho")) / denom
        else:
            U = (lam * M * (f + w) + np.fft.fft2(rhs_img, norm="ortho")) / denom_nobox
        u = np.real(np.fft.ifft2(U, norm="ortho")) # u = Re(ifft2(U))
        
        # v 步
        if box is not None:
            v = np.clip(u + e, box[0], box[1])
        gu = grad(u)
        # h 步
        c = np.sum(np.abs(d))
        q = gu + g
        eta = np.linalg.norm(q)
        if eta == 0.0:
            r = rng.standard_normal(q.shape)
            h = r * (np.cbrt(c / rho2) / np.linalg.norm(r))
        elif c == 0.0:
            h = q
        else:
            h = _tau_root(c / (rho2 * eta ** 3)) * q
        # d 步
        d = _shrink(gu + b, 1.0 / (rho1 * np.linalg.norm(h)))
        # 对偶 / Bregman 更新
        b = b + gu - d
        g = g + gu - h
        w = w + f - fwd(u, M)
        if box is not None:
            e = e + u - v
        if record:
            hist["gd"].append(np.linalg.norm(gu - d))
            hist["gh"].append(np.linalg.norm(gu - h))
            hist["obj"].append(np.sum(np.abs(gu)) / np.linalg.norm(gu))
            if u_true is not None:
                hist["re"].append(rel_err(u, u_true))
        if it >= min_iter and np.linalg.norm(u - u_old) / max(np.linalg.norm(u), 1e-300) < tol and \
                np.linalg.norm(fwd(u, M) - f) / np.linalg.norm(f) < res_tol:     # 迭代下限 50
            break
    info = {"iters": it}
    if record:
        info["hist"] = {k: np.array(vv) for k, vv in hist.items()}
    return u, info


# ----------------------------------------------------------------------------------------
# L1 - alpha*L2 on gradient —— [27] Algorithm 2（alpha=0 即 TV/split Bregman [15]）
# ----------------------------------------------------------------------------------------
def l1_minus_al2_grad(f, M, alpha=0.5, mu=100.0, lam=1.0, max_dca=10, max_breg_outer=300,
                      max_breg_inner=1, tol=1e-8, res_tol=1e-6, min_iter=50, u0=None):
    """[27] 的约束模型 (2.7)：min ||Dx u||_1 + ||Dy u||_1 - alpha*|| sqrt(|Dx u|^2+|Dy u|^2) ||_1  s.t. Au = f，
    即"各向异性 TV - alpha*各向同性 TV"（论文中记作 L1 - 0.5L2，alpha = 0.5）。
    [27] Algorithm 2（c = 0，与 [27] 实验一致）：
        for DCA:   q = ∇u/|∇u|（逐像素，分母为 0 处取 0）；bx = by = 0
          for BregmanOuter:
            for BregmanInner:
              u = (μA^TA - λΔ)^{-1}(μA^T z + λ∇^T(d - b))
              d = shrink(∇u + b + αq/λ, 1/λ);  b = b + ∇u - d
            z = z + f - Au
    参数（按约定与 L1/L2-grad 一致）：数据项权重 μ = 100（对应 L1/L2-grad 的 λ），分裂罚参数 λ = 1（对应 ρ）。
    [27] 未给出 MaxDCA / MaxBregman 的取值；此处取 MaxDCA=10、BregmanOuter=300、BregmanInner=1，
    总 u 更新次数 = 3000（与 3000 次迭代上限一致）。alpha=0 时取 max_dca=1、max_breg_outer=3000 即为 TV。
    停止（每个 Bregman 外层步检查）：每个 DCA 步内 Bregman 外层迭代 >= min_iter（50），
          且相对变化 < tol 且 ||Au-f||/||f|| < res_tol，理由同 l1dl2_grad。
    无盒约束（与 [15]/[27] 原文一致）。z（数据 Bregman 变量）在 DCA 迭代间不重置，与 [27] 一致。"""
    N1, N2 = M.shape
    denom = mu * sym_mask(M) + lam * lap_eig(N1, N2)     # 对称化掩膜（见 sym_mask），仅用于 u 步分母
    denom = np.where(denom == 0, 1.0, denom)   # 零频若未采样（M[0,0]=0）则避免除零；该掩膜总包含零频
    u = np.zeros((N1, N2)) if u0 is None else u0.copy()
    z = f.copy()
    d = grad(u); b = np.zeros_like(d)
    q = np.zeros_like(d)
    total = 0
    for k in range(max_dca):
        if alpha != 0 and k > 0:
            gu = grad(u)
            mag = np.sqrt(gu[0] ** 2 + gu[1] ** 2)
            q = np.where(mag > 0, gu / np.where(mag > 0, mag, 1.0), 0.0)
        b = np.zeros_like(d)
        for kb in range(1, max_breg_outer + 1):
            u_old = u
            for _ in range(max_breg_inner):
                U = (mu * M * z + np.fft.fft2(lam * grad_T(d - b), norm="ortho")) / denom
                u = np.real(np.fft.ifft2(U, norm="ortho"))
                gu = grad(u)
                d = _shrink(gu + b + alpha * q / lam, 1.0 / lam)
                b = b + gu - d
                total += 1
            z = z + f - fwd(u, M)
            if kb >= min_iter and np.linalg.norm(u - u_old) / max(np.linalg.norm(u), 1e-300) < tol and \
                    np.linalg.norm(fwd(u, M) - f) / np.linalg.norm(f) < res_tol:     # 迭代下限 50
                break
    return u, {"iters": total}


def tv_grad(f, M, mu=100.0, lam=1.0, maxit=3000, tol=1e-8, min_iter=50):
    """L1 on gradient（各向异性 TV），约束形式，split Bregman [15]；即 l1_minus_al2_grad 取 alpha = 0。"""
    return l1_minus_al2_grad(f, M, alpha=0.0, mu=mu, lam=lam, max_dca=1, max_breg_outer=maxit,
                             max_breg_inner=1, tol=tol, min_iter=min_iter)
