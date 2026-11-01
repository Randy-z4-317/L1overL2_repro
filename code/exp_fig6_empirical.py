# -*- coding: utf-8 -*-
"""
Fig.6（第 5 节，经验验证）：
  (a) min ||x||_1/||x||_2  s.t.  [A; 1^T] x = [0; 1]（和为 1 约束以排除 x=0），
      F = 2:2:18，每个 F 50 个随机 DCT 矩阵（64x1024），画比值均值 ± 标准差；
      求解：L1/L2 ADMM（rho1=rho2=100，最多 3000 次），以该扩展系统的 L1 解为初值。
      【注】论文写作 Ã = [A; ones(n,1)]，按约定改为全 1 行 ones(1,n)。
      【建议改为100】右端和为 1 时 ADMM 结果对尺度敏感，见 task_a 中的注释。
  (b) 有最小间隔（MS = 40）时 L1 的成功率；(c) 无间隔约束时 L1 的成功率；
      F ∈ {2,6,10,14,18}，s = 2:2:20，每点 50 次；(b)(c) 使用完全相同的矩阵；成功判据 rel.err <= 1e-3。
用法：
    python exp_fig6_empirical.py --part a  [--workers 2 --time-budget 3000]
    python exp_fig6_empirical.py --part bc [--workers 2 --time-budget 3000]
    python exp_fig6_empirical.py --plot
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_rng, dct_matrix, sparse_signal, l1_linprog, rel_err, obj_ratio
import solvers_sparse as S
from exp_compare import load, save

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "figures")
F_A = list(range(2, 19, 2))
F_BC = [2, 6, 10, 14, 18]
S_BC = list(range(2, 21, 2))
MS = 40


def task_a(args):
    F, trial = args
    rng = make_rng(61, F, trial)
    A = dct_matrix(64, 1024, F, rng)
    At = np.vstack([A, np.ones((1, 1024))])
    # 【建议改为100】和为 1 时解的幅值很小，在 rho1=rho2=100 下 ADMM 结果对尺度敏感
    #  （测试中和为 1 与和为 100 得到的比值明显不同）；目前按约定保持和为 1。
    bt = np.concatenate([np.zeros(64), [1.0]])
    x0 = l1_linprog(At, bt)
    x, info = S.l1l2_admm(At, bt, x0, rho1=100.0, rho2=100.0, maxit=3000, rng=make_rng(61, F, trial, 99))
    return {"F": F, "trial": trial, "ratio": float(obj_ratio(x)), "ratio_l1init": float(obj_ratio(x0)),
            "iters": info["iters"]}


def task_bc(args):
    F, s, trial = args
    rng = make_rng(62, F, s, trial)
    A = dct_matrix(64, 1024, F, rng)            # (b)(c) 共用同一个矩阵
    x_ms = sparse_signal(1024, s, rng, min_sep=MS)
    x_no = sparse_signal(1024, s, rng)
    e_ms = rel_err(l1_linprog(A, A @ x_ms), x_ms)
    e_no = rel_err(l1_linprog(A, A @ x_no), x_no)
    return {"F": F, "s": s, "trial": trial, "err_ms": float(e_ms), "err_no": float(e_no)}


def run(part, trials, workers, budget):
    os.makedirs(RES_DIR, exist_ok=True)
    path = os.path.join(RES_DIR, f"fig6_{part}.json")
    data = load(path)
    if part == "a":
        done = {(r["F"], r["trial"]) for r in data["records"]}
        tasks = [(F, t) for F in F_A for t in range(trials) if (F, t) not in done]
        fn = task_a
    else:
        done = {(r["F"], r["s"], r["trial"]) for r in data["records"]}
        tasks = [(F, s, t) for F in F_BC for s in S_BC for t in range(trials) if (F, s, t) not in done]
        fn = task_bc
    print(f"{path}: 已完成 {len(done)}，待运行 {len(tasks)}")
    t0 = time.time()
    pool = Pool(workers) if workers > 1 else None
    it = pool.imap_unordered(fn, tasks) if pool else map(fn, tasks)
    try:
        for k, r in enumerate(it, 1):
            data["records"].append(r)
            if k % 5 == 0 or k == len(tasks):
                save(path, data)
            if k % 50 == 0:
                print(f"  {k}/{len(tasks)} 用时 {time.time() - t0:.0f}s", flush=True)
            if budget and time.time() - t0 > budget:
                print("到达时间预算，已保存，退出（再次运行可续跑）")
                break
    finally:
        save(path, data)
        if pool:
            pool.terminate()


def plot(trials):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 9))
    axa = plt.subplot2grid((2, 4), (0, 1), colspan=2)
    axb = plt.subplot2grid((2, 4), (1, 0), colspan=2)
    axc = plt.subplot2grid((2, 4), (1, 2), colspan=2)
    pa = os.path.join(RES_DIR, "fig6_a.json")
    if os.path.exists(pa):
        rec = load(pa)["records"]
        Fs = sorted({r["F"] for r in rec})
        mu = [np.mean([r["ratio"] for r in rec if r["F"] == F]) for F in Fs]
        sd = [np.std([r["ratio"] for r in rec if r["F"] == F]) for F in Fs]
        axa.errorbar(Fs, mu, yerr=sd, fmt="-o", capsize=3)
        for F, m_, s_ in zip(Fs, mu, sd):
            print(f"F={F:2d}  mean ratio={m_:.3f}  std={s_:.3f}")
    axa.set_xlabel("F"); axa.set_ylabel("ratio"); axa.set_title("(a) The mean of ratios", y=-0.22)
    pbc = os.path.join(RES_DIR, "fig6_bc.json")
    if os.path.exists(pbc):
        rec = load(pbc)["records"]
        for F in F_BC:
            for ax, key in ((axb, "err_ms"), (axc, "err_no")):
                ss = sorted({r["s"] for r in rec if r["F"] == F})
                if not ss:
                    continue
                sr = [100.0 * sum(r[key] <= 1e-3 for r in rec if r["F"] == F and r["s"] == s) / trials for s in ss]
                ax.plot(ss, sr, "-o", ms=4, label=f"F = {F}")
    for ax, t in ((axb, "(b) Success rates with MS"), (axc, "(c) Success rates without MS")):
        ax.set_xlim(1, 21); ax.set_ylim(0, 100); ax.set_xlabel("sparsity"); ax.set_ylabel("%")
        ax.set_title(t, y=-0.22); ax.legend(loc="lower left")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig6_empirical.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "bc"])
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=0)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.part:
        run(a.part, a.trials, a.workers, a.time_budget)
    if a.plot:
        plot(a.trials)
