# -*- coding: utf-8 -*-
"""
Fig.3（过采样 DCT，F=5、F=10）与 Fig.4（Gaussian，r=0.1、r=0.8）：
6 种模型（L1, TL1, Lp, L1-L2, L1/L2, L1/L2-box）的成功率 / 模型失败率 / 算法失败率。

用法（每组配置单独运行，支持断点续跑，结果追加保存在 results/compare_<mat>_<par>.json）：
    python exp_compare.py --matrix dct --param 5
    python exp_compare.py --matrix dct --param 10
    python exp_compare.py --matrix gauss --param 0.1
    python exp_compare.py --matrix gauss --param 0.8
    python exp_compare.py --plot            # 由已保存结果绘制 figures/fig3.png、figures/fig4.png
可选：--workers N（多进程），--time-budget 秒（到时安全退出，再次运行继续），--trials 50。

设置汇总：
  * A in R^{64x1024}；s = 2:2:24；每个稀疏度 50 次试验；固定随机种子（common.make_rng）
  * 成功：||x* - x||/||x|| <= 1e-3
  * 失败分类（只对失败试验）：F(x_true) > F(x*) → 模型失败；F(x_true) < F(x*) → 算法失败；
    F 为各模型自身的目标函数（common.OBJECTIVES）；三类比率的分母均为总试验数（50）
  * 初值：L1 解（linprog）；L1/L2 的 rho1=rho2=100，最多 3000 次，tol 1e-8
  * L1-L2 / TL1：DCA，外层 tol 1e-8 / 最多 3000 次；内层 ADMM delta=100 / 1e-5 / 1000 次（--dca-delta 可改，原约定为 1）
  * Lp：p=0.5，[9] 原文投影梯度法，固定 40000 步；初值默认 L1 解（论文 4.2 节），--lp-init minnorm 为 [9] 原文的最小范数初值
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (make_rng, dct_matrix, gaussian_matrix, sparse_signal, l1_linprog,
                    rel_err, OBJECTIVES)
import solvers_sparse as S

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "figures")
METHODS = ["L1", "TL1", "Lp", "L1-L2", "L1/L2", "L1/L2-box"]
M_ROWS, N_COLS = 64, 1024
SPARSITIES = list(range(2, 25, 2))
SUCCESS_TOL = 1e-3


def run_trial(task):
    matrix, param, s, trial, opt = task
    code = 1 if matrix == "dct" else 2
    rng = make_rng(code, int(round(param * 100)), s, trial)
    A = dct_matrix(M_ROWS, N_COLS, param, rng) if matrix == "dct" else gaussian_matrix(M_ROWS, N_COLS, param, rng)
    x = sparse_signal(N_COLS, s, rng)
    b = A @ x
    P = S.AffineProjector(A, b)
    rec, tim = {}, {}

    t = time.time(); x1 = l1_linprog(A, b); tim["L1"] = time.time() - t
    rec["L1"] = x1
    t = time.time()
    rec["TL1"], _ = S.dca_tl1(A, b, x1, a=1.0, delta=opt["dca_delta"], inner=opt["dca_inner"],
                              outer_max=opt["maxit"], inner_max=opt["inner_max"], inner_tol=opt["inner_tol"])
    tim["TL1"] = time.time() - t
    t = time.time()
    lp0 = x1 if opt["lp_init"] == "l1" else P(np.zeros(N_COLS))
    rec["Lp"], _ = S.lp_chartrand(A, b, lp0, p=0.5, n_iter=opt["lp_iters"], proj=P)
    tim["Lp"] = time.time() - t
    t = time.time()
    rec["L1-L2"], _ = S.dca_l1ml2(A, b, x1, delta=opt["dca_delta"], inner=opt["dca_inner"],
                                  outer_max=opt["maxit"], inner_max=opt["inner_max"], inner_tol=opt["inner_tol"])
    tim["L1-L2"] = time.time() - t
    yrng = make_rng(code, int(round(param * 100)), s, trial, 99)
    t = time.time()
    rec["L1/L2"], _ = S.l1l2_admm(A, b, x1, rho1=opt["rho"], rho2=opt["rho"], maxit=opt["maxit"], rng=yrng, proj=P)
    tim["L1/L2"] = time.time() - t
    t = time.time()
    rec["L1/L2-box"], _ = S.l1l2_admm(A, b, x1, rho1=opt["rho"], rho2=opt["rho"], box=(-1.0, 1.0),
                                      maxit=opt["maxit"], rng=yrng, proj=P)
    tim["L1/L2-box"] = time.time() - t

    out = {"s": s, "trial": trial}
    for mth in METHODS:
        F = OBJECTIVES[mth]
        out[mth] = {"err": float(rel_err(rec[mth], x)), "F_true": float(F(x)),
                    "F_rec": float(F(rec[mth])), "time": tim[mth]}
    return out


def res_path(matrix, param):
    return os.path.join(RES_DIR, f"compare_{matrix}_{param:g}.json")


def load(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"records": []}


def save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def run(args):
    os.makedirs(RES_DIR, exist_ok=True)
    path = res_path(args.matrix, args.param)
    data = load(path)
    opt = {"dca_delta": args.dca_delta, "dca_inner": args.dca_inner, "inner_max": args.inner_max,
           "inner_tol": args.inner_tol, "maxit": args.maxit, "rho": args.rho,
           "lp_init": args.lp_init, "lp_iters": args.lp_iters}
    if data.get("options") not in (None, opt):
        print("警告：已有结果使用的参数与本次不同：", data["options"])
    data["options"] = opt
    done = {(r["s"], r["trial"]) for r in data["records"]}
    sp = args.sparsities if args.sparsities else SPARSITIES
    tasks = [(args.matrix, args.param, s, t, opt) for s in sp for t in range(args.trials) if (s, t) not in done]
    print(f"{path}: 已完成 {len(done)}，待运行 {len(tasks)}")
    t0 = time.time()
    if args.workers > 1:
        pool = Pool(args.workers)
        it = pool.imap_unordered(run_trial, tasks)
    else:
        pool = None
        it = map(run_trial, tasks)
    try:
        for k, r in enumerate(it, 1):
            data["records"].append(r)
            save(path, data)
            if k % 10 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)}  用时 {time.time() - t0:.0f}s", flush=True)
            if args.time_budget and time.time() - t0 > args.time_budget:
                print("到达时间预算，已保存，退出（再次运行可续跑）")
                break
    finally:
        if pool is not None:
            pool.terminate()


def summarize(records, trials):
    """返回 {method: {"s": [...], "success": [...], "model": [...], "algo": [...]}}（百分比）。"""
    out = {}
    ss = sorted({r["s"] for r in records})
    for mth in METHODS:
        suc, mod, alg = [], [], []
        for s in ss:
            rs = [r[mth] for r in records if r["s"] == s]
            n_tot = trials if trials else len(rs)
            sc = sum(r["err"] <= SUCCESS_TOL for r in rs)
            mf = sum((r["err"] > SUCCESS_TOL) and (r["F_true"] > r["F_rec"]) for r in rs)
            af = sum((r["err"] > SUCCESS_TOL) and (r["F_true"] < r["F_rec"]) for r in rs)
            suc.append(100.0 * sc / n_tot); mod.append(100.0 * mf / n_tot); alg.append(100.0 * af / n_tot)
        out[mth] = {"s": ss, "success": suc, "model": mod, "algo": alg}
    return out


STYLE = {"L1": ("tab:blue", "o"), "TL1": ("tab:orange", "s"), "Lp": ("tab:green", "^"),
         "L1-L2": ("tab:red", "v"), "L1/L2": ("tab:purple", "d"), "L1/L2-box": ("tab:brown", "*")}


def plot(trials):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    groups = [("fig3", "dct", [5.0, 10.0], "F", "oversampled DCT matrices"),
              ("fig4", "gauss", [0.1, 0.8], "r", "Gaussian matrix case")]
    rows = [("success", "Success rates"), ("model", "Model failures"), ("algo", "Algorithm failures")]
    for name, mat, pars, sym, title in groups:
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        missing = False
        for j, par in enumerate(pars):
            path = res_path(mat, par)
            if not os.path.exists(path):
                missing = True
                continue
            summ = summarize(load(path)["records"], trials)
            for i, (key, lab) in enumerate(rows):
                ax = axes[i, j]
                for mth in METHODS:
                    c, mk = STYLE[mth]
                    ax.plot(summ[mth]["s"], summ[mth][key], color=c, marker=mk, ms=5, label=mth)
                ax.set_xlim(0, 25); ax.set_ylim(0, 100)
                ax.set_xlabel("sparsity"); ax.set_ylabel("%")
                ax.set_title(f"({'abcdef'[2 * i + j]}) {lab} ({sym} = {par:g})", y=-0.28, fontsize=11)
                ax.legend(fontsize=8, loc="best")
        fig.suptitle(f"Success rates, model failures, algorithm failures for 6 algorithms ({title})")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(FIG_DIR, f"{name}.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        print("saved", out, "(部分配置缺失)" if missing else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", choices=["dct", "gauss"])
    ap.add_argument("--param", type=float)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--sparsities", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=0)
    ap.add_argument("--maxit", type=int, default=3000)
    ap.add_argument("--rho", type=float, default=100.0)
    ap.add_argument("--dca-delta", type=float, default=100.0)
    ap.add_argument("--dca-inner", choices=["admm", "lp"], default="admm")
    ap.add_argument("--inner-max", type=int, default=1000)
    ap.add_argument("--inner-tol", type=float, default=1e-5)
    ap.add_argument("--lp-init", choices=["l1", "minnorm"], default="l1")
    ap.add_argument("--lp-iters", type=int, default=40000)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        plot(a.trials)
    else:
        if a.matrix is None or a.param is None:
            ap.error("需要 --matrix 和 --param")
        run(a)
