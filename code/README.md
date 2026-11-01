# A Scale Invariant Approach for Sparse Signal Recovery —— Python 复现

原文：Rahimi, Wang, Dong, Lou, arXiv:1812.08852v4。依赖见 `requirements.txt`（**必须安装 scipy**，L1 初值使用 `scipy.optimize.linprog`）。

## 文件
| 文件 | 内容 |
|---|---|
| `common.py` | 矩阵/信号生成、L1（linprog）、各模型目标函数、随机种子 |
| `solvers_sparse.py` | L1/L2(+box) ADMM、Lp（Chartrand [9]）、DCA-L1−L2、DCA-TL1 |
| `mri.py` | Shepp-Logan(256)、径向线掩膜（radial_mask，scipy.ndimage.rotate）、L1/L2-grad、TV、L1−0.5L2、FBP |
| `exp_fig1_toy.py` | Fig.1 |
| `exp_fig2_convergence.py` | Fig.2 |
| `exp_compare.py` | Fig.3（DCT F=5,10）/ Fig.4（Gaussian r=0.1,0.8） |
| `exp_mri.py` | Fig.5（6 条线）及 8 条线验证 |
| `exp_fig6_empirical.py` | Fig.6 |

## 运行
```bash
python exp_fig1_toy.py
python exp_fig2_convergence.py --part all --plot
for p in 5 10; do python exp_compare.py --matrix dct --param $p --workers 4; done
for p in 0.1 0.8; do python exp_compare.py --matrix gauss --param $p --workers 4; done
python exp_compare.py --plot
python exp_mri.py --lines 6 --method all && python exp_mri.py --lines 6 --plot
python exp_mri.py --lines 8 --method all && python exp_mri.py --lines 8 --plot
python exp_fig6_empirical.py --part a --workers 4
python exp_fig6_empirical.py --part bc --workers 4
python exp_fig6_empirical.py --plot
```
蒙特卡洛脚本逐试验保存到 `results/*.json`，中断后再次运行会自动续跑（`--time-budget` 可设定单次运行时长）。

## 已确认的设置
F=20 不跑；s=2:2:24、每点 50 次；成功 rel.err ≤ 1e−3；失败率分母为总试验数；
L1/L2：ρ1=ρ2=100，最多 3000 次，tol 1e−8；DCA：外层 1e−8/3000，内层 ADMM δ=100（原约定 1，因速度改为 100）、1e−5、1000 次；
Lp：[9] 原文投影梯度，固定 40000 步；MRI：λ=100、ρ1=ρ2=ρ3=1、3000 次、256×256、FFT norm="ortho"；
Gaussian 矩阵不做列归一化；Table 1 不做。

## 实现中补充的约定（论文未说明）
1. 迭代下限 50：L1/L2(-box) ADMM、DCA 内层 ADMM、L1/L2-grad、TV、L1−0.5L2（每个 DCA 步内）前 50 次不判停；
   DCA 外层不设下限。L1/L2 ADMM 初值为 y=z=x0、v=w=0。
2. DCA 内层：除相对变化外，还要求原始残差 < 1e−5，否则 δ=1 时 ADMM 初期的停滞平台会被误判为收敛。
3. MRI 各算法：除相对变化外，还要求 ‖Au−f‖/‖f‖ < 1e−6（同样的原因，否则 TV 第 2 步就停止）。
4. TV / L1−0.5L2：μ=100（数据项）、λ=1（分裂罚参数）；L1−0.5L2 取 MaxDCA=10 × 300 次 Bregman = 3000 次 u 更新；c=0。
5. FBP 用零填充逆 FFT 实现。
6. Lp：初值默认为 L1 解（论文 4.2 节）；`--lp-init minnorm` 为 [9] 原文的最小范数初值。
7. Fig.6a：保持和为 1 约束，代码中标注"建议改为100"（ADMM 结果对尺度敏感）。
