import numpy as np
import matplotlib.pyplot as plt
import control as ctl
import math

# 1) 定义被控对象 G(s)
num = [273000]
den = [1, 195, 12650, 273000]
G = ctl.tf(num, den)

# 2) 选定 K 的扫描范围
Krange = np.linspace(0, 100, 1000)

# 3) 手动计算每个 K 下的闭环极点，并存储
all_poles = []
for K in Krange:
    # 定义 PI 型控制器 C(s) = (s + K)/s ()
    C = ctl.tf([1, K], [1, 0])   # 分子 [1,K] 表示 (s+K)，分母 [1,0] 表示 s
    # 开环传递函数
    L = G * C
    # 闭环传递函数（单位负反馈）
    closed_loop_tf = ctl.feedback(L, 1)
    # 取闭环极点
    poles = ctl.poles(closed_loop_tf)
    poles = sorted(poles, key=lambda pole: (np.real(pole), np.imag(pole)))
    all_poles.append(poles)

all_poles = np.array(all_poles)  # 形状 (len(Krange), 系统阶数)

# 4) 在指定的 K 值处打印极点和阻尼比
for K_value in [30,40,50,60]:
    idx = np.argmin(np.abs(Krange - K_value))
    poles_at_K = all_poles[idx]
    # 这里假设“最后一个”是你关心的共轭复极点分支
    # 如果顺序不对，可以手动改成自己需要的极点索引
    sorted_poles = sorted(poles_at_K, key=lambda pole: (np.real(pole), np.imag(pole)))
    print(poles_at_K)
    conjugate_pole = poles_at_K[2]
    real_part = np.real(conjugate_pole)
    imag_part = np.imag(conjugate_pole)
    wn = math.sqrt(real_part**2 + imag_part**2)
    zeta = -real_part / wn
    print(f"K = {K_value:.1f}, pole = {conjugate_pole:.4f}, "
          f"wn = {wn:.4f}, zeta = {zeta:.4f}")

# 5) 绘制根轨迹
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
n_poles = all_poles.shape[1]  # 系统阶数
for i in range(n_poles):
    ax.scatter(np.real(all_poles[:, i]),
               np.imag(all_poles[:, i]), color='red', s=10)

# 6) 设置坐标范围、标题、网格等
ax.set_xlim([-150, 20])
ax.set_ylim([-170, 170])
ax.grid(True)
ax.set_title("Root Locus with C(s) = 1 + K/s")
ax.set_xlabel("Real Axis")
ax.set_ylabel("Imag Axis")
ax.legend()
plt.show()
