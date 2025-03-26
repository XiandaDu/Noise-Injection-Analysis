import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal

# 系统定义
num = [4*50000]               # 分子：Kp = 10
den = [1, 170, 50000]          # 分母：s(s+2)
system = signal.TransferFunction(num, den)

# 频率范围
w = np.logspace(1, 4, 1000)
w, mag, phase = signal.bode(system, w)

# 找 wgc: 幅值 = 0 dB 时的频率
# np.sign() 用于检测变化, np.diff() 用于找到跨越零的地方
crossings = np.where(np.diff(np.sign(mag)) < 0)[0]
if len(crossings) > 0:
    wgc_index = crossings[0]
    wgc = w[wgc_index]
    PM = phase[wgc_index] + 180
else:
    wgc = None
    PM = None

# 找 wpc: 相位 = -180° 时的频率
wpc_index = np.argmin(np.abs(phase + 180))
wpc = w[wpc_index]
GM = -mag[wpc_index]  # dB 表示

# 找距离-180° 37°的点：即相位 -143°
target_phase = -180 + 42  # -143°
target_index = np.argmin(np.abs(phase - target_phase))
target_w = w[target_index]
target_mag = mag[target_index]

# 画图
plt.figure()
plt.subplot(2,1,1)
plt.semilogx(w, mag)
plt.axhline(0, color='gray', linestyle='--')
if wgc is not None:
    plt.axvline(wgc, color='red', linestyle='--', label=f'wgc = {wgc:.2f} rad/s')
plt.title('Bode Magnitude Plot')
plt.ylabel('Magnitude (dB)')
plt.legend()

plt.subplot(2,1,2)
plt.semilogx(w, phase)
plt.axhline(-180, color='gray', linestyle='--')
plt.axvline(wpc, color='blue', linestyle='--', label=f'wpc = {wpc:.2f} rad/s')
plt.axvline(target_w, color='green', linestyle='--', label=f'w at phase = -143°: {target_w:.2f} rad/s')
plt.scatter(target_w, phase[target_index], color='green', zorder=5)
plt.text(target_w, phase[target_index], f' {target_w:.2f} rad/s\n({target_mag:.2f} dB)', color='green', fontsize=9, verticalalignment='bottom', horizontalalignment='right')
plt.title('Bode Phase Plot')
plt.ylabel('Phase (deg)')
plt.xlabel('Frequency (rad/s)')
plt.legend()
plt.tight_layout()
plt.show()

# 输出结果
if wgc is not None:
    print(f"Gain Crossover Frequency (wgc): {wgc:.2f} rad/s")
    print(f"Phase Margin (PM): {PM:.2f} degrees")
else:
    print("No gain crossover frequency found.")
print(f"Phase Crossover Frequency (wpc): {wpc:.2f} rad/s")
print(f"Gain Margin (GM): {GM:.2f} dB")
print(f"Frequency corresponding to phase = -143°: {target_w:.2f} rad/s")
print(f"Corresponding magnitude: {target_mag:.2f} dB")
