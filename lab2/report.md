# Lab 2：基于 FLIP 的流体仿真实验报告

**作者**: 宋星杰 | **日期**: 2026年5月 | **状态**: 草稿

## 1 概述

本实验实现了一个完整的 3D FLIP（Fluid Implicit Particle）流体仿真框架。FLIP 是一种**混合拉格朗日-欧拉方法**，它结合了粒子法（用于平流，避免数值耗散）和网格法（用于压力投影，高效求解不可压缩性）的优势。整个仿真基于 Taichi 并行计算框架实现，运行在 Vulkan 后端上，并使用 Taichi GGUI 提供交互式 3D 渲染。

仿真核心流程遵循 Ten Minute Physics（Lecture 18）提出的标准 FLIP 循环：粒子积分 → 碰撞处理 → 粒子分离 → P2G 传输 → 不可压缩求解 → G2P 传输。在此基础上，本实验额外实现了 APIC 方法（B4）、欧拉流体对比（B3）、CG 压力求解器（B2）、表面重建（B5）以及可视化增强（B1）。

## 2 核心实现

### 2.1 数据结构与交错网格

仿真采用 **MAC（Marker-and-Cell）交错网格** 约定。与将所有速度分量存储在网格单元中心的同位网格不同，MAC 网格将速度分量存储在单元面上：u 分量位于 $x$ 方向的面上，v 分量位于 $y$ 方向的面上，w 分量位于 $z$ 方向的面上。这种布局天然保证了压力投影的离散散度算子具有紧凑模板，避免了同位网格中常见的棋盘格不稳定性。

```python
# core.py — 交错网格声明
self.grid_u = ti.field(dtype=float, shape=(nx + 1, ny, nz))   # u: x-face
self.grid_v = ti.field(dtype=float, shape=(nx, ny + 1, nz))   # v: y-face
self.grid_w = ti.field(dtype=float, shape=(nx, ny, nz + 1))   # w: z-face
```

网格单元被标记为三种类型：**流体**（0）、**空气**（1）和**固体**（2）。流体单元含有粒子，空气单元为空，固体单元代表障碍物。每次子步中，`relabel_and_density` 核函数根据粒子位置重新标记单元类型，并同时统计每个单元内的粒子密度。

```python
# core.py — relabel_and_density
@ti.kernel
def relabel_and_density(self):
    for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
        if self.cell_type[i, j, k] != 2:
            self.cell_type[i, j, k] = 1  # 重置为空气
        self.particle_density[i, j, k] = 0.0
    for p in range(self.num_particles):
        ci = int(self.pos[p][0] / self.dx)
        cj = int(self.pos[p][1] / self.dx)
        ck = int(self.pos[p][2] / self.dx)
        if 0 <= ci < self.nx and 0 <= cj < self.ny and 0 <= ck < self.nz:
            self.cell_type[ci, cj, ck] = 0  # 标记为流体
            self.particle_density[ci, cj, ck] += 1.0
```

### 2.2 PIC/FLIP 混合传输

FLIP 方法的核心在于粒子与网格之间的速度传输。P2G（Particle-to-Grid）阶段将粒子速度散射到交错网格上，使用三线性插值权重。对于 u 分量，粒子位置需转换为 u 网格坐标系：$g_x = p_x / \Delta x$，$g_y = p_y / \Delta x - 0.5$，$g_z = p_z / \Delta x - 0.5$，以正确对齐到 $u$ 分量所在的面上。

```python
# flip.py — P2G 传输（u 分量）
gx = px / dx
gy = py / dx - 0.5
gz = pz / dx - 0.5
i0, j0, k0 = int(ti.floor(gx)), int(ti.floor(gy)), int(ti.floor(gz))
fx, fy, fz = gx - i0, gy - j0, gz - k0
for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
    w = ((fx if di else 1-fx) * (fy if dj else 1-fy)
         * (fz if dk else 1-fz))
    self.grid_u[i0+di, j0+dj, k0+dk] += w * vx
    self.grid_u_weight[i0+di, j0+dj, k0+dk] += w
```

G2P（Grid-to-Particle）阶段从网格上收集速度，并执行 **PIC/FLIP 混合**。PIC 方法直接使用网格插值速度，数值稳定但耗散性强；FLIP 方法使用网格速度的增量（$\Delta \mathbf{u} = \mathbf{u}^{n+1} - \mathbf{u}^n$）更新粒子速度，保留更多细节但可能引入噪声。通过参数 `flip_ratio` 可以在两者之间平滑插值：

$$\mathbf{v}_p^{n+1} = (1 - \alpha) \cdot \mathbf{v}_{\text{PIC}} + \alpha \cdot \mathbf{v}_{\text{FLIP}}$$

其中 $\alpha$ 即为 `flip_ratio`，默认值 0.95（FLIP95）在稳定性和细节保留之间取得了良好的平衡。

```python
# flip.py — G2P 传输
new_vel = self._interp_vel(grid_u, grid_v, grid_w, px, py, pz)
old_grid_vel = self._interp_vel(grid_u_old, grid_v_old, grid_w_old, px, py, pz)
delta = new_vel - old_grid_vel
pic_vel = new_vel
flip_vel = old_vel + delta
self.vel[p] = (1.0 - flip_ratio) * pic_vel + flip_ratio * flip_vel
```

### 2.3 不可压缩求解

不可压缩性是流体仿真的核心约束 $\nabla \cdot \mathbf{u} = 0$。本实验实现了两种求解器：**Red-Black Gauss-Seidel（GS）** 和 **共轭梯度法（CG）**。

GS 求解器采用红黑排序实现并行化：将网格单元按 $(i+j+k) \mod 2$ 分为红黑两组，同组单元之间无数据依赖，可以安全并行更新。每次迭代中，单元的散度被计算为六个相邻面上的速度差之和，随后通过超松弛因子 $\omega$ 进行校正：

$$\text{correction} = \omega \cdot \frac{\nabla \cdot \mathbf{u}}{s}$$

其中 $s$ 是非固体邻居的数量。`compensate_drift` 选项将粒子密度偏差纳入散度修正，缓解长时间模拟中的体积损失问题。

CG 求解器直接求解压力 Poisson 方程 $\mathbf{L}\mathbf{p} = \mathbf{b}$，其中 $\mathbf{L}$ 是离散 Laplacian 算子，$\mathbf{b}$ 是速度散度向量。相比 GS 的固定迭代次数，CG 在理论上保证在 $N$ 步内收敛（$N$ 为未知数个数），且每步仅需一次矩阵-向量乘积和两次内积运算。

```python
# core.py — CG 求解核心循环
for _ in range(num_iters):
    self._cg_compute_Ap()          # 计算 Ap = L * p
    pAp = self._cg_compute_pAp()   # 计算 p^T A p
    alpha = rr / pAp               # 步长
    self._cg_update(alpha)          # 更新 x 和 r
    rr_new = self._cg_compute_rdotr()
    beta = rr_new / rr             # 共轭方向系数
    self._cg_update_p(beta)
```

求解完成后，压力梯度被应用到交错网格速度上，消除散度并保持不可压缩性。

### 2.4 粒子碰撞与分离

`integrate_and_collide` 核函数将粒子积分、域边界碰撞和障碍物碰撞合并为一次遍历。对于球形障碍物，碰撞检测计算粒子到球心的距离：若距离小于球半径，粒子被推至球面，法向速度分量被消除。

```python
# core.py — 障碍物碰撞
diff = self.pos[i] - obs_pos
dist = diff.norm()
if dist < obs_r and dist > 1e-8:
    n = diff / dist
    self.pos[i] = obs_pos + n * obs_r  # 推至表面
    rel_vel = self.vel[i] - obs_vel
    vn = rel_vel.dot(n)
    if vn < 0:
        self.vel[i] -= n * vn  # 消除法向速度
```

`push_particles_apart` 使用基于网格的空间哈希来加速邻近粒子搜索。每个粒子被注册到其所在的网格单元中，分离遍历时仅检查相邻 27 个单元内的粒子。当两粒子距离小于阈值时，施加位置修正以保持最小间距，防止粒子聚集。

## 3 Bonus 实现

### 3.1 B1：可视化与交互增强

粒子着色支持三种模式：**速度**模式根据 $|\mathbf{v}|$ 将粒子从蓝色渐变为红色，直观展示流场速度分布；**密度**模式根据单元内粒子数量着色，反映压缩和稀疏区域；**统一**模式使用固定蓝色，用于观察整体形态。

交互方面，实现了**鼠标可控球形障碍物**的拖拽功能。左键点击时，通过射线-球体相交测试判断是否选中障碍物：

$$t = \frac{-b - \sqrt{b^2 - 4ac}}{2a}, \quad a = \mathbf{d} \cdot \mathbf{d}, \quad b = 2(\mathbf{o} - \mathbf{c}) \cdot \mathbf{d}, \quad c = (\mathbf{o} - \mathbf{c}) \cdot (\mathbf{o} - \mathbf{c}) - r^2$$

若选中，粒子沿 $y = \text{const}$ 平面拖动障碍物，并估计障碍物速度传递给流体。

### 3.2 B2：共轭梯度压力求解器

CG 求解器在 core.py 中实现，包含残差初始化、矩阵-向量乘积、内积计算和参数更新四个核函数。离散 Laplacian 算子将每个流体单元的压力与其非固体邻居的压力差求和，形成标准的 7 点模板。

GUI 中提供了 GS/CG 切换按钮，允许在运行时动态切换求解器并观察效果差异。

### 3.3 B3：欧拉流体对比

欧拉仿真器在纯网格上运行，采用 **Semi-Lagrangian 对流** 方法。对于每个速度分量面，从当前位置沿速度场反向追踪一个时间步长，然后在追踪终点进行三线性插值获得新值：

```python
# eulerian.py — Semi-Lagrangian 对流
x, y, z = i * dx, (j + 0.5) * dx, (k + 0.5) * dx
vel = self._interp_vel_self(x, y, z, dx)
px, py, pz = x - dt * vel[0], y - dt * vel[1], z - dt * vel[2]
self.grid_u[i, j, k] = self._interp_u(px, py, pz, dx)
```

这种方法无条件稳定（无 CFL 限制），但引入了数值耗散，导致细节丢失。相比之下，FLIP 的粒子平流在保留涡旋和小尺度结构方面表现显著更好。

<!-- TODO: 添加 PIC/FLIP 与欧拉方法的仿真截图对比 -->

### 3.4 B4：APIC 方法

APIC（Affine Particle-In-Cell）方法通过在每个粒子上维护一个 $3 \times 3$ 的仿射矩阵 $\mathbf{C}_p$，在传输过程中保留局部速度梯度信息。P2G 阶段，散射到网格上的速度包含仿射贡献：

$$v_{\text{affine}} = v_p + \mathbf{C}_p \cdot (\mathbf{x}_{\text{face}} - \mathbf{x}_p)$$

```python
# apic.py — APIC P2G 仿射项
affine_vx = vx + C[0,0] * dpx + C[0,1] * dpy + C[0,2] * dpz
self.grid_u[ni, nj, nk] += w * affine_vx
```

G2P 阶段，新的仿射矩阵通过交错网格上的中心差分计算速度梯度得到。APIC 相比 FLIP 的关键优势在于它**严格保持角动量**，避免了 FLIP 中常见的体积损失和粒子聚集问题。

GUI 中提供 FLIP/APIC 切换按钮，结合 `flipRatio` 滑块，可以在同一场景中实时对比 PIC（$\alpha=0$）、FLIP（$\alpha=1$）、FLIP95（$\alpha=0.95$）和 APIC 的行为差异。

<!-- TODO: 添加 PIC/FLIP/APIC 对比截图 -->

### 3.5 B5：表面重建

表面重建模块从 FLIP 粒子位置出发，通过两步过程提取三角形网格。首先，将粒子密度通过**高斯核函数**散射到一个比仿真网格更精细的密度场上：

$$\rho(\mathbf{x}) = \sum_p \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{|\mathbf{x} - \mathbf{x}_p|^2}{2\sigma^2}\right)$$

然后，密度场被归一化到 $[0, 1]$ 范围，使用标准 **Marching Cubes** 算法在给定阈值处提取等值面。Marching Cubes 遍历每个体素单元，根据 8 个角点值是否超过阈值确定 256 种情况之一，查表生成三角形面片。

```python
# surface.py — Marching Cubes 等值面提取
case_idx = 0
for c in range(8):
    if vals[c] >= threshold:
        case_idx |= (1 << c)
# 查表获取三角面片
```

重建后的网格在 Taichi GGUI 中以线框模式渲染，与粒子可视化叠加显示。

<!-- TODO: 添加表面重建效果截图 -->

## 4 Demo 展示

<!-- TODO: 填充各 Demo 的截图和说明 -->

**Basic / B1 / B2（基础 FLIP 仿真）**

运行命令：`uv run lab2` 或 `uv run lab2 --demo b1 --debug`

<!-- 截图位置 -->

**B3（欧拉流体）**

运行命令：`uv run lab2 --demo b3`

<!-- 截图位置 -->

**B4（APIC 对比）**

运行命令：`uv run lab2 --demo b4`，或在 Basic Demo 中使用 Toggle FLIP/APIC 按钮

<!-- 截图位置 -->

**B5（表面重建）**

运行命令：`uv run lab2 --demo b5`

<!-- 截图位置 -->

## 5 交互方式

仿真支持以下交互操作：

- **时间步长 / flipRatio / 重力**：通过滑块实时调整
- **暂停/继续**：点击按钮切换
- **场景切换**：Dam Break / Drop / Double Dam
- **分辨率切换**：Low (16) / Med (24) / High (32)
- **障碍物**：选择预设障碍物配置（1-3 个球体）
- **拖动障碍物**：左键点击障碍物后拖动
- **旋转视角**：右键拖动
- **平移视角**：左键拖动（未选中障碍物时）
- **缩放**：按 R/F 键
- **着色模式**：Speed / Density / Uniform
- **求解器切换**：GS / CG
- **方法切换**：FLIP / APIC
- **调试模式**：`--debug` 参数启动或点击 Debug 按钮

## 6 总结

本实验从零实现了一个完整的 3D FLIP 流体仿真系统，涵盖了粒子-网格速度传输、交错网格上的压力投影、粒子碰撞处理等核心模块。在基础功能之上，通过实现 APIC、CG 求解器、欧拉对比和表面重建，深入探索了不同方法在数值精度、动量守恒和视觉效果上的差异。
