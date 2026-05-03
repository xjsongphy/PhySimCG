# Lab 2 作业报告

## 基本信息
- 学生学号: (请填写)
- 姓名: (请填写)

---

## 一、核心代码思路

### 1. FLIP 算法框架

本项目实现了一个基于 FLIP（Fluid-In-Cell）的流体仿真框架，结合了拉格朗日粒子法和欧拉网格法的优势。

**核心流程**：
```
初始化 → 重力积分 → 粒子碰撞处理 → 粒子分离 → 粒子-网格速度传递 → 密度计算 →
压力投影（Gauss-Seidel/CG） → 网格-粒子速度传递 → 循环
```

### 2. 粒子-网格（P2G）与网格-粒子（G2P）速度传递

使用**交错网格（MAC网格）**存储速度场：
- u: x-方向速度在 x-面上
- v: y-方向速度在 y-面上
- w: z-方向速度在 z-面上

**P2G（Particle to Grid）**：
```python
for p in particles:
    # 找到粒子所在的网格单元
    ci, cj, ck = cell_index(p)
    # 将粒子速度加到所有8个角点的网格速度上（中心点）
    grid_u[ci, cj, ck] += p.vel[0] / 8
    grid_v[ci, cj, ck] += p.vel[1] / 8
    grid_w[ci, cj, ck] += p.vel[2] / 8
```

**G2P（Grid to Particle）**：
```python
for p in particles:
    # 从粒子位置插值得到网格速度
    p.vel[0] = interp(grid_u, p.pos)
    p.vel[1] = interp(grid_v, p.pos)
    p.vel[2] = interp(grid_w, p.pos)
```

### 3. 压力投影（Gauss-Seidel）

求解泊松方程：
```
∇²p = (∇·u) / Δt
```

使用 Gauss-Seidel 迭代法：
```python
for _ in range(num_pressure_iters):
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                # 计算散度
                div = (u[i+1] - u[i] + v[j+1] - v[j] + w[k+1] - w[k]) / dx
                # 压力修正
                p[i,j,k] = (div + p[i+1] + p[i-1] + p[i,j+1] + p[i,j-1] + p[i,j,k+1] + p[i,j,k-1]) / 6
                # 更新速度场
                u[i,j,k] -= p[i,j,k] / dx
                v[i,j,k] -= p[i,j,k] / dx
                w[i,j,k] -= p[i,j,k] / dx
```

### 4. B5 表面重建与渲染

**步骤**：
1. **密度场构建**：使用高斯核将粒子密度"涂抹"到网格上
2. **SDF 计算**：计算每个网格单元的 Signed Distance Field
3. **Marching Cubes 提取网格**：使用标准 256 情况表提取等值面
4. **网格渲染**：在 UI 中渲染生成的三角网格

```python
def build_surface_mesh(pos_field, nx, ny, nz, dx, threshold=0.5, mc_res=1):
    # 1. 构建密度场
    density = gaussian_splat(pos_field)

    # 2. Marching Cubes 提取网格
    for i, j, k in grid:
        case_idx = compute_case(density[i,j,k])
        triangles = tri_table[case_idx]
        # 生成顶点和三角形

    return vertices, triangles
```

---

## 二、实现的交互方法

### 1. 基础交互

通过 GUI 窗口提供以下交互：

| 控件 | 功能 | 范围 |
|------|------|------|
| `dt` 滑块 | 调整时间步长 | 0.001 ~ 0.03 |
| `flipRatio` 滑块 | 切换 PIC/FLIP 方法 | 0.0 ~ 1.0 |
| `gravity` 滑块 | 调整重力加速度 | -20.0 ~ 0.0 |
| `Pause/Resume` 按钮 | 暂停/继续仿真 |
| `Resolution` 菜单 | 选择网格分辨率 |

### 2. 高级交互

- **Solver 切换**：在 Gauss-Seidel 和 CG 之间切换
- **Method 切换**：在 FLIP 和 APIC 之间切换
- **Scene 切换**：支持多种场景（Dam Break、Drop、Double Dam 等）

### 3. 3D 交互

- 鼠标左键拖动：旋转相机视角
- 滚轮：缩放视图

---

## 三、Demo 展示

### Demo 1: FLIP 流体仿真

**场景**：Dam Break（水槽破裂）

**特点**：
- 粒子数量：约 10,000
- 网格分辨率：24×48×24
- 颜色模式：按速度大小染色
- 障碍物：可选圆柱形障碍物

**操作**：
1. 运行 `uv run lab2 --demo basic`
2. 使用滑块调整 `flipRatio`，观察 PIC→FLIP 平滑过渡
3. 调整 `dt` 观察数值稳定性

### Demo 2: B5 表面重建

**场景**：Dam Break with Obstacle（带障碍物的水槽）

**特点**：
- 表面重建阈值：0.3
- MC 网格细化倍数：2
- 每 10 帧重建一次

**操作**：
1. 运行 `uv run lab2 --demo b5`
2. 观察粒子水柱流下并在障碍物处形成复杂表面
3. 表面网格实时渲染，可切换线框模式

### Demo 3: B3 Eulerian 流体

**场景**：Grid-based Eulerian fluid

**特点**：
- Semi-Lagrangian 对流
- 密度场传播
- 按密度颜色显示

**操作**：
1. 运行 `uv run lab2 --demo b3`
2. 切换到 "Density" 颜色模式
3. 观察密度波在网格上的传播

---

## 四、关键技术点

### 1. 空间哈希加速粒子分离

使用网格哈希表快速查找近邻粒子：
```python
def push_particles_apart(self, num_iters):
    for _ in range(num_iters):
        grid.fill(0)
        for p in particles:
            grid[p] = particles_in_cell
        for p in particles:
            for neighbor in grid[p]:
                separate(p, neighbor)
```

### 2. IDP 体积守恒补偿

参考 Implicit Density Projection for Volume Conserving Liquids，通过位置修正补偿 FLIP 的体积损失。

### 3. Marching Cubes 实现

- 使用标准 256 情况表
- 边连接表：12 条边
- 三角形表：每种情况最多 15 个三角形

---

## 五、实验结果与对比

### FLIP vs PIC

| 方法 | 数值耗散 | 动量保持 | 计算复杂度 |
|------|----------|----------|------------|
| PIC | 高（粒子拖尾） | 低 | 高（频繁 G2P） |
| FLIP | 低 | 高 | 中（较少 G2P） |
| FLIP95（本实现） | 低 | 中 | 中 |

### Gauss-Seidel vs CG

| 方法 | 收敛速度 | 内存使用 | 稳定性 |
|------|----------|----------|--------|
| Gauss-Seidel | 慢（需要 ~100 iters） | 低 | 稳定 |
| CG（可选） | 快（~10 iters） | 中 | 稳定 |

---

## 六、总结

本 Lab 2 实现了一个完整的 FLIP 流体仿真框架，包含：
1. ✅ FLIP 算法核心流程（P2G、G2P、压力投影）
2. ✅ 交互式 GUI（时间步长、flipRatio、重力等）
3. ✅ 多种可视化模式（速度、密度）
4. ✅ B5 表面重建与渲染（Marching Cubes）
5. ✅ B3 欧拉流体对比
6. ✅ B4 APIC 实现

项目代码结构清晰，支持多种 Demo 切换，便于理解和实验。

---

## 七、参考资料

- Ten Minute Physics (Lecture 18)
- Implicit Density Projection for Volume Conserving Liquids (SCA/TVCG 2019)
- Marching Cubes: A 3D Surface Construction Algorithm
- Taichi 官方文档与 Gallery
