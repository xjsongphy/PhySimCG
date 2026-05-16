# 有限元仿真讲义

**作者**: 计算机图形学课程组 | **日期**: 2026年05月15日 | **状态**: 定稿

## 引言

有限元方法（Finite Element Method, FEM）是计算机图形学中模拟变形物体的重要技术。从电影特效中的角色动画到视频游戏中的软体模拟，从工程仿真到科学计算，FEM技术都有着广泛的应用。本讲义将系统介绍基于有限元方法的变形仿真理论基础和主要算法，帮助读者理解FEM的核心概念和实现技术。

有限元仿真的核心思想是将连续的变形体离散化为有限个简单几何单元的集合。在三维空间中，这些单元通常是四面体；在二维流形嵌入三维空间的情况下，如布料模拟，这些单元是三角形。每个单元内部的变形场被假设为线性变化，从而使得我们能够通过计算节点处的力和位移来模拟整个物体的变形行为。

## 符号与物理量

### 0.1 基本约定

本讲义采用以下书写约定：
- 标量使用普通字体（如$J$、$\rho$）
- 向量使用粗体小写字母（如$\mathbf{x}$、$\mathbf{v}$、$\mathbf{f}$）
- 二阶张量使用粗体大写字母（如$\mathbf{F}$、$\mathbf{G}$、$\mathbf{P}$）
- 矩阵使用粗体大写字母（如$D_m$、$D_s$、$H$）

### 0.2 几何量

| 符号 | 名称 | 定义 | 物理意义 |
|------|------|------|----------|
| $\mathbf{X}$ | 参考位置 | 未变形状态下物质点的位置 | 描述物质点在初始时刻的空间位置 |
| $\mathbf{x}$ | 当前位置 | 变形后物质点的位置 | 描述物质点在当前时刻的空间位置 |
| $\boldsymbol{\varphi}$ | 变形函数 | $\mathbf{x} = \boldsymbol{\varphi}(\mathbf{X})$ | 从参考构型到当前构型的映射关系 |

### 0.3 变形度量

| 符号 | 名称 | 定义 | 物理意义 |
|------|------|------|----------|
| $\mathbf{F}$ | 变形梯度 | $\mathbf{F} = \frac{\partial \mathbf{x}}{\partial \mathbf{X}} = \nabla_{\mathbf{X}} \boldsymbol{\varphi}(\mathbf{X})$ | 描述局部线性变形，包含拉伸、旋转和剪切 |
| $J$ | 体积比 | $J = \det(\mathbf{F})$ | 变形后体积与参考体积的比值 |
| $\mathbf{G}$ | Green-Lagrange应变张量 | $\mathbf{G} = \frac{1}{2}(\mathbf{F}^T\mathbf{F} - \mathbf{I})$ | 度量应变，自动排除刚体旋转的影响 |
| $I_C$ | 右Cauchy-Green张量第一不变量 | $I_C = \|\mathbf{F}\|_F^2 = \text{tr}(\mathbf{F}^T\mathbf{F})$ | 变形梯度的平方长度，用于Neo-Hookean模型 |

**变形度量的层次关系：**
$$
\mathbf{F} \xrightarrow{\mathbf{F}^T\mathbf{F}} \mathbf{C} \xrightarrow{\frac{1}{2}(\mathbf{C}-\mathbf{I})} \mathbf{G}
$$
其中$\mathbf{C} = \mathbf{F}^T\mathbf{F}$是右Cauchy-Green变形张量。

### 0.4 应力与能量

| 符号 | 名称 | 定义 | 物理意义 |
|------|------|------|----------|
| $\Psi(\mathbf{F})$ / $W(\mathbf{G})$ | 应变能密度函数 | 单位参考体积内的弹性能量 | 描述材料的本构关系 |
| $E$ | 总弹性能量 | $E = \int_{\Omega_0} \Psi(\mathbf{F}) d\mathbf{X} = \sum_e V_e^{ref} \Psi(\mathbf{F}_e)$ | 整个物体的弹性势能 |
| $\mathbf{P}$ | First Piola-Kirchhoff应力张量 | $\mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}$ | 参考构型下的应力，能量对变形梯度的导数 |
| $\mathbf{S}$ | Second Piola-Kirchhoff应力张量 | $\mathbf{S} = \frac{\partial W}{\partial \mathbf{G}}$ | 参考构型下的应力，能量对Green-Lagrange应变的导数 |
| $\boldsymbol{\sigma}$ | 柯西应力张量 | 当前构型下的真实应力 | 描述当前构型中单位面积上的表面力 |
| $\mathbf{f}_i$ | 节点力 | $\mathbf{f}_i = -\left(\frac{\partial E}{\partial \mathbf{x}_i}\right)^T$ | 节点$i$上的内力，能量对节点位置的负梯度 |

**应力张量的转换关系：**
$$
\mathbf{P} = \mathbf{F}\mathbf{S}, \quad \boldsymbol{\sigma} = \frac{1}{J}\mathbf{P}\mathbf{F}^T
$$

**能量、应力与力的关系：**
$$
\Psi \xrightarrow{\frac{\partial}{\partial \mathbf{F}}} \mathbf{P} \xrightarrow{\text{离散化}} \mathbf{f}_i
$$

### 0.5 运动学量

| 符号 | 名称 | 定义 | 物理意义 |
|------|------|------|----------|
| $\mathbf{v}$ | 速度场 | 物质点的运动速度 | 描述物质点的运动状态 |
| $\rho$ | 密度 | 单位体积内的质量 | 描述物质的质量分布 |
| $\mathbf{g}$ | 体力加速度 | 如重力加速度$\mathbf{g} = (0, -9.81, 0)$ | 描述单位质量受到的体力 |

### 0.6 材料参数

| 符号 | 名称 | 定义 | 物理意义 |
|------|------|------|----------|
| $\lambda, \mu$ | Lamé参数 | 描述材料弹性性质的基本参数 | $\mu$是剪切模量，$\lambda$与体积压缩相关 |
| $Y$ | 杨氏模量 | 描述材料抵抗拉伸变形的能力 | 数值越大，材料越硬 |
| $\nu$ | 泊松比 | 描述拉伸时横向收缩的程度 | 取值范围$(-1, 0.5)$，接近0.5表示接近不可压缩 |

**参数转换公式：**

从Lamé参数到杨氏模量和泊松比：
$$
Y = \frac{\mu(3\lambda + 2\mu)}{\lambda + \mu}, \quad \nu = \frac{\lambda}{2(\lambda + \mu)}
$$

从杨氏模量和泊松比到Lamé参数：
$$
\mu = \frac{Y}{2(1 + \nu)}, \quad \lambda = \frac{Y\nu}{(1 + \nu)(1 - 2\nu)}
$$

### 0.7 离散化相关量

| 符号 | 名称 | 定义 | 物理意义 |
|------|------|------|----------|
| $D_m$ | 参考构型边矩阵 | $D_m = [\mathbf{X}_1-\mathbf{X}_0 \quad \mathbf{X}_2-\mathbf{X}_0 \quad \mathbf{X}_3-\mathbf{X}_0]$ | 四面体参考构型中的边向量矩阵 |
| $D_s$ | 当前构型边矩阵 | $D_s = [\mathbf{x}_1-\mathbf{x}_0 \quad \mathbf{x}_2-\mathbf{x}_0 \quad \mathbf{x}_3-\mathbf{x}_0]$ | 四面体当前构型中的边向量矩阵 |
| $V_e^{ref}$ | 单元参考体积 | 单元在参考构型中的体积 | 用于计算单元的应变能 |
| $H$ | 力矩阵 | $H = -V^{ref}\mathbf{P} D_m^{-T}$ | 矩阵的各列给出节点的内力 |

**离散化变形梯度的计算：**
$$
\mathbf{F} = D_s D_m^{-1}
$$

### 0.8 时间积分量

| 符号 | 名称 | 定义 | 物理意义 |
|------|------|------|----------|
| $\Delta t$ | 时间步长 | 仿真中每一步的时间间隔 | 影响数值稳定性和计算效率 |
| $\mathbf{y}$ | 显式预测位置 | $\mathbf{y} = \mathbf{x}_n + \mathbf{v}_n\Delta t$ | 隐式积分中的初始猜测位置 |

### 0.9 物理量的量纲

为了帮助理解各物理量的相对大小，以下是一些典型量级：

| 物理量 | 典型数值范围 | 单位（SI） |
|--------|--------------|------------|
| 密度$\rho$ | $1000$（水）~ $8000$（钢） | $\text{kg}/\text{m}^3$ |
| 杨氏模量$Y$ | $10^3$（软组织）~ $10^{11}$（钢） | $\text{Pa}$ |
| 时间步长$\Delta t$ | $10^{-5}$ ~ $10^{-3}$ | $\text{s}$ |
| 节点力$\mathbf{f}$ | 取决于变形程度和单元大小 | $\text{N}$ |

## 第一章 连续介质力学基础

### 1.1 守恒定律概述

连续介质力学建立在几个基本守恒定律之上。这些守恒定律描述了连续物质在运动过程中必须遵循的基本规律，为有限元仿真提供了理论基础。在图形学应用中，我们主要关注质量守恒、动量守恒和角动量守恒这三个基本定律。

### 1.2 质量守恒

质量守恒定律要求物质的质量在运动过程中保持不变。从固定空间控制体的积分形式出发，对于固定区域$\Omega$，区域内质量的变化率等于穿过边界的净流入质量通量：

$$
\frac{d}{dt}\int_\Omega \rho\,dV = -\int_{\partial\Omega} \rho \mathbf{v} \cdot \mathbf{n}\, dA
$$

利用散度定理将面积分转化为体积分，就得到连续性方程的微分形式：

$$
\frac{\partial\rho}{\partial t} + \nabla \cdot (\rho\mathbf{v}) = 0
$$

其中$\rho$表示密度，$\mathbf{v}$表示速度场。这个方程适用于所有连续介质，包括可压缩和不可压缩的情形。

对于不可压缩材料，密度随体元的物质导数为零：$\frac{D\rho}{Dt} = 0$。将这一条件代入连续性方程，由于$\frac{D\rho}{Dt} = \frac{\partial\rho}{\partial t} + \mathbf{v} \cdot \nabla\rho = 0$，结合连续性方程消去密度的时间导数和空间梯度项，最终得到：

$$
\nabla \cdot \mathbf{v} = 0
$$

这个不可压缩条件确保了材料的体积在运动过程中保持不变。

### 1.3 线性动量平衡

线性动量平衡定律描述了物体的运动如何响应作用在其上的力。柯西动量方程是适用于所有连续介质的普适方程：

$$
\rho \frac{D\mathbf{v}}{Dt} = \nabla \cdot \boldsymbol{\sigma} + \rho\mathbf{g}
$$

其中$\boldsymbol{\sigma}$是柯西应力张量，它统一描述了所有表面力的贡献。$\mathbf{g}$是体力加速度，如重力加速度。

应力张量$\boldsymbol{\sigma}$是一个对称的二阶张量，其对称性来自于角动量守恒的要求。应力张量的物理意义可以通过表面的受力来理解：对于一个单位法向量为$\mathbf{n}$的表面，其上的牵引力$\mathbf{t}$（单位面积上的力）为：

$$
\mathbf{t} = \boldsymbol{\sigma}\mathbf{n}
$$

这个关系式表明，给定任意表面的法向量，应力张量能够唯一确定该表面上的牵引力。

### 1.4 角动量平衡

在经典连续介质理论中，如果忽略体偶力和偶应力，局部角动量守恒要求柯西应力张量的反对称部分为零，因此有：

$$
\boldsymbol{\sigma} = \boldsymbol{\sigma}^T
$$

直观地说，如果应力张量存在反对称部分，那么无穷小体元会受到非零内力矩，这将违反局部角动量平衡。这个条件确保了材料内部的力矩平衡，即任意小体积元在仅受内力作用时不会产生自发旋转。

## 第二章 变形度量

### 2.1 变形场的概念

在开始讨论有限元方法之前，我们需要建立描述物体变形的数学框架。考虑一个物体在参考构型（未变形状态）下的位置$\mathbf{X}$和在当前构型（变形状态）下的位置$\mathbf{x}$。这两个位置之间的映射关系由变形函数$\boldsymbol{\varphi}$给出：

$$
\mathbf{x} = \boldsymbol{\varphi}(\mathbf{X})
$$

这个函数描述了物体中每个物质点从参考位置到当前位置的映射。对于一般的变形，这个函数可能是高度非线性的。

### 2.2 变形梯度

变形梯度$\mathbf{F}$是描述局部变形的核心量。它定义为变形函数关于参考位置的梯度：

$$
\mathbf{F} = \frac{\partial \mathbf{x}}{\partial \mathbf{X}} = \nabla_{\mathbf{X}} \boldsymbol{\varphi}(\mathbf{X})
$$

这个定义的物理意义可以通过泰勒展开来理解。在参考构型中某点$\mathbf{X}_k$附近，变形后的位置可以近似为：

$$
\mathbf{x} \approx \boldsymbol{\varphi}(\mathbf{X}_k) + \mathbf{F}(\mathbf{X} - \mathbf{X}_k)
$$

这个近似表明，变形梯度$\mathbf{F}$将参考构型中的无穷小向量$\mathbf{X} - \mathbf{X}_k$线性映射到当前构型中的对应向量。因此，变形梯度完全描述了局部的拉伸、旋转和剪切变形。

这种线性近似的思想为有限元离散化提供了理论基础。在有限元方法中，我们将连续的变形场离散化为有限个简单几何单元的集合，对于三维软体通常使用四面体单元，对于布料等二维流形使用三角形单元。在每个单元内部，我们假设变形场是线性的，这意味着变形梯度在单元内是常数。

现在我们来考虑如何在离散化后计算变形梯度。对于四面体单元，如果我们知道四个顶点的参考位置$\mathbf{X}_0, \mathbf{X}_1, \mathbf{X}_2, \mathbf{X}_3$和当前位置$\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3$，就可以建立变形关系。根据线性假设，单元内任意向量都满足变形梯度的映射关系：

$$
\mathbf{F}[\mathbf{X}_1 - \mathbf{X}_0 \quad \mathbf{X}_2 - \mathbf{X}_0 \quad \mathbf{X}_3 - \mathbf{X}_0] = [\mathbf{x}_1 - \mathbf{x}_0 \quad \mathbf{x}_2 - \mathbf{x}_0 \quad \mathbf{x}_3 - \mathbf{x}_0]
$$

这个关系式告诉我们，变形梯度将参考构型中的边向量映射到当前构型中的对应边向量。由于我们选择了节点0作为参考点，三组边向量构成了一个线性方程组。通过求解这个方程组，我们得到变形梯度的显式表达式：

$$
\mathbf{F} = [\mathbf{x}_1 - \mathbf{x}_0 \quad \mathbf{x}_2 - \mathbf{x}_0 \quad \mathbf{x}_3 - \mathbf{x}_0][\mathbf{X}_1 - \mathbf{X}_0 \quad \mathbf{X}_2 - \mathbf{X}_0 \quad \mathbf{X}_3 - \mathbf{X}_0]^{-1}
$$

这个公式的几何意义非常清晰。右边的第一个矩阵由当前构型中的边向量组成，第二个矩阵由参考构型中的边向量组成，它们的比值就是从参考构型到当前构型的线性变换。这个线性变换既包含了刚体的旋转和平移，也包含了形状的拉伸、压缩和剪切，完整地描述了单元的局部变形状态。

### 2.3 Green-Lagrange应变张量

变形梯度包含了刚体旋转的信息，但在计算应变能时，我们希望排除纯旋转的影响。Green-Lagrange应变张量$\mathbf{G}$通过以下定义实现了这一目标：

$$
\mathbf{G} = \frac{1}{2}(\mathbf{F}^T\mathbf{F} - \mathbf{I})
$$

这个定义的优点是它对于刚体旋转保持不变。对于任何旋转矩阵$\mathbf{R}$，如果$\mathbf{F} = \mathbf{R}$，则$\mathbf{G} = \mathbf{0}$。这符合物理直觉：纯旋转不应该产生应变。

为了理解Green-Lagrange应变的物理意义，考虑一个在参考构型中长度为$l_0 = \|\mathbf{X}_{ba}\|$的线段，变形后长度为$l = \|\mathbf{x}_{ba}\|$。利用变形梯度的定义$\mathbf{x}_{ba} = \mathbf{F}\mathbf{X}_{ba}$，我们可以计算长度的相对变化：

$$
\frac{l^2 - l_0^2}{l_0^2} = \frac{\|\mathbf{x}_{ba}\|^2 - \|\mathbf{X}_{ba}\|^2}{\|\mathbf{X}_{ba}\|^2} = \frac{\mathbf{X}_{ba}^T(\mathbf{F}^T\mathbf{F} - \mathbf{I})\mathbf{X}_{ba}}{\|\mathbf{X}_{ba}\|^2} = 2\mathbf{n}^T\mathbf{G}\mathbf{n}
$$

其中$\mathbf{n} = \mathbf{X}_{ba}/\|\mathbf{X}_{ba}\|$是参考构型中线段的单位方向向量。这个公式表明，$2\mathbf{n}^T\mathbf{G}\mathbf{n}$ 描述的是该方向上线段的平方长度相对变化，而不是长度本身的相对变化。对于小变形情形，平方长度变化与普通长度变化近似成正比，因此 Green-Lagrange 应变可以看作有限变形下对线性应变的推广。

## 第三章 弹性能量与应力

### 3.1 应变能密度函数

在弹性理论中，材料的力学行为通过应变能密度函数来描述。这个函数$\Psi(\mathbf{F})$或$W(\mathbf{G})$给出了单位参考体积内存储的弹性能量。对于超弹性材料，应力可以通过应变能密度函数的导数得到，这保证了加载和卸载过程中的能量守恒。

物体的总弹性能量是应变能密度在参考构型上的积分：

$$
E = \int_{\Omega_0} \Psi(\mathbf{F}) d\mathbf{X} = \int_{\Omega_0} W(\mathbf{G}) d\mathbf{X}
$$

在有限元离散化中，由于我们假设每个单元内部的变形梯度是常数，应变能密度在单元内也是常数。因此，总能量可以简化为各单元能量的求和：

$$
E = \sum_{e} V_e^{ref} \Psi(\mathbf{F}_e)
$$

其中$V_e^{ref}$是单元$e$在参考构型中的体积，$\mathbf{F}_e$是该单元的变形梯度。

### 3.2 First Piola-Kirchhoff应力张量

应力张量描述了材料内部的受力状态。在有限变形理论中，有几种不同的应力张量定义，它们的区别在于使用参考构型还是当前构型来度量面积和力。First Piola-Kirchhoff应力张量$\mathbf{P}$定义为应变能密度函数关于变形梯度的导数：

$$
\mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}
$$

这个定义的物理意义可以通过虚功原理来理解。如果我们将物体中的某节点产生一个虚位移$\delta\mathbf{x}_i$，则相应的能量变化为：

$$
\delta E = \frac{\partial E}{\partial \mathbf{x}_i} \cdot \delta\mathbf{x}_i = -\mathbf{f}_i \cdot \delta\mathbf{x}_i
$$

其中$\mathbf{f}_i$是该节点上的内力。因此，节点力为：

$$
\mathbf{f}_i = -\left(\frac{\partial E}{\partial \mathbf{x}_i}\right)^T
$$

对于单个单元，能量关于节点位置的导数可以通过链式法则计算：

$$
\frac{\partial \Psi}{\partial \mathbf{x}_i} = \frac{\partial \Psi}{\partial \mathbf{F}} : \frac{\partial \mathbf{F}}{\partial \mathbf{x}_i} = \mathbf{P} : \frac{\partial \mathbf{F}}{\partial \mathbf{x}_i}
$$

其中":"表示双点积（张量收缩）。这个公式建立了First Piola-Kirchhoff应力张量与节点力之间的关系。

### 3.3 从应力计算四面体节点力

对于一个线性四面体单元，记参考构型中的边矩阵为：

$$
D_m = [\mathbf{X}_1-\mathbf{X}_0 \quad \mathbf{X}_2-\mathbf{X}_0 \quad \mathbf{X}_3-\mathbf{X}_0]
$$

当前构型中的边矩阵为：

$$
D_s = [\mathbf{x}_1-\mathbf{x}_0 \quad \mathbf{x}_2-\mathbf{x}_0 \quad \mathbf{x}_3-\mathbf{x}_0]
$$

则单元内的变形梯度为：

$$
\mathbf{F} = D_s D_m^{-1}
$$

设单元参考体积为$V^{ref}$，应变为：

$$
E_e = V^{ref}\Psi(\mathbf{F})
$$

由节点力定义：

$$
\mathbf{f}_i = -\frac{\partial E_e}{\partial \mathbf{x}_i}
$$

我们需要计算能量对节点位置的导数。通过链式法则和变形梯度的表达式，可以推导出一个力矩阵$H$，它的三列分别对应节点1、2、3的内力。这个矩阵的推导利用了变形梯度对节点位置的依赖关系——只有$D_s$矩阵依赖于当前节点位置，而$D_m$矩阵在参考构型下是常数。

具体的推导结果为：

$$
H = -V^{ref}\mathbf{P} D_m^{-T}
$$

其中：

$$
\mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}
$$

是 First Piola-Kirchhoff 应力张量。矩阵 $H$ 的三列分别给出节点 1、2、3 的内力：

$$
\mathbf{f}_1 = H_{:,1}, \quad \mathbf{f}_2 = H_{:,2}, \quad \mathbf{f}_3 = H_{:,3}
$$

节点 0 的内力由力平衡得到：

$$
\mathbf{f}_0 = -\mathbf{f}_1 - \mathbf{f}_2 - \mathbf{f}_3
$$

## 第四章 超弹性材料模型

### 4.1 材料模型的分类

在有限元仿真中，材料模型的选择对仿真结果有着重要影响。超弹性材料模型是指那些能够从应变能密度函数导出应力-应变关系的材料。这类材料的主要特点是应力状态只取决于当前的变形状态，而与变形历史无关。

图形学中常用的超弹性材料模型包括St. Venant-Kirchhoff模型、Neo-Hookean模型和Corotated模型。这些模型在计算复杂度和物理准确性之间有着不同的权衡。

### 4.2 St. Venant-Kirchhoff模型

St. Venant-Kirchhoff（StVK）模型是最简单的超弹性材料模型之一。它的应变能密度函数定义为：

$$
W_{StVK}(\mathbf{G}) = \frac{\lambda}{2}\text{tr}^2(\mathbf{G}) + \mu\|\mathbf{G}\|_F^2
$$

其中$\lambda$和$\mu$是Lamé参数，$\text{tr}(\mathbf{G})$是Green-Lagrange应变张量的迹，$\|\mathbf{G}\|_F$是Frobenius范数。

参数$\lambda$与材料的体积压缩响应密切相关，尤其影响体积变化的惩罚强度；参数$\mu$是剪切模量，主要控制材料抵抗剪切和形状畸变的能力。

从应变能密度函数可以导出Second Piola-Kirchhoff应力张量：

$$
\mathbf{S} = \frac{\partial W}{\partial \mathbf{G}} = 2\mu\mathbf{G} + \lambda\text{tr}(\mathbf{G})\mathbf{I}
$$

然后通过变形梯度得到First Piola-Kirchhoff应力张量：

$$
\mathbf{P} = \mathbf{F}\mathbf{S} = \mathbf{F}[2\mu\mathbf{G} + \lambda\text{tr}(\mathbf{G})\mathbf{I}]
$$

StVK模型的优点是形式简单，计算效率高。然而，它在处理大变形时可能会出现数值不稳定的问题，特别是当材料被压缩到接近零体积时。

### 4.3 Neo-Hookean模型

Neo-Hookean模型是另一种常用的超弹性材料模型。与StVK模型基于Green-Lagrange应变不同，Neo-Hookean模型直接基于变形梯度的不变量。其应变能密度函数定义为：

$$
W_{NH}(\mathbf{F}) = \frac{\lambda}{2}\log^2(J) + \frac{\mu}{2}(I_C - 3) - \mu\log(J)
$$

其中$J = \det(\mathbf{F})$是变形梯度的行列式，$I_C = \|\mathbf{F}\|_F^2 = \text{tr}(\mathbf{F}^T\mathbf{F})$是右Cauchy-Green变形张量的第一不变量。

其中$-\mu\log J$项的作用是保证在无变形状态$\mathbf{F} = \mathbf{I}$时应力为零，并与$\frac{\mu}{2}(I_C - 3)$项共同给出合理的等方弹性响应。纯体积变形通常并不会使能量为零；只有在无变形状态下，能量才为零。

从应变能密度函数可以导出First Piola-Kirchhoff应力张量：

$$
\mathbf{P} = \frac{\partial W}{\partial \mathbf{F}} = \mu(\mathbf{F} - \mathbf{F}^{-T}) + \lambda\log(J)\mathbf{F}^{-T}
$$

Neo-Hookean模型在处理大变形时比StVK模型更加稳定，特别是对于接近不可压缩的材料。然而，它需要计算矩阵的逆，计算成本相对较高。

### 4.4 Corotated模型

Corotated模型是图形学中广泛使用的一种材料模型。它的核心思想是将变形梯度分解为旋转部分和拉伸部分，然后只在拉伸部分计算应变能。

具体而言，对于变形梯度$\mathbf{F}$，我们通过极性分解得到$\mathbf{F} = \mathbf{R}\mathbf{S}$，其中$\mathbf{R}$是旋转矩阵，$\mathbf{S}$是对称的正定矩阵。Corotated模型的应变能密度函数定义为：

$$
W_{CR}(\mathbf{F}) = \frac{\lambda}{2}(\text{tr}(\mathbf{S}) - 3)^2 + \mu\|\mathbf{S} - \mathbf{I}\|_F^2
$$

这个定义确保了纯旋转不产生应变能，因为当$\mathbf{F} = \mathbf{R}$时，$\mathbf{S} = \mathbf{I}$，从而$W_{CR} = 0$。

Corotated模型的主要优点是它在去除刚体旋转后再度量形变，因此能够处理较大的刚体旋转，同时保持类似线性弹性的简单形式。体积保持能力主要由体积惩罚项和Lamé参数$\lambda$决定，而不是Corotated模型本身自动保证的。

### 4.5 材料参数的选择

Lamé参数$\lambda$和$\mu$与更直观的材料参数——杨氏模量$Y$和泊松比$\nu$之间有着确定的关系。杨氏模量描述了材料抵抗拉伸变形的能力，泊松比描述了材料在拉伸时横向收缩的程度。

从Lamé参数到杨氏模量和泊松比的转换为：

$$
Y = \frac{\mu(3\lambda + 2\mu)}{\lambda + \mu}, \quad \nu = \frac{\lambda}{2(\lambda + \mu)}
$$

从杨氏模量和泊松比到Lamé参数的转换为：

$$
\mu = \frac{Y}{2(1 + \nu)}, \quad \lambda = \frac{Y\nu}{(1 + \nu)(1 - 2\nu)}
$$

在图形学应用中，常见的参数选择是：杨氏模量取值范围从几千帕到几万帕，泊松比取值范围从0.2到0.4。对于接近不可压缩的材料（如橡胶），泊松比接近0.5；对于可压缩的材料（如海绵），泊松比较小。

## 第五章 时间积分

### 5.1 显式时间积分

一旦我们能够计算每个节点上的力，就可以通过时间积分来更新节点的位置和速度。最简单的时间积分方法是显式欧拉方法，但它在能量守恒方面表现较差。更好的选择是辛欧拉方法，它在保持能量守恒方面表现优异。

辛欧拉方法的更新公式为：

$$
\mathbf{v}(t_{n+1}) = \mathbf{v}(t_n) + \frac{1}{m}\mathbf{f}(t_n)\Delta t
$$

$$
\mathbf{x}(t_{n+1}) = \mathbf{x}(t_n) + \mathbf{v}(t_{n+1})\Delta t
$$

这个方法的主要特点是先更新速度，再使用更新后的速度来更新位置。这种顺序看似简单，但它保证了方法的辛性质。辛欧拉方法不是严格能量守恒方法，但它是辛积分格式。对于保守系统，它通常能使能量误差在长时间内保持有界振荡，因此比显式欧拉更适合长时间动力学仿真。

显式时间积分的主要优点是实现简单，每步计算只需要计算当前状态的力。然而，它的时间步长受到稳定性条件的限制。对于基于有限元方法的软体仿真，稳定性条件大致为：

$$
\Delta t \leq C\frac{h}{\sqrt{E/\rho}}
$$

其中$C$是一个常数，$h$是网格尺寸，$E$是杨氏模量，$\rho$是密度。这意味着材料越硬、网格越细，允许的时间步长就越小。

### 5.2 隐式时间积分

隐式时间积分方法通过求解一个优化问题来计算下一时刻的状态。具体而言，我们寻找使得以下目标函数最小的位置：

$$
\mathbf{x}_{n+1} = \arg\min_{\mathbf{x}} g(\mathbf{x})
$$

其中目标函数定义为：

$$
g(\mathbf{x}) = \frac{1}{2\Delta t^2} \|\mathbf{x} - \mathbf{y}\|_M^2 + E(\mathbf{x})
$$

这里$\mathbf{y} = \mathbf{x}_n + \mathbf{v}_n\Delta t$是显式预测的位置，$\|\cdot\|_M$是质量加权范数，$E(\mathbf{x})$是弹性能量。

这个优化问题的第一项$\frac{1}{2\Delta t^2} \|\mathbf{x} - \mathbf{y}\|_M^2$保持了与显式预测的接近，第二项$E(\mathbf{x})$确保了物理合理性。这种公式化的好处是它自然地处理了约束和碰撞，可以通过修改能量函数来引入各种物理效果。

求解这个优化问题通常需要迭代方法。牛顿迭代法是最常用的选择，它利用目标函数的Hessian矩阵来构造二次近似：

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \nabla^2 g(\mathbf{x}_k)^{-1} \nabla g(\mathbf{x}_k)
$$

然而，对于大规模系统，直接计算和求逆Hessian矩阵是不现实的。实践中通常使用共轭梯度法等迭代求解器来近似求解线性系统。

隐式时间积分的主要优点是对于许多刚性较强的问题，相较显式方法具有更好的数值稳定性，允许使用更大的时间步长。但在强非线性、接触碰撞或材料接近不可压缩时，时间步长仍可能受到收敛性、精度和非线性求解稳定性的限制。

## 第六章 总结

本讲义系统介绍了基于有限元方法的变形仿真理论基础。从连续介质力学的基本守恒定律出发，我们建立了变形梯度、应变张量和应力张量的概念。通过应变能密度函数，我们定义了超弹性材料的力学行为，并介绍了常用的材料模型。最后，我们讨论了显式和隐式时间积分方法，以及它们各自的优缺点。

在实际应用中，方法的选择取决于具体问题的特点。对于需要实时交互的应用，显式时间积分配合简单的材料模型通常是首选；对于需要高精度的离线仿真，隐式时间积分配合复杂的材料模型更为合适。无论哪种选择，理解这些方法的物理基础和数学原理都是正确应用它们的前提。

有限元仿真仍然是一个活跃的研究领域，新的方法和改进不断涌现。从更高精度的离散格式到更高效的数值算法，从物理准确性的提升到计算效率的改进，有限元仿真技术将继续推动计算机图形学和科学计算的发展。

## 参考文献

1. Sifakis, E., & Barbic, J. (2012). FEM simulation of 3D deformable solids: a practitioner's guide to theory, discretization and model reduction. *SIGGRAPH ASIA 2012 Courses*.

2. Li, M., et al. (2022). Dynamic deformables: implementation and production practicalities (now with code!). *SIGGRAPH 2022 Courses*.

3. Belytschko, T., Liu, W. K., Moran, B., & Elkhodary, K. (2013). *Nonlinear finite elements for continua and structures* (2nd ed.). Wiley.

4. Bonet, J., & Wood, R. D. (2008). *Nonlinear continuum mechanics for finite element analysis*. Cambridge University Press.

5. GAMES103. *Physics Simulation in Visual Computing*. https://games-cn.org/games103/
