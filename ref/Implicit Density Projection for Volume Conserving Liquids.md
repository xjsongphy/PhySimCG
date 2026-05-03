# Implicit Density Projection for Volume Conserving Liquids

Tassilo Kugelstadt , Andreas Longva , Nils Thuerey , and Jan Bender

Abstract—We propose a novel implicit density projection approach for hybrid Eulerian/Lagrangian methods like FLIP and APIC to enforce volume conservation of incompressible liquids. Our approach is able to robustly recover from highly degenerate configurations and incorporates volume-conserving boundary handling. A problem of the standard divergence-free pressure solver is that it only has a differential view on density changes. Numerical volume errors, which occur due to large time steps and the limited accuracy of pressure projections, are invisible to the solver and cannot be corrected. Moreover, these errors accumulate over time and can lead to drastic volume changes, especially in long-running simulations or interactive scenarios. Therefore, we introduce a novel method that enforces constant density throughout the fluid. The density itself is tracked via the particles of the hybrid Eulerian/Lagrangian simulation algorithm. To achieve constant density, we use the continuous mass conservation law to derive a pressure Poisson equation which also takes density deviations into account. It can be discretized with standard approaches and easily implemented into existing code by extending the regular pressure solver. Our method enables us to relax the strict time step and solver accuracy requirements of a regular solver, leading to significantly higher performance. Moreover, our approach is able to push fluid particles out of solid obstacles without losing volume and generates more uniform particle distributions, which makes frequent particle resampling unnecessary. We compare the proposed method to standard FLIP and APIC and to previous volume correction approaches in several simulations and demonstrate significant improvements in terms of incompressibility, visual realism, and computational performance.

Index Terms—Fluid simulation, volume conservation, FLIP, APIC

# 1 INTRODUCTION

OVER the last decades, fluid simulation has become an important tool in the visual effects industry. In recent years, it has also started to get relevant for interactive applications like games or virtual training simulations due to advances in algorithms and consumer hardware. Hybrid methods which combine Eulerian and Lagrangian approaches, like FLIP and MPM, are very popular as they combine the advantages of both viewpoints. They have been successfully used to simulate a large variety of materials like water, highly viscous fluids, sand, snow, deformable solids, and they perform especially well when it comes to interactions between different types of materials.

However, a central challenge for all of these methods is to enforce the incompressibility constraints of the underlying physical models. A major problem of the pressure projection, which enforces incompressibility, is that large time steps or large solver tolerances yield numerical volume errors that cannot be corrected by the pressure solver. These errors accumulate over time and lead to a visible loss of volume, resulting in visual artifacts that are obvious and disturbing for viewers. For long-term simulations and interactive

T. Kugelstadt, A. Longva, and J. Bender are with the RWTH Aachen University, Aachen 52062, Germany. E-mail: {kugelstadt, longva, bender}@cs.rwth-aachen.de.   
N. Thuerey is with the Technical University of Munich, Munchen 80333, Germany. E-mail: nils.thuerey@tum.de.

Manuscript received 19 July 2019; accepted 9 Oct. 2019. Date of publication 15 Oct. 2019; date of current version 25 Feb. 2021.

(Corresponding author: Tassilo Kugelstadt.)

Recommended for acceptance by I. Hotz.

Digital Object Identifier no. 10.1109/TVCG.2019.2947437

scenarios these problems become particularly apparent. In practice, this forces users to accept tediously long run-times induced by small time steps and large iteration counts.

In this work, we propose a novel method to track the fluid density by using the particles of the hybrid simulator. This enables us to measure the absolute compression or expansion of the fluid. We use the continuous mass conservation law to derive a pressure Poisson equation which enforces not only a divergence-free velocity field, but also constant density throughout the fluid. In this way our approach prevents volume loss and therefore improves the visual quality of the simulation results. Moreover, since our density projection method can correct errors that occur in the standard FLIP or APIC simulations, we can relax the strict solver accuracy and time step requirements. This speeds up simulations by a factor of up to 8 while producing visually comparable results without noticeable volume changes. Another benefit of the proposed method is that enforcing constant density also leads to more uniform particle distributions, which further improves the quality of the simulation results, and makes frequent resampling of the particles unnecessary. We also propose a way to robustly handle particles that accidentally enter solid obstacles—a common problem in FLIP and APIC simulations. It can be incorporated in the density projection method by applying Neumann boundary conditions so that the particle distribution is optimized globally and particles leave obstacles without being projected onto other fluid particles. Finally, our method can be easily incorporated into existing hybrid simulation methods like FLIP or APIC by extending the standard pressure solver.

![](images/4754898baa745275caa6967a6d59344053231d646772063935268984272de270.jpg)

![](images/2819cd2266b76e89c131414e30ba69bb0e2e6d1474fff91aa8a16a022b7c3799.jpg)  
Fig. 1. Our implicit density projection method allows the efficient simulation of large-scale fluid scenarios while preventing undesired volume changes. Left: A double dam break with 8.8 million particles and $2 5 6 ^ { 3 }$ grid cells is hitting statues. Right: A complex river with up to 26 million particles and $1 6 0 0 \times 4 0 0 \times 8 0 0$ 256grid cells is simulated at large time steps without noticeable volume loss.

In several comparisons of our approach with FLIP, APIC, and previous volume correction methods we demonstrate significant improvements in terms of incompressibility, visual realism and computational performance. Moreover, we show that our method can robustly handle large scale scenarios with complex boundaries by simulating scenes with up to 26 million fluid particles. Finally, it enables us to produce realistic results without noticeable volume loss, even for large time steps (see Fig. 1) which improves the performance considerably. To summarize, our method yields significant quality and performance gains for a wide range of relevant liquid simulation scenarios, and is simple to integrate into existing solvers.

# 2 RELATED WORK

Three-dimensional simulations of fluids were first employed in computer graphics by Foster and Metaxas [1]. Stam subsequently proposed the Eulerian stable fluids scheme [2], which was the basis for a popular class of liquid solvers with particle level sets and second order free surface boundary conditions [3], [4]. The fluid implicit particle (FLIP) method likewise combines grids and particles, and has been especially popular for detailed liquid simulations and visual effects productions [5]. The stable coupling of fluids with immersed bodies has been an important direction of work [6], [7]. More recently, generic approaches for coupling different solvers have also been proposed [8]. The pressure solve is a central part of Eulerian and hybrid solvers. It typically dominates their performance, and hence approaches such as dimensionality reduction [9], [10], fast iterative solvers [11] and efficient methods for grid-based adaptivity [12] have been proposed to reduce its runtime impact. For details regarding Eulerian fluid solvers we recommend the books by Bridson [13] or Kim [14].

The advection step of fluid simulations has received special attention, for example in the form of error correction schemes [15], [16] and schemes for conserving mass and momentum [17], [18]. The latter ones have also been used by Lentine et al. [19] to simulate liquids with very large time steps using the particle level set method. We instead focus on large time steps in hybrid Eulerian/Lagrangian simulations.

The hybrid algorithms originate from the Particle in Cell (PIC) method, in the context of which other researchers

have proposed improvements in transferring quantities between particles and grid, such as the Affine Particle in Cell (APIC) [20] and Polynomial Particle in Cell [21] methods. Closely related, the Material Point Method (MPM), targets a wider range of material behaviors with a hybrid particlegrid approach similar to the FLIP method. While it was first proposed for snow simulations in the graphics context [22], it has since been extended to a wide range of material behaviors and simulation types [23], [24].

The FLIP algorithm itself has seen numerous extensions and improvements. For example, a narrow band particle placement was proposed to speed up calculations [25], [26]. In addition, researchers have noticed that the particle distribution of FLIP particles tends to cause problems over time. Several methods have aimed at alleviating this issue. Ando et al. [27] have proposed a position correction method inspired by SPH kernels. This method was extended by Um et al. [28] to sub-grid corrections. These approaches are closely related to our method because they can be used to prevent volume changes by correcting the particle positions. However, they are applying correction forces using explicit integration schemes which can cause an unstable behavior in a simulation with high stiffness values and large time steps as we show in our experiments. An implicit position correction has been proposed by Sato et al. [26], who use position based distance constraints to push particles apart when they are too close to each other. But their method is designed to correct the positions of particles in narrow band FLIP simulations and our experiments show that it cannot prevent volume loss in regular FLIP or APIC simulations with large time steps. Another approach for position correction was recently presented by Takahashi and Lin [29] which is based on position based fluids [30], an SPH method that enforces a constant density constraint. However, it requires costly particle neighborhood searches which can be avoided by our method where the particles only communicate indirectly via the grid.

The volume of fluids can be also controlled by adding divergence to the right-hand side of the Poisson equation in the pressure solve. This has been introduced by Feldmann et al. [31] to simulate the expansion of fluids in suspended particle explosions. Later it was used by Kim et al. [32] to control the volume of air bubbles. Their approach prevents a

volume change of entire fluid regions, in their case each air bubble. However, in the case of FLIP there are no distinct fluid regions, such as bubbles, so that this approach can be only applied globally. This is insufficient because the fluid usually gets compressed in some localized regions but the volume control will expand the whole fluid leading to worse particle distributions in formerly uncompressed regions. A local volume correction was proposed by Losasso et al. [33] in the context of a hybrid SPH and particle level set method. They use the mass conservation law to derive an additional term for the Poisson equation which penalizes too high particle numbers in local fluid regions. A similar approach has been proposed in the computational physics community by Liu et al. [34]. Gerszewski and Bargteil [35] applied this approach in FLIP simulations and refer to it as mass-full FLIP. In contrast to Losasso et al. they do not add an additional term to the right-hand side of the pressure Poisson equation because in the presence of strongly compressed fluids this can result in strong oscillations, popping and even explosions. To avoid these artifacts they use an additional solve on position-level and directly correct the particle positions without changing the velocity field as proposed by Narain et al. [36] for the simulation of granular materials or Irving et al. [37] for the simulation of incompressible deformable solids. Similar approaches are also popular in the SPH community [38], [39]. Our method is closely related to the one of Gerszewski and Bargteil who focused on the simulation of large scale splashing liquids. They use a unilateral incompressibility solve which is based on costly LCP solves to achieve realistic splashes. In contrast, our work focuses on the conservation of fluid volume which enables simulations with large time steps and lower solver accuracies and does not require expensive LCP solves.

# 3 METHOD

First, we briefly review the standard approach for fluid simulation and then discuss the volume conservation problem that we address in our work. An overview of our full simulation loop can be found as pseudo-code in Algorithm 1. The additional steps of our implicit density projection method are highlighted in blue and are derived and discussed in the following.

One Step of FLIP with Implicit Density Pro-Algorithm 1.jection. Additional Steps for the Density Projection are Highlighted   

<table><tr><td>1:</td><td>Advect particles through grid</td></tr><tr><td>2:</td><td>Compute grid density using Eq. (12)</td></tr><tr><td>3:</td><td>If necessary: handle degenerate configurations (see Section 3.2)</td></tr><tr><td>4:</td><td>Apply boundary conditions as described in Section 3.3</td></tr><tr><td>5:</td><td>Solve constant density PPE (10) with regular pressure solver</td></tr><tr><td>6:</td><td>Compute position change δx using Eq. (11)</td></tr><tr><td>7:</td><td>Correct particle positions by δx</td></tr><tr><td>8:</td><td>Transfer velocity from particles to grid</td></tr><tr><td>9:</td><td>Add velocity change due to body forces</td></tr><tr><td>10:</td><td>Compute RHS of Eq. (9)</td></tr><tr><td>11:</td><td>Apply boundary conditions</td></tr><tr><td>12:</td><td>Solve divergence-free PPE (9)</td></tr><tr><td>13:</td><td>Compute u(t+Δt) with Eq. (5)</td></tr><tr><td>14:</td><td>Transfer velocities from grid to particles</td></tr></table>

In the continuous theory, incompressible flows are modeled by the Navier-Stokes equation and the incompressibility condition

$$
\frac {\partial \mathbf {u}}{\partial t} = - \mathbf {u} \cdot \nabla \mathbf {u} + \mathbf {g} + v \Delta \mathbf {u} - \frac {1}{\rho} \nabla p, \tag {1}
$$

$$
\nabla \cdot \mathbf {u} = 0, \tag {2}
$$

where  denotes the velocity field, $\pmb { \mathrm { g } }$ the acceleration of gravity, $\nu$ uthe kinematic viscosity, $\rho$ gthe density and $p$ the pressure. Further, suitable boundary conditions are assumed as discussed in detail in the book of Bridson [13].

Most Eulerian methods assume that a divergence-free velocity field is sufficient to keep the fluid incompressible. But this only holds as long as $\nabla \cdot \mathbf { u } = 0$ is perfectly fulfilled u 0at all times and at all positions inside the fluid. In practice, simulations have limited spatial and temporal resolution, and the accuracy of the pressure solver is also limited, and as a result this condition cannot be perfectly fulfilled. This can lead to density changes and undesired compression or expansion of the fluid. An even bigger problem is that the divergence of the velocity field only gives a differential view on density changes, but the absolute density error is invisible to the solver as we discuss below. This means that density errors which accumulate over time cannot be corrected by the pressure solver. Therefore, high accuracy of the linear system solver and sufficiently small time steps are mandatory to avoid volume errors. However, our experiments show that even when using small time step sizes, that are determined by a CFL number [13] smaller than 1, and solving the pressure system accurately, undesired volume errors in the fluid cannot be completely avoided. This leads to undesired visual artifacts. Moreover, the time step and accuracy restrictions are major reasons for the high computation costs of the simulation.

# 3.1 Density Projection

While our main goal is to improve the visual quality of the simulation results by preventing undesired volume loss, we also want to overcome the strict time step and accuracy requirements in order to gain speedups. Therefore, we introduce a novel density projection method in this section. We track the density in the fluid by using the particles of the hybrid simulator. Density can be computed at each grid cell center by interpolating the particle mass onto the grid and dividing it by the cell volume, which is discussed in more detail in Section 3.2.

To see how this helps us on the theoretical side, we consider the mass conservation law:

$$
0 = \frac {\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf {u}). \tag {3}
$$

Here it becomes apparent that the divergence of the velocity field only measures density changes and not the absolute density of the fluid as we mentioned above. For incompressible fluids we need the additional constraint that the density of the fluid is constant

$$
\rho = \rho_ {0} = \text {c o n s t}, \tag {4}
$$

where $\rho _ { 0 }$ is the rest density of the fluid. Usually this con-0straint is inserted into Eq. (3) so that the density derivatives vanish, which results in the incompressibility condition $\nabla \cdot \mathbf { u } = 0$ . In contrast, we continue with the principle of u 0mass conservation to derive a pressure formulation that takes absolute density errors into account and is able to correct them. We discretize the Navier-Stokes Eq. (1) using the standard operator splitting approach

$$
\mathbf {u} (t + \Delta t) = \mathbf {u} ^ {*} - \Delta t \frac {1}{\rho_ {0}} \nabla p, \tag {5}
$$

where $\mathbf { u } ^ { * }$ denotes the intermediate velocity field after applyuing advection and non-pressure forces. To compute the pressure, we discretize the time derivative in the mass conservation law (3) using backward Euler

$$
\frac {\rho (t + \Delta t) - \rho^ {*} (t)}{\Delta t} + \nabla \cdot [ \rho (t + \Delta t) \mathbf {u} (t + \Delta t) ] = 0, \tag {6}
$$

where $\rho ^ { * } ( t )$ denotes the intermediate density field. Plugging the velocity of the next time step from Eq. (5) and the incompressibility constraint $\rho ( t + \Delta t ) = \rho _ { 0 }$ into Eq. (6) results in

$$
\frac {\rho_ {0} - \rho^ {*} (t)}{\Delta t} + \rho_ {0} \nabla \cdot \mathbf {u} ^ {*} - \rho_ {0} \Delta t \frac {1}{\rho_ {0}} \nabla^ {2} p = 0, \tag {7}
$$

which can be rearranged to

$$
\frac {\Delta t}{\rho_ {0}} \nabla^ {2} p = \nabla \cdot \mathbf {u} ^ {*} + \frac {1}{\Delta t} \left(1 - \frac {\rho^ {*} (t)}{\rho_ {0}}\right). \tag {8}
$$

This yields a pressure Poisson equation (PPE) with an additional term on the right-hand side. Instead of only considering the divergence of the intermediate velocity field, we also take changes of the intermediate density field into account. This means that pressure not only counteracts compression that happens during one time step due to divergence, but also counteracts compression that already happened in the past. In the following subsections we will discuss spatial discretization, how to compute density and how boundary conditions are applied.

# 3.2 Discretization

The obvious way to discretize the PPE (8) would be to use standard MAC grids. However, our observations show that — especially in the presence of large density deviations — this can lead to strong oscillations, popping artifacts and even explosions. This has been reported by other researchers as well [33], [35], [36] but it becomes especially problematic in situations with large time steps and inaccurate pressure solves.

Splitting the PPE. In the context of granular materials Narain et al. [36] proposed to resolve the aforementioned issues with an additional solve to correct the densities on position level by instantaneously moving the particles without changing the velocity. Mathematically speaking the PPE gets split into two separate equations using the superposition principle [40]

$$
\frac {\Delta t}{\rho_ {0}} \nabla^ {2} p _ {1} = \nabla \cdot \mathbf {u} ^ {*}, \tag {9}
$$

$$
\frac {\Delta t}{\rho_ {0}} \nabla^ {2} p _ {2} = \frac {1}{\Delta t} \left(1 - \frac {\rho^ {*} (t)}{\rho_ {0}}\right), \tag {10}
$$

where Eq. (9) is the usual PPE that eliminates divergence from the intermediate velocity field. The second PPE (10) results in correction pressures $p _ { 2 }$ that counteract compres-2sion and expansion. These two equations are discretized using standard MAC grids and can be solved with any pressure solver. Clearly, the sum of the exact solutions to the two equations solves Eq. (8) which can be easily seen by adding the equations and substituting $p _ { 1 } + p _ { 2 }$ with $p$ (this is 1 2also true when boundary conditions are applied cf. Section 3.3). However, now we have the possibility to treat the results of both PPEs differently. $p _ { 1 }$ is used in the standard 1way to update the grid velocity field with Eq. (5) such that it becomes divergence-free. When we plug $p _ { 2 }$ into Eq. (5), multiply by $\Delta t$ 2and reorder the terms we get the position changes on the grid

$$
\delta \mathbf {x} = \delta \mathbf {u} \Delta t = \left(\mathbf {u} (t + \Delta t) - \mathbf {u} ^ {*}\right) \Delta t = - \frac {\Delta t ^ {2}}{\rho_ {0}} \nabla p _ {2}. \tag {11}
$$

To update the particles, $\delta \mathbf { x }$ gets interpolated at their posixtions and the particles are moved without changing the velocity.

This solves the problem of oscillations and popping artifacts. We believe the reason is that correcting the positions directly does not introduce additional divergence to the velocity field as solving Eq. (8) does. In the latter case the additional divergence is needed such that the density gets corrected during one time step. Afterwards the divergence has to be removed again by the pressure solver. Since this solve does not exactly remove the divergence completely due to numerical inaccuracies, this once again leads to density deviations and oscillations. Similar problems arise when solving hard constraints in rigid body simulations, where the strategy of solving constraints on a velocity level and applying a separate position correction has also been successfully applied [41], [42].

Details on the implementation of the pressure and density solve in the simulation loop can be found in Algorithm 1. Since advection is the only step that changes the particle positions and therefore introduces density errors we correct the density directly after the advection step.

Density Computation. The density can be computed at the grid cell centers by mapping the particle mass $m _ { p }$ to the grid and dividing it by the cell volume $V$ so that the density at grid cell $i , j , k$ becomes

$$
\rho_ {i, j, k} = \frac {m _ {i , j , k}}{V} = \frac {1}{V} \sum_ {p} m _ {p} N (\mathbf {x} _ {p}). \tag {12}
$$

Here, $m _ { i , j , k }$ denotes the mass at the cell center which is interpolated from the particles at positions $\mathbf { x } _ { p }$ with the interpolation kernel $N ( \mathbf { x } _ { p } )$ x. As a default, we use the common trilinear xkernel. We also experimented with higher order kernels that are often used in MPM [43], but found that they lead to significant numerical dissipation. We initialize the particle mass so that we get rest density on the grid for uniform initial samplings. With $N$ particles per cell the mass of each particle is set to $m _ { p } = \rho _ { 0 } V / N$ .

![](images/76953427ae9c823577e9299ccfa8a10cf806d2eeb7b28a8de499ddf1d9932f89.jpg)  
Fig. 2. Push-out boundary conditions applied to the velocity component between a solid cell $s$ and a fluid cell $f$ of a MAC grid. Blue cells denote fluid cells, green cells denote solid cells. The quantity $\delta \mathbf { x }$ is given by the xdistance of the particle with the deepest penetration ion the cell. The velocity boundary conditions at each face of the cell are set such that particles will tend to move out of the solid.

Particle Deficiency. The method described above gives accurate density estimates within the fluid body. But at the free surface and solid obstacles we have the problem that air and solid cells do not contain any particles, but they may be overlapped by the interpolation kernels from fluid cells. This means that the density for the outermost fluid layer gets underestimated, which leads to undesired clumping of the particles that looks like artificial surface tension. In SPH this is known as the particle deficiency problem [44]. It can be overcome by clamping the density so that it cannot get smaller than the rest density. We only apply this clamping if at least one neighboring cell contains air. In the fluid, the particle attraction is actually desired, because there the density estimates are correct and undesired expansion of the fluid gets corrected. At solid walls, the particle deficiency can be overcome by also sampling the solids with particles. For non-moving objects, the mass has to be transferred to the grid only once which means that we get accurate density estimates there without any additional computation costs at runtime.

Limiting Displacements. We observed that large density errors can lead to very drastic correction displacements in a single time step, which can cause oscillations and popping artifacts. A simple way to avoid this is to clamp the righthand side of Eq. (10). The intuition behind this is that the solver only sees a fraction of the density error which can be corrected safely in one time step. The remaining error will be corrected during the next time steps and oscillations are avoided. Our tests showed that clamping $\rho ^ { * } / \rho _ { 0 }$ to the inter-0val : ; : limits the displacements so that the particles are 0 5 1 5not moved more than one cell width in one time step. Note that such extreme density deviations are rare in real-world applications.

Handling Degenerate Configurations. Our method can handle even degenerate particle configurations like very large numbers of particles in one grid cell (see Fig. 7). However, it is still possible that several particles are nearly at the same position so that they cannot be separated because they get the same interpolated position correction. To overcome this problem of conincidence we redistribute the particles in cells that contain far too many particles. As an indicator for the particle number per cell we use the density. When it is higher

than $1 . 5 \rho _ { 0 }$ we redistribute the particles in the cell. For lower 1 5 0thresholds the redistribution is done more often and can impact the performance. For higher thresholds, the chance of missing coincident particles increases. The redistribution is done by splitting the cell into uniform subcells. Each particle is placed randomly in a region close to the center of a subcell to ensure the particles cover the whole volume of a cell. The new particle velocities are interpolated from the grid. As the redistribution only happens rarely and is limited to few cells, the computational overhead is negligible.

# 3.3 Boundary Conditions

The boundary conditions (BC) of the density projection are similar to the ones used in the pressure projection. At the fluid-air interface, we have the Dirichlet boundary condition $p _ { 2 } = 0$ , meaning that the fluid can be moved into the air 2 0without any resistance. At fluid-solid interfaces one can use the Neumann BC $\delta \mathbf { x } \cdot \mathbf { n } = 0 \Rightarrow \nabla p _ { 2 } \cdot \mathbf { n } = 0 ,$ , where is the x n 0 2 n 0 nnormal vector of the solid surface. It prevents particles from being moved into solid obstacles and is identical to the BC for non-moving solids in the usual pressure solve.

Push-Out Boundary Condition. In addition to improving the conservation of volume, our formulation can be leveraged to enhance the treatment of solids in the flow. For solid walls the Neumann condition is typically used, and it works well as long as no fluid particles enter the solid obstacles. It is nonetheless a common problem in hybrid simulations that particles accidentally enter solid obstacles due to numerical errors in the advection and the grid-based velocity field. A standard approach is to project the particles back to the solid surface [5]. However, this can lead to particle clumping and volume loss, because the particles are moved into cells that might already have rest density or even too much mass inside them (see Fig. 9, top). Moreover, it is possible that several particles get projected to nearly the same position such that they cannot be separated because the interpolations from the grid result in the same velocities and displacements.

To overcome these problems we adapt the Neumann BC such that particles are pushed out of solid obstacles. In this way the density and therefore the particle distribution is optimized in a global fashion. When particles get moved out of the boundary into already filled cells, particles in these cells are also moved so that the density stays constant.

Usually solid objects are represented as signed distance fields (SDF). This means that the distance to the boundary $d ( \mathbf { x } )$ of a particle at position can be easily determined by x xquerying the SDF. Further, we can find the direction towards the closest point on the surface by computing the gradient of the distance function. Particles can be pushed out of the obstacle in one time step by the displacement

$$
\delta \mathbf {x} = - \frac {\nabla d (\mathbf {x})}{\| \nabla d (\mathbf {x}) \|} d (\mathbf {x}), \tag {13}
$$

where the negative sign comes from the convention that the distance value inside of objects is negative. If more than one particle is inside a solid cell, we use the $\delta \mathbf { x }$ of the one with the xdeepest penetration. The boundary displacement $\delta \mathbf { x }$ is set at xall MAC faces of the cell (see Fig. 2). When particles are deeper than one cell inside the obstacle the displacements

are set in the same way. There are no active pressure DOFs, so the pressure solve is not influenced. But when the displacements are applied to the particles afterwards, they get moved closer to the surface so that they get pushed out of the object in the following time steps. As mentioned above, we want to avoid very drastic position corrections. Therefore, we clamp the displacement $\delta \mathbf { x }$ at the solid boundary to half xof the cell width. This allows particles to leave solid objects on a stable path over several time steps, even for very deep penetrations (see Fig. 9, bottom).

The BC can be integrated into the solver by computing so called ghost pressures $p ^ { \mathrm { g h o s t } }$ for the solid obstacle cells [13]. They are denoted as ghost pressures since usually the pressure field is only defined for fluid and air cells but not for solid cells. The ghost pressure values can be determined by considering the pressure update Eq. (11) and discretizing the pressure gradient using finite differences. For one face of the MAC grid with index $i ,$ where one adjacent cell $f$ contains fluid and the other cell $s$ is solid, we get

$$
\delta x _ {i} = - \frac {\Delta t ^ {2}}{\rho_ {0}} \frac {p _ {f} - p _ {s} ^ {\mathrm {g h o s t}}}{\Delta x}, \tag {14}
$$

where $\Delta x$ is the grid cell spacing. This equation can be used to apply the BC so that the displacement of Eq. (13) is applied to push the particles out of the boundary. When we rearrange the equation, we can compute the ghost pressure for the solid cell as

$$
p _ {s} ^ {\mathrm {g h o s t}} = p _ {f} - \frac {\rho_ {0} \Delta x}{\Delta t ^ {2}} \delta x _ {i}. \tag {15}
$$

Next we consider the standard finite difference discretization of the Poisson Eq. (10) for the fluid cell $f$

$$
\frac {\Delta t}{\Delta x ^ {2}} \left(\alpha p _ {f} - p _ {s} ^ {\text {g h o s t}} - p _ {n}\right) = \frac {1}{\Delta t} \left(\rho_ {0} - \rho_ {f}\right), \tag {16}
$$

where $\alpha = 4$ in 2d and $\alpha = 6$ in ${ 3 \mathrm { d } } ,$ , and $p _ { n }$ contains the values 4 6of the remaining neighbors in the discrete 5 point (2d) or 7 point (3d) Laplacian stencil. Inserting $p _ { s } ^ { \mathrm { g h o s t } }$ from Eq. (15) results in

$$
\frac {\Delta t}{\Delta x ^ {2}} \left(\alpha p _ {f} - p _ {f} + \frac {\rho_ {0} \Delta x}{\Delta t ^ {2}} \delta x _ {i} - p _ {n}\right) = \frac {1}{\Delta t} \left(\rho_ {0} - \rho_ {f}\right). \tag {17}
$$

We can move the displacement term to the right-hand side yielding

$$
\frac {\Delta t}{\Delta x ^ {2}} \left((\alpha - 1) p _ {f} - p _ {n}\right) = \frac {1}{\Delta t} \left(\rho_ {0} - \rho_ {f} - \frac {\rho_ {0}}{\Delta x} \delta x _ {i}\right). \tag {18}
$$

This shows that the push-out Neumann BC can be easily implemented by subtracting $\begin{array} { r } { \frac { \rho _ { 0 } } { \Delta x \Delta t } \delta x _ { i } } \end{array}$ on the right hand side. It is completely analogous to the Neumann BCs for moving solid obstacles in the pressure projection where we have to subtract the relative velocity. This means that we can use the standard pressure solver and only exchange the right hand side. The benefit of this boundary condition is evaluated in detail in Section 4.3.

# 4 RESULTS

In this section we discuss results and compare our method to FLIP, APIC and previous volume correction methods in

terms of visual quality and computational performance. Therefore, we implemented our method as a plug-in for the Mantaflow framework [45] which was used to create all presented simulations. It was straightforward to integrate our method into the main simulation loop, and in the same way it should also be possible to incorporate our method into any existing FLIP or APIC solver without much effort. In the following subsections and in the accompanying video, we present several simulation results to demonstrate the improvements of our method compared to FLIP and APIC. If not stated otherwise, we use a solver tolerance of $\varepsilon = 1 0 ^ { - 3 }$ 10for the maximum norm of the residual, which is the default of Mantaflow. We start with a comparison of the volume conservation. Then we demonstrate that our approach leads to improved particle distributions and boundary handling. Finally, we present a performance comparison of FLIP and our method.

# 4.1 Volume Conservation

Incompressibility or conservation of volume is a very important visual feature of liquids like water. Therefore, we created several simulations to compare our method to FLIP and APIC in terms of incompressibility.

Volume Computation. In particle-based simulations volume is not exactly defined because the fluid is represented by a sparse set of points in space. Therefore, it is not possible to have an exact volume measure, and when we talk about volume, we mean that we computed it using the following approximation. In our initial sampling we have $N _ { \mathrm { i n i t } }$ initparticles in a completely filled fluid cell, which occupy the cell volume V. In the simulations we used $N _ { \mathrm { i n i t } } = 3 ^ { 2 }$ in 2d and $N _ { \mathrm { i n i t } } = 2 ^ { 3 }$ init 3in 3d. This value is used to compute the porinit 2tion of volume $V _ { i , j , k }$ of the cell with index $i , j , k$ that is actually covered by fluid as

$$
V _ {i, j, k} = \min  \left(\frac {N _ {i , j , k}}{N _ {\mathrm {i n i t}}} V, V\right), \tag {19}
$$

where $N _ { i , j , k }$ denotes the number of fluid particles in the cell. This means that cells with $N _ { \mathrm { i n i t } }$ or more particles inside are initcompletely filled and cells with less than $N _ { \mathrm { i n i t } }$ particles are only partially filled.

2d Double Dam Break. To evaluate volume conservation of FLIP and our method for different time step sizes $\Delta t$ and pressure solver tolerances $\varepsilon ,$ we simulated a double dam break scene with a large variation of these solver parameters. In a first series of simulations we used a fixed solver tolerance of $\varepsilon = 1 0 ^ { - 3 }$ . The simulations were run with fixed 10time step sizes of 1 ms, 5 ms, 10 ms and 20 ms. Here, ms1 is the smallest step size that a CFL number of 1 suggests during the whole scene.

The resulting simulations can be seen in the accompanying video, where deviation of the density from the rest density is color-coded from white indicating low deviation to red indicating high deviation. To evaluate the incompressibility quantitatively, we plotted the relative deviation of the total fluid volume from the initial volume in Fig. 3. As expected, our method conserves the fluid volume within the accuracy of the measurement, and the volume changes after 30 seconds of simulation are below 2 percent for all tested time step sizes. When using our method and a large time step

![](images/65185961a396070e96b8867aefc93ef1aa6469a9be21fba2d0dff6bd3c5e1dbd.jpg)  
Fig. 3. Comparison of volume errors for FLIP and our method in a 2d double dam break with particles, $1 2 8 \times 1 2 8$ grid cells, and a solver tolerance of $1 0 ^ { - 3 }$ 71k 128 128. The total fluid volume divided by the initial volume is 10plotted over time for several time step sizes. Our method keeps the volume change below 2 percent for all considered time steps. In contrast, FLIP suffers from significant volume loss, especially for large time steps.

size of 20 ms, the graph shows a volume loss during the first splash which is quickly corrected in the following frames. Note that the volume loss is not corrected immediately by our method since we clamped the right-hand side of Eq. (10) to improve the stability (see Section 3.2). In contrast, FLIP suffers from significant volume changes, which get worse when the time step size increases, and which accumulate over time. For 20 ms time steps, nearly 58 percent of the fluid volume gets lost during the 30 seconds of the simulation. For time steps of size 10 ms and 5 ms the fluid loses 26 percent and 14 percent of its volume, respectively. For these step sizes the volume deviation can be directly observed in the video, because it leads to lower water levels when the fluid comes to rest. In the simulation with $\Delta t = 1$ the volume also devi-1 msates by 10 percent compared to the rest volume, but it leads to a rise in the water level. Here, the change comes from void regions inside the fluid body where FLIP produces a very uneven particle sampling (see Fig. 8). We will discuss the improvements that our method provides in more detail in Section 4.2. Further, we evaluate the dependence of the solver tolerance on incompressibility. Therefore, the 2d double dam break simulation is repeated with $\Delta t = 5$ and solver tolerances of $\varepsilon = 1 0 ^ { - 2 } , \varepsilon = \mathsf { \bar { 1 0 } ^ { - 3 } } , \varepsilon = 1 0 ^ { - 4 }$ 5and $\varepsilon = 1 0 ^ { - 6 }$ . The result-10 10 10 10ing simulations are shown in the accompanying video, and the relative volume deviation over time is plotted in Fig. 4. Our method is able to keep the volume nearly constant, and the volume change was less than 1 percent for all tested tolerances. In contrast to this, the FLIP simulation suffers from

![](images/f4363b303baa37cf4e16d6a73fa37240209f3e8967d92843e32c6fe09d6bcb6c.jpg)  
Fig. 4. Comparison of volume errors for FLIP and our method in a 2d double dam break with particles, $1 2 8 \times 1 2 8$ grid cells, and a time step of $\Delta t = \mathrm { m s } 5$ 71k 128 128. The total fluid volume divided by the initial volume is ms5plotted over time for several pressure solver tolerances. Our method keeps the volume change below 1 percent for all considered tolerances. In contrast, FLIP suffers from volume loss, especially for large tolerances.

significant volume loss when the tolerance is too large. For $\varepsilon = 1 0 ^ { - 2 }$ one third of the fluid volume gets lost during the 10simulation. For the stricter tolerance values the volume changes by roughly 13 percent-14 percent and does not improve significantly for smaller $\varepsilon$ values.

Comparison to Related Work. To compare our approach to previous volume correction methods we simulated the 2d double dam break with the parameters $\varepsilon = 1 0 ^ { - 3 }$ and $\Delta t = 2 0$ 10with various methods. One frame of the simula-20 mstions is depicted in Fig. 5. First, we tried to solve Eq. (8) directly, which results in explosion-like artifacts as reported by Losasso et al. [33]. We implemented their solution which reduces the problem by averaging the density deviation on the RHS of Eq. (8) over a time interval of length t. For small values of $\tau$ the artifacts are still present. Larger values reduce the artifacts but then the volume correction is not strong enough and the volume errors become visible. As proposed in [33] we use $\tau = 1 \mathrm { s }$ for the scene in the video. It 1salready suffers from significant volume loss and even for much larger values of $\tau$ there are artifacts.

We also implemented the method of Ando et al. [27] which uses SPH like weak spring forces to push the particles apart when they come too close to each other. This prevents volume loss but it suffers from stability issues due to the explicit integration. This results in undesired high frequency oscillations in the particle distribution. Decreasing the stiffness of the springs helps with this issue but then the

![](images/c35d764e0609f3886e48e9b5ff72e01852f253a6acb11a9d5ff0c2df6f292b03.jpg)

![](images/18372ccb7ef8e4eacd51eede42b891ca6cab942dbea92b8e888122886eab30c5.jpg)

![](images/237be6c1555eb2a613becce4bcd503eebb93dafa88b8103489d00cbafbbd935f.jpg)

![](images/11a69121f55d096ff4fff07c430191dcd5940b855c8bbadf7c62d7b03c9feda1.jpg)

![](images/29148ac9fb220fff789d4c28551bd0c3ea45d1a2ecf5fe01b4716b27020ba721.jpg)

Fig. 5. Comparison to previous volume correction methods in a 2d double dam break scene. a) - f) Show the same frame of simulations with different approaches. The velocities are color coded from blue (low) to white (high). a): Standard FLIP looses a significant portion of the fluid volume. b): Solving Eq. (8) results in explosion artifacts. c): The averaging approach of Losasso et al. [33] suffers from popping artifacts (cf. accompanying video). d): The weak spring position correction of Ando and Tsuruno [27] suffers from high frequency oscillations of the particles (cf. accompanying video). e): The distance constraints of Sato et al. [26] are not able to prevent the volume loss. f): Our method preserves the fluid volume and improves the particle distribution without artifacts.   
![](images/4b6e4c016dbd6ab8a2dd3aea3ce33768a66f792060fd47072d88a6fecd8db8b9.jpg)  
Authorized licensed use limited to: Peking University. Downloaded on May 03,2026 at 10:02:42 UTC from IEEE Xplore. Restrictions apply.

![](images/d973d582dfd682a461ee2b356499e0179cc7b62883da7e06230c693059a949f3.jpg)

![](images/206db4b4748f093a22f1b62851f9672581a852299acb9ae9c307e5fa07a49775.jpg)

![](images/779fa14a526956148a992115bf723b32c9445426094d3b39fcc3406915852204.jpg)  
Fig. 6. Comparison of the volume conservation of APIC and our method in a complex 3d scene where fluid gets swirled around by a rotating cuboid. Left: Initial configuration. Middle: In the APIC simulation the water levels constantly drops during 30 second of simulation such that 38 percent of the fluid volume gets lost. Right: When simulated with APIC in combination with our density projection the volume change is below 0.5 percent.

volume errors become evident. We also compare to the correction method of Sato et al. [26] which uses implicit position based distance constraints to push the particles apart. However, this method was designed to improve the particle distributions near the surface in narrow band FLIP simulations and it is not capable of preventing the volume loss. In contrast to these previous methods, our approach successfully prevents volume errors without introducing artifacts. Moreover, we do not need particle neighborhood searches as Ando et al. and Sato et al.

3d Rotating Cuboid. To also evaluate the incompressibility in a more complex 3d scenario, we simulate a basin of water with a rotating cuboid in it. The scene contains 8 million particles and is simulated on a grid with $1 2 8 ^ { 3 }$ cells. The time step size is 128determined adaptively by using a CFL number of 1. The simulation was performed using FLIP and APIC, each with and without our method. The initial configuration and one frame after 30 seconds of the APIC simulation with and without our method is depicted in Fig. 6. In the FLIP / APIC simulation the fluid lost $3 8 . 5 \% / 3 8 \%$ of its initial volume, which can be 38 5% 38%easily observed. With our method, the volume is preserved up to 0.5 percent, which is not visually noticeable.

Stability Tests. To test the stability of our method, we simulate two test scenarios. In the first one, we take a fluid basin in which the lower half is filled with 1 million particles and let the simulation run for a few seconds with a deactivated pressure solver. In this time, all fluid particles drop onto the floor. Then we reactivate the pressure solver. Without our method, nothing happens when the solver is reactivated, because the velocity field is already divergence-free and none of the lost volume is recovered. With our density projection, the entire fluid volume is stably recovered in less than a second. A side

by side comparison of this simulation with and without our method can be found in the accompanying video.

In a second test we use the same scene, but instead of deactivating the pressure solver, we move the 1 million particles into a single grid cell as an initial condition for the simulation. Without our method, the FLIP simulation is not able to recover the fluid volume, and after a few seconds the fluid covers only a one grid cell thick layer on the floor. With our approach, the entire fluid volume stably recovers in less than one second. The initial condition and three frames from the recovering process are depicted in Fig. 7.

Complex Scenarios. We performed two simulations with a large number of particles and complex boundary geometry to verify that our method also works in practical scenarios. In the first one, we set up a double dam break with statues in the fluid basin in order to generate interesting splashes (see Fig. 1, left). The simulation runs on a $2 5 6 ^ { 3 }$ grid, contains 2568.8 million particles and the time step is determined adaptively with a CFL number of 1. The second scenario is a river which is flowing through a complex canyon geometry with two waterfalls (see Fig. 1, right). This simulation is performed on a $1 0 0 \ \mathrm { m } \times 2 5 \ \mathrm { m } \times 5 0 \ \mathrm { m }$ domain which was discretized with $1 6 0 0 \times 4 0 0 \times 8 0 0$ 0 mgrid cells, and it contains up 1600 400 800to 26 million particles. It runs stably with a large time step of 20 ms. These examples demonstrate that our approach can handle complex simulations efficiently and robustly without noticeable volume loss.

# 4.2 Particle Distribution

Another problem that is solved by enforcing constant density fields is the uneven particle sampling of regular FLIP and APIC simulations. This is demonstrated in Fig. 8, where

![](images/27ee612fbbd232335249b294f17104f2a404e3c364807848871aa9b2b337daf5.jpg)

![](images/b9a0c0b0f579246d459cfcb68f3d9668d3822654848591315c3716651c1ecd6d.jpg)

![](images/ea9afc5cf04823ca6224f196acb40307f9831990d4f10646bdd33448d8f445ef.jpg)

![](images/adfd1474d63052366bc1cd815b3e58bf38fa04af137fcc710b079da3c3f8d67f.jpg)  
Fig. 7. Stability test for our method: We place 1 million particles in a single cell of a $6 4 ^ { 3 }$ grid and then start the simulation. With our method the fluid 64recovers its volume stably in less than a second. With regular FLIP the fluid cannot recover its volume.

![](images/2202d4e28d8f93eac322be25200932c230b2b759a157b8e19d71420f11e440ba.jpg)

![](images/290647946629f68a01e0a7833b442124dd67e6924cd9c115ca04b1e25e46d6f2.jpg)  
Fig. 8. Comparison of particle distributions of FLIP (left) and our method (right) in a 2d double dam break scene after the fluid has settled. The simulation contains 71k particles on a $1 2 8 ^ { 2 }$ grid. Density errors are color 128coded, red refers to high and white to low errors. Left: The regular FLIP simulation suffers from uneven particle distributions with particle clumping and void regions. This reduces the visual quality and results in unphysical volume changes. Right: Our method produces more regular particle distributions which increase the visual quality and avoid volume errors.

we have taken one frame of the 2d double dam break simulation which we discussed in the previous subsection and zoom in to show the particle distribution. Here, density errors are color coded, where red refers to high errors, and white refers to low errors. The left part shows the regular FLIP simulation which contains particle clusters and void regions. They reduce the visual quality and lead to volume errors. The right part shows the simulation with our method. The particles are more regularly sampled which improves visual quality. The color coding shows that there are no observable density errors in the entire fluid volume.

# 4.3 Boundary Handling

In the following experiment we compare our push-out boundary conditions with the standard boundary handling

of FLIP [5]. We simulate a basin of water in 2d which has a solid obstacle in the form of the Stanford bunny inside. As an initial condition, we sample not only the fluid but also the bunny with fluid particles. When we start the simulation using our approach, all particles leave the bunny in a fraction of a second in a stable way such that they contribute to the fluid volume and the water level rises. We also repeat this simulation without our method, in which case all particles are immediately projected to the surface of the bunny. However, then they clump together in a narrow band around the obstacle. This does not lead to a rise in the water level, and the fluid volume inside the bunny gets lost. The initial condition and three frames of the simulation with and without our method are shown in Fig. 9. Thus, our method successfully recovers the liquid volume even for tough scenarios.

# 4.4 Computational Performance

We evaluate the computational costs of our method and compare them to standard FLIP by repeating a 3d double dam break simulation several times with different time steps and solver tolerances. It was performed on a $6 4 \times { 1 2 8 } ^ { - } \times { 6 4 }$ grid and contained 275k particles. The simula-64 128 64tion was run on a standard PC with an Intel Core i7 6700k quad-core CPU.

To evaluate the additional costs of our method, we run the simulation with the same parameters $\varepsilon = 1 0 ^ { - 4 }$ and $\Delta t = \mathrm { m s 2 }$ 10once with and without our method. Using stanms2dard FLIP, the simulation requires on average ms188 per time step, while with our method it needs ms242. This means that our density projection increases the computation cost per time step by 29 percent. However, we can actually achieve significant speedups by using larger time steps and a lower solver accuracy without losing visual quality and without drastic volume changes in the fluid.

![](images/837086afb6bb562383b3ca07accc9971dca82555a23b37326145175ba082b3cd.jpg)

![](images/c2b34ea4652d28c6bb0f19973d5f6a76235533a408c012349b70009bcdfd066c.jpg)

![](images/d5a3581c98397fcca71647b7518f087510b35974312aacda0e76a7167cbeb190.jpg)

![](images/a81626fba261869a54fc89e8d8cde0e773ba315b68aa8f03e0f170d3aad38f68.jpg)

![](images/91f72ca55851fccb0a7dd48ba24f5a105a02396b86bed0fc042a749f47e924bb.jpg)

![](images/00b7f5b6c76800e6cbf86ccbf4f21862cae905829404116f47fe026ee498230b.jpg)

![](images/94dc03b42d5163b1ba9afe92ac8b39ccb3eb47e52a08b9c5ea431732b7b8ddc0.jpg)

![](images/7ba221f4184e45a6e6ab07a556cf143a919300fd0049f1e875b53a2d4bb775bc.jpg)  
Fig. 9. Comparison of the standard FLIP boundary handling (top row) and our push-out boundary conditions (bottom row). The simulation is done on a $\mathrm { \bar { 1 } 8 5 ^ { 3 } }$ grid and contains 216k particles. Left column: As an initial condition we also sample a solid obstacle in form of a bunny with fluid particles. Top 185row: From left to right several frames of the regular FLIP simulation are depicted. The particles are projected to the surface where they are clumping together and the entire fluid volume that was inside the bunny gets lost. Bottom row: From left to right several frames of the simulation with our method are shown. The particle distribution gets optimized globally so that all particles leave the solid obstacle and the fluid volume is conserved. This can be observed as a rising water level.

![](images/feca36958ef99c32fa41b27ef1d775e1ca8a1749b032b4d99152b68e0eb16c56.jpg)

![](images/b39b1bfdcec390da71d86d20f2d06d70a5ab71f2ff4ae09704a3ffe4647c12b7.jpg)  
Fig. 10. Performance comparison of regular FLIP (left) and our method (right) in a double dam break with 275k particles and 64x128x64 grid cells. Our method robustly handles 10 times larger time steps and a higher solver tolerance than FLIP. It produces visually similar results without noticeable volume change while being 7.8 times faster.

Using our method, the simulation runs stably at $\Delta t =$ and $\varepsilon = 1 0 ^ { - 3 }$ with a volume change of 0.9 percent, 20 ms 10which is visually not noticeable. When using the same parameters with FLIP, 35 percent of the fluid volume gets lost in 10 seconds. We repeated the FLIP simulation several times and decreased the time step and the solver tolerance until we did not see any improvements in the volume conservation. For $\Delta t = 2$ and $\varepsilon = 1 0 ^ { - 4 }$ , FLIP still suffers 2 ms 10from a volume change of 10 percent which did not improve with smaller time steps and tolerances. A side by side comparison of this simulation and the one using our method at $\Delta t = 2 0$ and $\varepsilon = 1 0 ^ { - 3 }$ is shown in Fig. 10. Using these set-20 ms 10tings, our method produces visually similar results while being $7 . 8 x$ faster than the standard FLIP simulation. In general, scenarios can arise where using larger time steps is not an option because higher numerical damping can dissipate vorticity or details in the flow. However, in interactive or real time applications where the computation times are strictly limited, our method allows for significant speedups without volume loss. In future work we plan to further investigate the effect of large time steps on the kinetic energy and vorticity of the fluid and whether it can be improved, e.g., with fast energy projections [46] or micropolar models [47].

# 5 CONCLUSION

We presented an implicit density projection method which extends hybrid simulation methods, like FLIP or APIC, so that constant density can be efficiently enforced. This has the central advantage that volume errors that accumulate over time, and which are invisible to the regular pressure solver, can accurately and efficiently be corrected. In comparison to previous methods, our approach yields excellent conservation of volume without suffering from visual artifacts. This enables the use of larger times steps and less accurate pressure projections, which results in significant speedups. Further, it improves the particle distributions such that a frequent particle resampling is not required. Another benefit is that fluid particles can be pushed out of solid obstacles using our boundary conditions without introducing particle clumping and volume loss. In summary, our approach provides numerous benefits that lead

to improved simulation quality and performance and is applicable to a wide range of practical scenarios.

Since our method is an extension of FLIP and APIC, it has the same limitations except for the ones that we addressed. One of these limitations is that it requires a dense particle sampling. Therefore, an interesting direction for future work would be a combination of our method with the narrow band FLIP approach of Ferstl et al. [25] to decrease the computation times. Moreover, we plan to investigate the applicability of our method in interactive applications by considering GPU implementations similar to the ones proposed by Wu et al. [48] or Gao et al. [49]. Finally, we have focused on liquid simulations in this work, but we plan to incorporate our approach into the material point method to simulate volume conserving deformable solids.

# ACKNOWLEDGMENTS

This work is supported by the German Research Foundation (DFG) under contract number BE 5132/4-1.

# REFERENCES

[1] N. Foster and D. Metaxas, “Realistic animation of liquids,” Graphical Models Image Process., vol. 58, no. 5, pp. 471–483, 1996.   
[2] J. Stam, “Stable fluids,” in Proc. ACM Conf. Comput. Graphics Interactive Techn., 1999, pp. 121–128.   
[3] N. Foster and R. Fedkiw, “Practical animation of liquids,” in Proc. 28th Annu. Conf. Comput. Graphics Interactive Techn., New York, NY, USA, 2001, pp. 23–30, doi: 10.1145/383259.383261.   
[4] D. Enright, D. Nguyen, F. Gibou, and R. Fedkiw, “Using the particle level set method and a second order accurate pressure boundary condition for free-surface flows,” in Proc. 4th ASME-JSME Joint Fluids Eng. Conf., 2003.   
[5] Y. Zhu and R. Bridson, “Animating sand as a fluid,” ACM Trans. Graphics, vol. 24, no. 3, pp. 965–972, 2005.   
[6] C. Batty, F. Bertails, and R. Bridson, “A fast variational framework for accurate solid-fluid coupling,” ACM Trans. Graphics, vol. 26, no. 3, Jul. 2007, Art. no. 100.   
[7] A. Robinson-Mosher, T. Shinar, J. Gretarsson, J. Su, and R. Fedkiw, “Two-way coupling of fluids to rigid and deformable solids and shells,” ACM Trans. Graphics, vol. 27, no. 3, 2008, Art. no. 46.   
[8] M. Akbay, N. Nobles, V. Zordan, and T. Shinar, “An extended partitioned method for conservative solid-fluid coupling,” ACM Trans. Graphics, vol. 37, no. 4, 2018, Art. no. 86.   
[9] M. Lentine, W. Zheng, and R. Fedkiw, “A novel algorithm for incompressible flow using only a coarse grid projection,” ACM Trans. Graphics, vol. 29, no. 4, 2010, Art. no. 114.   
[10] R. Ando, N. Thuerey, and C. Wojtan, “A dimension-reduced pressure solver for liquid simulations,” Comput. Graphics Forum, vol. 34, no. 2, pp. 473–480, 2015.   
[11] A. McAdams, E. Sifakis, and J. Teran, “A parallel multigrid poisson solver for fluids simulation on large grids,” in Proc. ACM SIG-GRAPH/Eurographics Symp. Comput. Animation, 2010, pp. 65–74.   
[12] M. Aanjaneya, M. Gao, H. Liu, C. Batty, and E. Sifakis, “Power diagrams and sparse paged grids for high resolution adaptive liquids,” ACM Trans. Graphics, vol. 36, no. 4, 2017, Art. no. 140.   
[13] R. Bridson, Fluid Simulation for Computer Graphics, Second Edition. New York, NY, USA: Taylor & Francis, 2015.   
[14] D. Kim, Fluid Engine Development. AK Peters/CRC Press, 2017.   
[15] A. Selle, R. Fedkiw, B. Kim, Y. Liu, and J. Rossignac, “An unconditionally stable macCormack method,” J. Sci. Comput., vol. 35, no. 2–3, pp. 350–371, Jun. 2008.   
[16] J. Zehnder, R. Narain, and B. Thomaszewski, “An advectionreflection solver for detail-preserving fluid simulation,” ACM Trans. Graphics, vol. 37, no. 4, 2018, Art. no. 85.   
[17] M. Lentine, M. Aanjaneya, and R. Fedkiw, “Mass and momentum conservation for fluid simulation,” in Proc. ACM SIGGRAPH/Eurographics Symp. Comput. Animation, 2011, pp. 91–100.   
[18] N. Chentanez and M. Muller, “Mass-conserving eulerian liquid €simulation,” in Proc. ACM SIGGRAPH/Eurographics Symp. Comput. Animation, 2012, pp. 245–254.

[19] M. Lentine, M. Cong, S. Patkar, and R. Fedkiw, “Simulating free surface flow with very large time steps,” in Proc. ACM SIG-GRAPH/Eurographics Symp. Comput. Animation, 2012, pp. 107–116.   
[20] C. Jiang, C. Schroeder, A. Selle, J. Teran, and A. Stomakhin, “The affine particle-in-cell method,” ACM Trans. Graphics, vol. 34, no. 4, Jul. 2015, Art. no. 51.   
[21] C. Fu, Q. Guo, T. Gast, C. Jiang, and J. Teran, “A polynomial particle-in-cell method,” ACM Trans. Graphics, vol. 36, no. 6, 2017, Art. no. 222.   
[22] A. Stomakhin, C. Schroeder, L. Chai, J. Teran, and A. Selle, “A material point method for snow simulation,” ACM Trans. Graphics, vol. 32, no. 4, 2013, Art. no. 102.   
[23] C. Jiang, T. Gast, and J. Teran, “Anisotropic elastoplasticity for cloth, knit and hair frictional contact,” ACM Trans. Graphics, vol. 36, no. 4, 2017, Art. no. 152.   
[24] Y. Hu et al., “A moving least squares material point method with displacement discontinuity and two-way rigid body coupling,” ACM Trans. Graphics, vol. 37, no. 4, 2018, Art. no. 150.   
[25] F. Ferstl, R. Ando, C. Wojtan, R. Westermann, and N. Thuerey, “Narrow band FLIP for liquid simulations,” Comput. Graphics Forum, vol. 35, no. 2, pp. 225–232, 2016.   
[26] T. Sato, C. Wojtan, N. Thuerey, T. Igarashi, and R. Ando, “Extended narrow band flip for liquid simulations,” Comput. Graphics Forum, vol. 37, pp. 169–177, 2018.   
[27] R. Ando and R. Tsuruno, “A particle-based method for preserving fluid sheets,” in Proc. ACM SIGGRAPH/Eurographics Symp. Comput. Animation, 2011, pp. 7–16.   
[28] K. Um, S. Baek, and J. Han, “Advanced hybrid particle-grid method with sub-grid particle correction,” Comput. Graphics Forum, vol. 33, no. 7, pp. 209–218, 2014.   
[29] T. Takahashi and M. C. Lin, “A geometrically consistent viscous fluid solver with two-way fluid-solid coupling,” Comput. Graphics Forum, vol. 38, no. 2, pp. 49–58, 2019.   
[30] M. Macklin and M. Muller, “Position based fluids,” ACM Trans. €Graphics, vol. 32, no. 4, 2013, Art. no. 104.   
[31] B. E. Feldman, J. F. O’Brien, and O. Arikan, “Animating suspended particle explosions,” in Proc. ACM SIGGRAPH Papers, 2003, pp. 708–715.   
[32] B. Kim, Y. Liu, I. Llamas, X. Jiao, and J. Rossignac, “Simulation of bubbles in foam with the volume control method,” in Proc. ACM SIGGRAPH 2007 Papers, San Diego, California: New York, NY, USA, 2007, doi: 10.1145/1275808.1276500.   
[33] F. Losasso, J. O. Talton, N. Kwatra, and R. Fedkiw, “Two-way coupled SPH and particle level set fluid simulation,” IEEE Trans. Vis. Comput. Graphics, vol. 14, no. 4, pp. 797–804, Jul./Aug. 2008.   
[34] J. Liu, S. Koshizuka, and Y. Oka, “A hybrid particle-mesh method for viscous, incompressible, multiphase flows,” J. Comput. Physics, vol. 202, no. 1, pp. 65–93, 2005.   
[35] D. Gerszewski and A. W. Bargteil, “Physics-based animation of large-scale splashing liquids,” ACM Trans. Graphics, vol. 32, no. 6, pp. 185–1, 2013.   
[36] R. Narain, A. Golas, and M. C. Lin, “Free-flowing granular materials with two-way solid coupling,” ACM Trans. Graphics, vol. 29, no. 6, 2010, Art. no. 1.   
[37] G. Irving, C. Schroeder, and R. Fedkiw, “Volume conserving finite element simulations of deformable models,” ACM Trans. Graphics, vol. 26, no. 3, Jul. 2007, Art. no. 13.   
[38] J. Bender and D. Koschier, “Divergence-free SPH for incompressible and viscous fluids,” IEEE Trans. Vis. Comput. Graphics, vol. 23, no. 3, pp. 1193–1206, Mar. 2017.   
[39] S. Band, C. Gissler, M. Ihmsen, J. Cornelis, A. Peer, and M. Teschner, “Pressure boundaries for implicit incompressible SPH,” ACM Trans. Graphics, vol. 37, no. 2, Feb. 2018, Art. no. 14.   
[40] R. Haberman, Applied Partial Differential Equations, 4th ed. Englewood Cliffs, NJ, USA: Prentice Hall, 2003.   
[41] M. Cline and D. Pai, “Post-stabilization for rigid body simulation with contact and constraints,” in Proc. IEEE Int. Conf. Robot. Autom., 2003, vol. 3, pp. 3744–3751.   
[42] K. Erleben, “Rigid body contact problems using proximal operators,” in Proc. ACM SIGGRAPH / Eurographics Symp. Comput. Animation, 2017, Art. no. 13.   
[43] M. Steffen, R. M. Kirby, and M. Berzins, “Analysis and reduction of quadrature errors in the material point method (MPM),” Int. J. Numerical Methods Eng., vol. 76, no. 6, pp. 922–948, 2008.   
[44] M. Ihmsen, J. Orthmann, B. Solenthaler, A. Kolb, and M. Teschner, “SPH fluids in computer graphics,” in Eurographics (State of the Art Reports). The Eurographics Assoc., 2014.

[45] N. Thuerey and T. Pfaff, “Mantaflow version 0.12,” 2018. [Online]. Available: http://mantaflow.com/   
[46] D. Dinev, T. Liu, J. Li, B. Thomaszewski, and L. Kavan, “FEPR: Fast energy projection for real-time simulation of deformable objects,” ACM Trans. Graphics, vol. 37, no. 4, 2018, pp. 79.   
[47] J. Bender, D. Koschier, T. Kugelstadt, and M. Weiler, “Turbulent micropolar sph fluids with foam,” IEEE Trans. Vis. Comput. Graphics, vol. 25, no. 6, pp. 2284–2295, Jun. 2018.   
[48] K. Wu, N. Truong, C. Yuksel, and R. Hoetzlein, “Fast fluid simulations with sparse volumes on the GPU,” in Computer Graphics Forum, vol. 37, Hoboken, NJ, USA: Wiley, 2018.   
[49] M. Gao, X. Wang, K. Wu, A. Pradhana, E. Sifakis, C. Yuksel, and C. Jiang, “GPU optimization of material point methods,” ACM Trans. Graphics, vol. 37, no. 6, 2018, Art. no. 254.

![](images/13a622ad957cfcf3d5a3383d3b5590d922e04e1e5cd9f36315319ebc5295e85a.jpg)

Tassilo Kugelstadt received the BSc degree in physics and the MSc degree in computer science in the Natural Sciences from JGU Mainz, in 2013 and 2015 respectively. He is working toward the PhD degree at RWTH Aachen University. His research interests include physically-based simu lation of deformable solids, elastic rods, and fluids.

![](images/724169bab2d1f3c62484b836ed1aaee2dc86eb392985eafc557736fe6192b9f6.jpg)

Andreas Longva received the MSc degree in applied mathematics from the Norwegian University of Science and Technology, Trondheim, Norway in 2017. He is working toward the PhD degree at RWTH Aachen University, Aachen, Germany. His research interests include the physics-based simulation of deformable solids, rigid bodies and fluids, with a particular emphasis on finite element methods, and numerical optimization.

![](images/8d2495769b31be25f53e419b873c1833593f31d1139e1ad6ced8d25a803df4bc.jpg)

Nils Thuerey is an associate professor with the Technical University of Munich (TUM), Munchen, €Germany. He works in the field of computer graphics, where a central theme of his research are physics simulations and deep learning algorithms. He received a tech-Oscar from the AMPAS in 2013 for his research on controllable smoke effects. He worked for three years as a post-doc at ETH Zurich and as R&D lead at ScanlineVFX, before starting at TUM in October 2013.

![](images/e7e4106a12d7379888f835e48d25622bc9c029b36c9d1dc31d551184dca4c4e7.jpg)

Jan Bender received the diploma, PhD, and habilitation degrees in computer science from the University of Karlsruhe, Karlsruhe, Germany. He is a professor of computer science and leader of the Computer Animation Group at RWTH Aachen University, Aachen, Germany. His research interests include interactive simulation methods, multibody systems, deformable solids, fluid simulation, collision handling, cutting, fracture, GPGPU, and realtime visualization.

$\vartriangleright$ For more information on this or any other computing topic, please visit our Digital Library at www.computer.org/csdl.