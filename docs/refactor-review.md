# Refactor review: fluid-sim

Date: 2026-09-28
Scope: everything under `src/` at commit `31ed43a`, plus the uncommitted move of the shaders to `src/shaders/`.
Method: read all the code, ran `cargo check` (it builds with 28 warnings), and ran quick numeric checks of the SPH parameters. The crate versions in section 3 come from crates.io and GitHub as of this date.

---

## 0. Summary

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 1 | Brute-force O(N^2) neighbour search: 28,000 particles gives about 1.57 billion pair evaluations per frame | The main frame-time cost | Medium (spatial grid + counting sort) |
| 2 | Particle mass and target density don't match: the initial lattice sits at 23.5% of rest density | The fluid starts under strong negative pressure and clumps; wrong physics | Trivial (change one constant) |
| 3 | The frame loop is fully serial: CPU waits for compute, then for render, then for present | GPU sits idle while the CPU records; CPU sits idle while the GPU runs | Medium |
| 4 | Allocators, command pools and descriptor sets are recreated every frame or on every construction | CPU overhead and extra VRAM blocks | Small |
| 5 | Sphere mesh indices run past the end of the vertex buffer | Undefined behaviour in the vertex fetch | Trivial |
| 6 | Render submission probably doesn't wait on the swapchain-acquire semaphore | Sync hazard; validation layers should report it | Small |
| 7 | 200 triangles per particle (5.6M triangles per frame) plus a 64-byte mat4 per instance | Raster and bandwidth cost that grows with N | Small (impostor billboards) |
| 8 | Requires `VK_EXT_shader_atomic_float`, but no shader uses float atomics | Rules out GPUs for no reason | Trivial |

Recommended stack for the refactor: **wgpu 30 + WGSL + glam + bytemuck/encase + winit 0.30**. Details are in section 3.

Housekeeping first:
- Commit the shader move. `git status` shows the old `src/compute_*.glsl` files as deleted and `src/shaders/` as untracked. The new files are byte-identical to the old ones.
- Run `cargo clean`. The build-script binaries in `target/` are SIGKILLed at launch (probably stale code signatures after a macOS update). A fresh target directory builds fine.
- The default toolchain was rustc 1.79 (June 2024). It is now 1.89.0 (August 2025). Check the minimum Rust version (`rust-version`) of wgpu 30 and winit 0.30 before migrating; the newest releases may need a later compiler than 1.89.
- `src/tmp.txt` is a tracked scratch file of old constants. Delete it.

---

## 1. Performance bottlenecks

### 1.1 O(N^2) neighbour search (largest cost)

[compute_density.glsl:37](../src/shaders/compute_density.glsl#L37) and [compute_pressure.glsl:61](../src/shaders/compute_pressure.glsl#L61) both loop over every particle for every particle. With `side = 30` plus `side = 10` cubes ([main.rs:147-148](../src/main.rs#L147-L148)), N = 28,000:

- 2 x N^2 = **1.57 x 10^9 pair iterations per frame**, each loading a 32-byte `Particle` from global memory.
- Each particle has about 57 real neighbours inside `h`, so over 99.8% of those iterations are rejected by the distance test.

**Fix: a uniform grid with cell size = h, filled by a counting sort.** This is the method in the Ihmsen et al. paper the README already cites. Per step:

1. `hash`: compute each particle's cell index (a linear index into a bounded grid; the domain is a fixed box, so no hashing is needed).
2. `count`: `atomicAdd` on a `u32` counter per cell. Integer atomics only; no float atomics.
3. `scan`: exclusive prefix sum over the cell counts. A simple two-level workgroup scan is enough for tens of thousands of cells.
4. `scatter`: write the particles into cell order (sorted position and velocity arrays).
5. `density` and `forces`: loop over the 27 neighbouring cells.

Expected work drops from N^2 to roughly N x 27 x (particles per cell), a few hundred times less at this N. Sorting the particle data itself, not just an index list, also makes neighbour reads coherent in memory. That matters as much as the algorithmic change.

If you keep brute force for a while as a reference, at least tile it through workgroup shared memory (the classic N-body pattern). Load 64 positions per tile into shared memory and loop over them. That cuts global memory traffic by about 64x.

### 1.2 The frame is fully serial

Per frame the CPU does:

1. [fluid.rs:548-554](../src/fluid.rs#L548-L554): submit compute, then `finished.wait(None)` (blocks).
2. [renderer.rs:725-729](../src/renderer.rs#L725-L729): submit render, then `wait_future.wait(None)` (blocks).
3. [renderer.rs:731-736](../src/renderer.rs#L731-L736): present, then `then_signal_fence_and_flush()`, and drop the result. In vulkano, dropping a flushed `FenceSignalFuture` blocks until the fence signals, so this is a third stall.

The CPU and GPU never overlap, and there is only ever one frame in flight.

**Fix:** record one command buffer per frame containing the compute dispatches, then the render pass. Submit it once, chained on the acquire future, and keep 2 frames in flight. In vulkano that means keeping `previous_frame_end: Option<Box<dyn GpuFuture>>` and calling `cleanup_finished()`. In wgpu it is the default behaviour. If compute for frame N+1 writes the buffer that frame N's render is still reading, double-buffer the instance buffer or rely on the barrier inside the single submission.

### 1.3 Objects recreated every frame

| Location | What is recreated | Fix |
|----------|-------------------|-----|
| [fluid.rs:465](../src/fluid.rs#L465) | `StandardCommandBufferAllocator` (new command pools) every compute step | Create it once and store it |
| [fluid.rs:467-546](../src/fluid.rs#L467-L546) | Three primary command buffers and three submissions for three dispatches | One command buffer with three dispatches. `AutoCommandBufferBuilder` inserts the barriers between them |
| [renderer.rs:595-607](../src/renderer.rs#L595-L607) | `StandardDescriptorSetAllocator` and the particle descriptor set, every frame | The buffer never changes, so build the set once |
| [renderer.rs:620](../src/renderer.rs#L620), [renderer.rs:654](../src/renderer.rs#L654) | A secondary command buffer per draw call | Record inline into the primary buffer (`SubpassContents::Inline`) |
| [fluid.rs:149](../src/fluid.rs#L149), [fluid.rs:242](../src/fluid.rs#L242), [mesh.rs:30](../src/mesh.rs#L30), [renderer.rs:200](../src/renderer.rs#L200) | A separate `StandardMemoryAllocator` in each place (four or more instances; `bind_compute` creates a new one on every reset) | Create one `Arc<StandardMemoryAllocator>` and pass it around. Each allocator reserves its own large device-memory blocks |
| [fluid.rs:296](../src/fluid.rs#L296) | Descriptor set allocator per `bind_compute` | Share one |

### 1.4 Host-side buffer writes

- [fluid.rs:281-294](../src/fluid.rs#L281-L294): `write().unwrap()` is called on both buffers inside the per-particle loop, which is 56,000 guard acquire/release cycles. Each drop can also flush non-coherent memory. Take the guards once, outside the loop. Better still, build a `Vec<Particle>` and create the buffer with `Buffer::from_iter`. With `bytemuck` it is a single `copy_from_slice`.
- The loop zero-fills the force buffer, which isn't needed because the pressure pass overwrites it.
- Every simulation buffer is `PREFER_DEVICE | HOST_SEQUENTIAL_WRITE`. The GPU owns this data after the first upload, so make these buffers device-only and upload once through a staging buffer. On Apple Silicon (unified memory) this makes no difference. On a discrete GPU without Resizable BAR it keeps compute reads out of the small BAR heap or host memory.

### 1.5 Shader-level inefficiencies

- [compute_pressure.glsl:64-66](../src/shaders/compute_pressure.glsl#L64-L66): `rho_j` is loaded and `length()` (a sqrt) is computed before the cutoff test. Test `dot(d, d) > h2` first, and load `rho_j` only after it passes.
- [compute_pressure.glsl:71](../src/shaders/compute_pressure.glsl#L71): `density_to_pressure(rho_j)` computes `pow(x, 7)` for every neighbour pair. Compute each particle's pressure once at the end of the density pass and store it in the `pressure` field. That field exists in `Particle` but is never written.
- `pressure_i / (rho_i * rho_i)` is loop-invariant; hoist it. `dir / max(dist, eps)` right after `if (dist < eps) continue` is redundant.
- Kernel constants (`pow(h, 9)`, `pow(h, 6)`) are recomputed in every invocation. Compute them on the CPU and pass them as push constants.
- The density pass reads only positions, but `Particle` is 32 bytes (position, density, velocity, pressure), so half of every load is wasted. Use structure-of-arrays: separate `vec4 pos` (w = density or pressure) and `vec4 vel` buffers.
- [fluid.rs:508-510](../src/fluid.rs#L508-L510): dispatch size is hard-coded to 3000 x 64 = 192,000 invocations for 28,000 particles. The grid-stride loop keeps this correct, but about 85% of the threads exit immediately. Use `ceil(N / 64)` workgroups.

### 1.6 Particle rendering

- [main.rs:145](../src/main.rs#L145) plus [mesh.rs:79](../src/mesh.rs#L79): a UV sphere with 200 triangles per instance comes to **5.6M triangles per frame**. Draw a camera-facing quad per particle (4 vertices, no vertex buffer at all), then discard fragments outside the circle and write the sphere's normal and depth in the fragment shader. That is about 100x less geometry and looks smoother.
- Each instance reads a 64-byte `mat4` ([compute_positions.glsl:97-98](../src/shaders/compute_positions.glsl#L97-L98)), of which only position and speed are used. A `vec4(pos, speed)` is 16 bytes. The render shader can then read the simulation's position buffer directly, and the positions pass no longer has to build matrices.

### 1.7 Minor

- [main.rs:257](../src/main.rs#L257): `println!` every frame (re-enabled in the working tree) does blocking stdout I/O inside the frame. Print once a second, or put the timing in the window title.
- `Cargo.toml` has no profile tuning. Add `[profile.dev.package."*"] opt-level = 2` so dependencies (nalgebra, vulkano) are optimized in dev builds.
- vulkano-shaders builds shaderc from source when no system copy is found, which is a large part of clean build time. wgpu and WGSL remove that step.

---

## 2. Wrong or inefficient code

### 2.1 Physics and simulation correctness

**(a) Rest density does not match the particle mass.** This is the most important correctness issue.
`PARTICLE_MASS = 0.01`, `h = 0.08`, and the initial spacing `d = cbrt(4*pi*h^3 / (3*50)) = 0.035` ([main.rs:146](../src/main.rs#L146)). Summing poly6 over that lattice gives **235 kg/m^3**, but `target_density = 1000` ([main.rs:139](../src/main.rs#L139)). The fluid therefore starts at 23.5% of rest density. The Tait equation of state clamps the ratio at 0.5 ([compute_pressure.glsl:43](../src/shaders/compute_pressure.glsl#L43)), so every particle starts at about -99 Pa. That is a uniform attraction, which causes clumping and the tensile instability.
Fix: derive the mass from the spacing, `mass = rho0 * d^3 = 0.0425 kg`, or compute it at startup by normalizing the lattice density sum. Also clamp pressure at 0 (`max(0, B*((rho/rho0)^7 - 1))`), or use a proper surface-tension or tensile-correction term instead of negative pressure.

**(b) Viscosity mixes force and acceleration.** [compute_pressure.glsl:77-81](../src/shaders/compute_pressure.glsl#L77-L81): the symmetric pressure term is already an acceleration. The viscosity term is the Müller force density `mu * sum m (vj - vi) / rho_j * lapW`, which still has to be divided by `rho_i`. It is added unscaled, so viscosity is about 1000x (rho_i) stronger than `viscosity_constant` implies. Gravity is added as an acceleration, and the positions pass treats the buffer as acceleration. The buffer is named `forces`, but it holds accelerations; rename it.

**(c) The timestep breaks CFL during impact.** `dt = 0.004` s ([main.rs:142](../src/main.rs#L142)) is derived from the viscous limit only. The second cube falls about 8.5 m and hits at about 12.9 m/s. The CFL limit `0.4 h / (v + c)` is then about **0.0023 s**. Use an adaptive dt, or several fixed substeps per frame.

**(d) Simulation speed depends on frame rate.** There is one fixed `dt` step per rendered frame, so the simulation runs slower in wall-clock time when FPS drops. Use a time accumulator with fixed substeps.

**(e) The initial jitter is not centred.** [fluid.rs:92-94](../src/fluid.rs#L92-L94): `(rand * 0.5 - 1.0) * sep` falls in `[-1.0, -0.5] * sep`. That is a constant offset of -0.75 sep, plus a +/-0.25 jitter. You probably meant `(rand - 0.5) * 0.5 * sep`. There is also a hard-coded `- 1.5` at [fluid.rs:100-102](../src/fluid.rs#L100-L102).

**(f) Two bounding boxes.** The reset path uses `x2: 5, z2: 5, damping 0.2` ([main.rs:170](../src/main.rs#L170)), and the frame loop uses `x2: 3, z2: 3, damping 0.3` ([main.rs:247](../src/main.rs#L247)). The drawn wireframe box is 3 x 3 x 3 ([main.rs:136](../src/main.rs#L136)), but the simulation's `y2` is 10, so the second cube spawns outside the drawn box. Define the domain once and derive the draw mesh from it.

**(g) Damping means opposite things.** On the CPU, `velocity *= -(1 - damping)` ([fluid.rs:57](../src/fluid.rs#L57)). On the GPU, `velocity *= -damping` ([compute_positions.glsl:61](../src/shaders/compute_positions.glsl#L61)). Pick one meaning (for example "restitution") and rename the field.

### 2.2 Dead or broken code

- `Fluid::update` ([fluid.rs:180-239](../src/fluid.rs#L180-L239)), `Particle::update`, `smoothing_kernel*`, `convert_density_pressure` and `GRAVITY` are unused (compiler warnings). The CPU path is also broken:
  - [fluid.rs:201-219](../src/fluid.rs#L201-L219): the pressure force is summed into a local that is then discarded. `pressure_forces` stays zero, so the pressure acceleration is always 0.
  - The CPU path uses a different, non-symmetric pressure formula and a linear EOS, not Tait.
  - It encodes speed in matrix element `(0,0)` (the x scale), while the GPU path uses `[0][1]`.
  - `Particle::update` ignores `y2` and snaps particles a full `h` inside the wall.

  Either delete it, or fix it and keep it as a small-N reference implementation for GPU tests. The second option is worth it: a CPU/GPU comparison test on 500 particles catches shader regressions cheaply.
- [compute_density.glsl:3-4](../src/shaders/compute_density.glsl#L3-L4) enables `GL_EXT_shader_atomic_float` and `GL_EXT_debug_printf`, but uses neither. [main.rs:44](../src/main.rs#L44) and [main.rs:90](../src/main.rs#L90) *require* `ext_shader_atomic_float` and `shader_buffer_float32_atomic_add`, so device selection rejects every GPU without them. Remove all three.
- `Renderer::instance`, `images`, `depth_buffer` and `showcase_flag` are stored and never read.

### 2.3 Vulkan and vulkano correctness

- **Sphere mesh reads out of bounds.** [mesh.rs:83-94](../src/mesh.rs#L83-L94) creates `res * res` = 100 vertices, but emits indices up to `(res-1+1)*res + (res-1) + 1 = 110`. The `j + 1` at `j = res-1` also wraps to the next ring. Loop `i` and `j` over `0..res-1` for the quads. This goes away entirely if you switch to impostors (1.6).
- **Render probably doesn't wait on the acquire semaphore.** [renderer.rs:725-736](../src/renderer.rs#L725-L736): the render command buffer is submitted from `sync::now(...)`, not from the future `acquire_next_image` returned. Only the present waits on that future, so the render pass may write to the image before the presentation engine has released it. The normal chain is `acquire_future.then_execute(queue, cmd).then_swapchain_present(...).then_signal_fence_and_flush()`. Enable `VK_LAYER_KHRONOS_validation` to confirm this and the next item.
- **Storage buffer in the vertex shader is not `readonly`.** [renderer.rs:95](../src/renderer.rs#L95): without `readonly`, the SPIR-V lacks `NonWritable`, which the spec only allows when `vertexPipelineStoresAndAtomics` is enabled. Add `readonly`.
- **Swapchain errors are not handled.** [renderer.rs:587](../src/renderer.rs#L587) calls `.expect()` on `acquire_next_image`. It panics on `OutOfDate`, ignores the `suboptimal` flag, and never checks the present result. `recreate_swapchain` ([renderer.rs:528](../src/renderer.rs#L528)) does not guard against a 0 x 0 extent, so minimizing the window will panic.
- [renderer.rs:218](../src/renderer.rs#L218): `min_image_count: 2` is not clamped to `surface_capabilities.min_image_count`/`max_image_count`, and [renderer.rs:211](../src/renderer.rs#L211) takes `surface_formats()[0]` without checking for sRGB.
- [main.rs:61](../src/main.rs#L61): the surface-support check is commented out because the surface is created later, inside `Renderer::init`. Create the window and surface first, then pick the queue family.
- [main.rs:77](../src/main.rs#L77): `cfg!(DEBUG)` is not a real cfg and is always false. Use `cfg!(debug_assertions)`.
- [renderer.rs:1](../src/renderer.rs#L1): `#[allow(unused_imports)]` applies only to the next item (the first `use`), which is why 13 unused-import warnings still show. Delete the unused imports instead.

### 2.4 Camera and input

- [camera.rs:31](../src/camera.rs#L31): `set_orientation` rejects the update when the *new* direction is nearly parallel to the *current* one, so small mouse moves are dropped. It does not stop the camera looking straight up or down, where `UP x forward` becomes zero and `normalize()` produces NaN. Clamp pitch instead, or better, store yaw and pitch angles.
- [renderer.rs:503-515](../src/renderer.rs#L503-L515): the mouse delta is normalized, so rotation speed ignores how far the mouse moved and depends on the event rate.
- [renderer.rs:421-449](../src/renderer.rs#L421-L449): the W/S movement vector (`aheadXZ`) is not renormalized after zeroing `y`, so forward speed drops as you look up or down. Movement is also driven by key-repeat events, not per-frame velocity.
- [camera.rs:80](../src/camera.rs#L80): `1 / tan(fov)` treats `fov` as a half-angle, so "30" is actually a 60-degree vertical FOV. `aspect_ratio` is stored as height/width. Both work, but the names are misleading.

### 2.5 Structure (for the refactor)

- `Fluid` holds `Option<Subbuffer>` and `Option<DescriptorSet>` fields that are only valid after `bind_compute`, and `reset` + `bind_compute` must be called as a pair. Build the GPU state in one constructor, and implement reset as a buffer upload.
- Particle data is packed into raw `f32` slices by hand with offsets ([fluid.rs:285-291](../src/fluid.rs#L285-L291)). Use a `#[repr(C)]` struct that derives `bytemuck::Pod` (or `encase::ShaderType`) so the Rust and shader layouts can't drift apart.
- Simulation parameters are scattered across `fluid.rs` consts, `main.rs` locals and `tmp.txt`. Put them in one `SimParams` struct, uploaded as a uniform or push constant.
- `unwrap()` is used everywhere. Use `anyhow` in `main` and return `Result` from setup code.

---

## 3. GPU stack options

Versions are from crates.io and GitHub as of 2026-09-28. Items marked *unverified* could not be confirmed.

| Crate | Latest | Status | Fit for this project |
|-------|--------|--------|----------------------|
| **wgpu** | 30.0.1 (2026-08-22) | Very active (backs Bevy) | **Best fit.** Native Metal backend on macOS (no MoltenVK), Vulkan/DX12 elsewhere. Compute, storage buffers, timestamp queries, push constants (`Features::PUSH_CONSTANTS`, native only), most subgroup ops |
| vulkano | 0.35.2 (2025-08-16) | Maintained, commits in Sept 2026 | Smallest migration from today. `vulkano-taskgraph` 0.35.1 describes itself as experimental. Still goes through MoltenVK on macOS |
| ash | 0.38.0+1.3.281 (2024-04, crates.io) | Git main active; releases infrequent | Maximum control but you write your own sync, descriptors and allocation (use with `gpu-allocator` 0.28). Too much boilerplate for this project |
| CubeCL | 0.11.0-pre.4 (2026-09-22) | Active, pre-release | Rust DSL kernels targeting CUDA, ROCm, Metal and wgpu. Built around Burn's tensor workloads; its own team says it has rough edges. No rendering |
| rust-gpu | community org | Active; README says not production-ready | Shaders in Rust compiled to SPIR-V. Interesting for sharing CPU/GPU code, risky as a foundation |
| krnl | - | *unverified* activity | Skip |
| rust-cuda | - | Active, NVIDIA only | Not relevant for macOS |
| winit | 0.30.13 stable; 0.31 beta | Active | 0.30 uses the `ApplicationHandler` trait API, a breaking change from 0.28's closure API |
| glam | 0.33.11 (2026-09-27) | Active | The standard graphics math crate with wgpu. nalgebra (0.35.0) is heavier than you need |

On macOS, MoltenVK remains the default Vulkan-on-Metal layer. KosmicKrisp (Mesa/LunarG) is a new Vulkan 1.4-conformant alternative that needs macOS 26+ and Apple Silicon. Whether MoltenVK's float-atomic support is complete today is *unverified*, but that doesn't matter here once the unused requirement is removed (2.2).

### Recommendation: wgpu + WGSL + glam + bytemuck (or encase) + winit 0.30

Why it fits:
- On macOS, wgpu talks to Metal directly. That removes the MoltenVK translation layer and its extension gaps, the main platform risk in the current setup.
- The planned grid pipeline (hash, integer-atomic count, prefix scan, scatter, neighbour loops) uses only core WGSL: `atomic<u32>`, workgroup memory and storage buffers. It needs no extensions.
- Synchronization between passes, and frames in flight, are handled for you. Several of the bugs in 2.3 can't happen in wgpu's model.
- No shaderc/C++ build step, which means faster clean builds.
- It is the same stack as Bevy, so compute + instanced-rendering examples are plentiful.

Costs:
- The three shaders need porting to WGSL. They are short (about 230 lines in total), and a rewrite is planned anyway.
- winit 0.30 changes event-loop setup.
- Push constants are native-only, which is fine since there's no web target. A uniform buffer works everywhere.

**Second choice: vulkano 0.35.2.** Choose this if you want to keep GLSL and the Vulkan mental model. You still need every fix from section 2.3 and the frame restructuring in 1.2, and macOS still depends on MoltenVK. Avoid `vulkano-taskgraph` until it leaves experimental status.

### Suggested migration order

1. Housekeeping (section 0) and the one-line physics fixes (2.1a, 2.1e, 2.2 atomic-float removal). Do these on the current stack so there is a correct baseline to compare against.
2. Restore the CPU reference implementation (fixed) for a small N and add a CPU/GPU comparison test.
3. Port to wgpu + winit 0.30 + glam with the *same* brute-force algorithm. Compare against the baseline.
4. Replace brute force with the counting-sort grid (1.1) and switch to structure-of-arrays buffers (1.5).
5. Impostor rendering (1.6), fixed-substep timestepping (2.1c/d), and GPU timestamp queries per pass to measure each change.
6. Optional: move to a solver that tolerates larger timesteps (PBF, DFSPH or IISPH) once the neighbour search is fast.

Sources: https://crates.io/crates/wgpu, https://github.com/gfx-rs/wgpu, https://crates.io/crates/vulkano, https://crates.io/crates/vulkano-taskgraph, https://crates.io/crates/ash, https://crates.io/crates/gpu-allocator, https://github.com/tracel-ai/cubecl, https://github.com/Rust-GPU/rust-gpu, https://crates.io/crates/winit, https://crates.io/crates/glam, https://www.lunarg.com/the-state-of-vulkan-on-apple-jan-2026/, https://docs.mesa3d.org/drivers/kosmickrisp.html
