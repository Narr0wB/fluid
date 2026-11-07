use glam::{UVec3, Vec3};
use wgpu::util::DeviceExt;

use crate::gpu::{self, GpuMirror};
use crate::renderer::mesh::{ColliderUniform, Mesh};

pub const SMOOTHING_RADIUS: f32 = 0.08;
/// Edge length of a spatial grid cell. Equal to H so all neighbours lie in the surrounding 3x3x3 cells.
pub const SPATIAL_CELL_SIZE: f32 = SMOOTHING_RADIUS;
const WORKGROUP_SIZE: u32 = 64;

#[derive(Clone, Copy, Debug)]
pub struct SimulationEnvironment {
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub collision_radius: f32,
}

impl SimulationEnvironment {
    pub fn grid_dims(&self) -> UVec3 {
        ((self.bounds_max - self.bounds_min) / SPATIAL_CELL_SIZE).floor().as_uvec3() + 1
    }

    pub fn cell_count(&self) -> u32 {
        self.grid_dims().element_product()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSimulationEnvironment {
    bounds_min: [f32; 3],
    dt: f32,
    bounds_max: [f32; 3],
    spatial_cell_size: f32,
    grid_dims: [u32; 3],
    max_speed: f32,
    collision_radius: f32,
    _pad: [f32; 3],
}

impl GpuMirror for SimulationEnvironment {
    type Gpu = GpuSimulationEnvironment;
    type Context = f32;

    fn to_gpu(&self, dt: f32) -> GpuSimulationEnvironment {
        GpuSimulationEnvironment {
            bounds_min: self.bounds_min.to_array(),
            dt,
            grid_dims: self.grid_dims().to_array(),
            bounds_max: self.bounds_max.to_array(),
            spatial_cell_size: SPATIAL_CELL_SIZE,
            max_speed: 0.4 * SMOOTHING_RADIUS / dt,
            collision_radius: self.collision_radius,
            _pad: [0.0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    pub density: f32,
    pub velocity: [f32; 3],
    pub pressure: f32,
}

pub fn particle_cube(spacing: f32, start: Vec3, velocity: Vec3, side: u32) -> Vec<Particle> {
    let mut state = 0x9E37_79B9u32 ^ side;
    let mut jitter = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32 - 0.5) * 0.5 * spacing
    };

    let mut particles = Vec::with_capacity((side * side * side) as usize);
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                let lattice = Vec3::new(i as f32, k as f32, j as f32) * spacing;
                let offset = Vec3::new(jitter(), jitter(), jitter());
                particles.push(Particle {
                    position: (start + lattice + offset).to_array(),
                    density: 0.0,
                    velocity: velocity.to_array(),
                    pressure: 0.0,
                });
            }
        }
    }
    particles
}

#[derive(Clone, Copy)]
pub struct FluidSettings {
    pub density: f32,
    pub pressure_const: f32,
    pub viscosity: f32,
    pub mass: f32,
    pub restitution: f32,
    pub friction: f32,
}


#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFluidParams {
    count: u32,
    active_buffer: u32,
    viscosity: f32,
    density: f32,
    pressure_const: f32,
    mass: f32,
    restitution: f32,
    friction: f32,
}


impl GpuMirror for FluidSettings {
    type Gpu = GpuFluidParams;
    type Context = (u32, u32);

    fn to_gpu(&self, (count, active_buffer): (u32, u32)) -> GpuFluidParams {
        GpuFluidParams {
            count,
            active_buffer,
            viscosity: self.viscosity,
            density: self.density,
            pressure_const: self.pressure_const,
            mass: self.mass,
            restitution: self.restitution,
            friction: self.friction,
        }
    }
}

pub struct Fluid {
    settings: FluidSettings,
    initial: Vec<Particle>,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    particles: [wgpu::Buffer; 2],
    active: usize,
}

impl Fluid {
    pub fn new(pipeline: &FluidComputePipeline, particles: Vec<Particle>, settings: FluidSettings) -> Self {
        let device = &pipeline.device;

        let particle_buffer_1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles_1"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let particle_buffer_2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_2"),
            size: particle_buffer_1.size() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let accelerations = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accelerations"),
            size: (particles.len() * size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid params"),
            size: size_of::<GpuFluidParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid compute"),
            layout: &pipeline.fluid_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buffer_1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buffer_2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: accelerations.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
            ],
        });

        Fluid {
            particles: [particle_buffer_1, particle_buffer_2],
            active: 0,
            initial: particles,
            settings,
            params_buffer,
            bind_group,
        }
    }

    /// The buffer holding the current particle state. Changes after every `step`.
    pub fn particles(&self) -> &wgpu::Buffer {
        &self.particles[self.active]
    }

    pub fn buffers(&self) -> &[wgpu::Buffer; 2] {
        &self.particles
    }

    fn swap_buffers(&mut self) {
        self.active ^= 1;
    }

    fn upload_params(&self, queue: &wgpu::Queue) {
        self.settings.upload(queue, &self.params_buffer, (self.count(), self.active as u32));
    }

    pub fn count(&self) -> u32 {
        self.initial.len() as u32
    }

    pub fn reset(&mut self, queue: &wgpu::Queue) {
        self.active = 0;
        queue.write_buffer(self.particles(), 0, bytemuck::cast_slice(&self.initial));
    }

    pub fn active(&self) -> usize {
        self.active
    }

}


pub struct FluidComputePipeline {
    env: SimulationEnvironment,

    device: wgpu::Device,
    queue: wgpu::Queue,

    /// Sorting steps
    count_pipeline: wgpu::ComputePipeline,
    scan_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,

    /// Compute steps
    density_pipeline: wgpu::ComputePipeline,
    force_pipeline: wgpu::ComputePipeline,
    integration_pipeline: wgpu::ComputePipeline,
    collision_pipeline: wgpu::ComputePipeline,

    environment_layout: wgpu::BindGroupLayout,
    fluid_layout: wgpu::BindGroupLayout,
    collider_layout: wgpu::BindGroupLayout,

    environment_buffer: wgpu::Buffer,
    environment_bind_group: wgpu::BindGroup,
    indices: wgpu::Buffer,
}

impl FluidComputePipeline {
    pub fn new(gpu: &gpu::Gpu, environment: SimulationEnvironment) -> Self {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fluid step"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compute.wgsl").into()),
        });

        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let fluid_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid compute layout"),
            entries: &[
                storage(0),
                storage(1),
                storage(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<GpuFluidParams>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let collider_layout = Self::create_collider_layout(&device);

        let environment_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("environment layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<GpuSimulationEnvironment>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let step_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fluid step layout"),
            bind_group_layouts: &[Some(&environment_layout), Some(&fluid_layout)],
            immediate_size: 0,
        });

        let collision_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fluid collision layout"),
            bind_group_layouts: &[Some(&environment_layout), Some(&fluid_layout), Some(&collider_layout)],
            immediate_size: 0,
        });

        let constants = [("H", SMOOTHING_RADIUS as f64)];
        let pipeline = |entry: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        };

        let environment_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim_env"),
            size: size_of::<GpuSimulationEnvironment>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell indices"),
            size: (environment.cell_count() as usize * size_of::<[u32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let environment_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("environment"),
            layout: &environment_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: environment_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: indices.as_entire_binding() },
            ],
        });

        FluidComputePipeline {
            count_pipeline: pipeline("count", &step_layout),
            scan_pipeline: pipeline("scan", &step_layout),
            scatter_pipeline: pipeline("scatter", &step_layout),
            density_pipeline: pipeline("density", &step_layout),
            force_pipeline: pipeline("pressure", &step_layout),
            integration_pipeline: pipeline("integration", &step_layout),
            collision_pipeline: pipeline("collide", &collision_layout),
            device,
            queue,
            environment_layout,
            fluid_layout,
            collider_layout,
            environment_buffer,
            environment_bind_group,
            indices,
            env: environment,
        }
    }


    pub fn step(&self, fluid: &mut Fluid, colliders: &[&Mesh], dt: f32) {
        let workgroups = fluid.count().div_ceil(WORKGROUP_SIZE);
        self.env.upload(&self.queue, &self.environment_buffer, dt);

        // Sort: reads particles[active], scatters into particles[active ^ 1].
        // Queue writes only land at the next submit, so the sort gets its own submit
        // to let the simulation passes below see the flipped `active_buffer`.
        fluid.upload_params(&self.queue);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fluid sort") });
        {
            encoder.clear_buffer(&self.indices, 0, Some(self.indices.size()));

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &self.environment_bind_group, &[]);
            pass.set_bind_group(1, &fluid.bind_group, &[]);

            for (pipeline, wg) in [(&self.count_pipeline, workgroups), (&self.scan_pipeline, 1), (&self.scatter_pipeline, workgroups)] {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(wg, 1, 1);
            }
        }
        self.queue.submit([encoder.finish()]);

        // The sorted buffer becomes the active one for the rest of the step.
        fluid.swap_buffers();
        fluid.upload_params(&self.queue);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fluid step") });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &self.environment_bind_group, &[]);
            pass.set_bind_group(1, &fluid.bind_group, &[]);

            for pipeline in [&self.density_pipeline, &self.force_pipeline, &self.integration_pipeline] {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            pass.set_pipeline(&self.collision_pipeline);
            for sdf in colliders.iter().filter_map(|mesh| mesh.sdf()) {
                pass.set_bind_group(2, sdf.bind_group(), &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        self.queue.submit([encoder.finish()]);
    }

    pub fn collider_layout(&self) -> &wgpu::BindGroupLayout {
        &self.collider_layout
    }

    fn create_collider_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("collider layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<ColliderUniform>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Instant;

    use glam::Mat4;

    use super::*;
    use crate::renderer::mesh::{MeshMaterial, create_model_layout};

    const DT: f32 = 0.004;
    const SPACING: f32 = 0.035;
    const RADIUS: f32 = 0.0175;

    fn settings() -> FluidSettings {
        FluidSettings {
            density: 1000.0,
            pressure_const: 3000.0,
            viscosity: 0.03,
            mass: 1000.0 * SPACING.powi(3),
            restitution: 0.3,
            friction: 0.1,
        }
    }

    fn environment() -> SimulationEnvironment {
        SimulationEnvironment {
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(2.0, 6.0, 2.0),
            collision_radius: RADIUS,
        }
    }

    fn read_particles(gpu: &gpu::Gpu, fluid: &Fluid) -> Vec<Particle> {
        let size = fluid.particles().size();
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(fluid.particles(), 0, &staging, 0, size);
        gpu.queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let view = staging.slice(..).get_mapped_range().unwrap();
        bytemuck::cast_slice(&view).to_vec()
    }

    fn torus_distance(position: Vec3, center: Vec3) -> f32 {
        let local = position - center;
        let ring = (local.x * local.x + local.z * local.z).sqrt() - 0.5;
        (ring * ring + local.y * local.y).sqrt() - 0.2
    }

    fn torus_collider(gpu: &gpu::Gpu, pipeline: &FluidComputePipeline, center: Vec3) -> Mesh {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/torus.obj");
        let material = MeshMaterial { color: Vec3::ONE, checker: false };
        let layout = create_model_layout(&gpu.device);
        let mesh = Mesh::load(&gpu.device, &gpu.queue, &layout, pipeline.collider_layout(), &path, material).unwrap();
        mesh.set_model(&gpu.queue, Mat4::from_translation(center), Vec3::ZERO);
        mesh
    }

    #[test]
    fn fluid_stays_bounded_and_outside_mesh() {
        let gpu = pollster::block_on(gpu::Gpu::headless()).unwrap();
        let pipeline = FluidComputePipeline::new(&gpu, environment());
        let center = Vec3::new(1.0, 0.6, 1.0);
        let torus = torus_collider(&gpu, &pipeline, center);
        let particles = particle_cube(SPACING, Vec3::new(0.7, 1.4, 0.7), Vec3::ZERO, 16);
        let settings = settings();
        let mut fluid = Fluid::new(&pipeline, particles, settings);

        for _ in 0..1500 {
            pipeline.step(&mut fluid, &[&torus], DT);
        }

        let result = read_particles(&gpu, &fluid);
        for p in &result {
            let position = Vec3::from(p.position);
            assert!(position.is_finite() && Vec3::from(p.velocity).is_finite(), "{p:?}");
            assert!(position.cmpge(pipeline.env.bounds_min).all() && position.cmple(pipeline.env.bounds_max).all(), "{p:?}");
            assert!(torus_distance(position, center) > RADIUS * 0.5, "inside mesh: {p:?}");
        }
        assert!(result.iter().any(|p| Vec3::from(p.position).y < 0.2), "fluid never reached the floor");
    }

    #[test]
    fn reset_restores_initial_particles() {
        let gpu = pollster::block_on(gpu::Gpu::headless()).unwrap();
        let pipeline = FluidComputePipeline::new(&gpu, environment());
        let particles = particle_cube(SPACING, Vec3::new(0.5, 1.0, 0.5), Vec3::ZERO, 6);
        let start = particles[0].position;
        let mut fluid = Fluid::new(&pipeline, particles, settings());

        for _ in 0..50 {
            pipeline.step(&mut fluid, &[], DT);
        }
        assert_ne!(read_particles(&gpu, &fluid)[0].position, start);

        fluid.reset(&gpu.queue);
        assert_eq!(read_particles(&gpu, &fluid)[0].position, start);
    }

    #[test]
    #[ignore = "full-scale run, prints stability and timing figures"]
    fn full_scene_report() {
        let gpu = pollster::block_on(gpu::Gpu::headless()).unwrap();
        let pipeline = FluidComputePipeline::new(&gpu, environment());
        let center = Vec3::new(1.0, 0.85, 1.0);
        let torus = torus_collider(&gpu, &pipeline, center);
        let mut particles = particle_cube(SPACING, Vec3::new(0.5, 2.5, 0.5), Vec3::ZERO, 30);
        particles.extend(particle_cube(SPACING, Vec3::new(0.8, 5.0, 0.8), Vec3::ZERO, 10));
        let mut fluid = Fluid::new(&pipeline, particles, settings());

        for chunk in 0..8 {
            let start = Instant::now();
            for _ in 0..250 {
                pipeline.step(&mut fluid, &[&torus], DT);
            }
            let result = read_particles(&gpu, &fluid);
            let per_step = start.elapsed().as_secs_f32() * 1000.0 / 250.0;
            let max_density = result.iter().map(|p| p.density).fold(0.0, f32::max);
            let max_speed = result.iter().map(|p| Vec3::from(p.velocity).length()).fold(0.0, f32::max);
            let mean_y = result.iter().map(|p| p.position[1]).sum::<f32>() / result.len() as f32;
            let finite = result.iter().all(|p| Vec3::from(p.position).is_finite());
            println!(
                "t={:.1}s step={per_step:.2}ms max_rho={max_density:.0} max_v={max_speed:.1} mean_y={mean_y:.2} finite={finite}",
                (chunk + 1) as f32 * 250.0 * DT,
            );
        }
    }
}
