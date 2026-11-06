use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::gpu;
use crate::renderer::mesh::{ColliderUniform, Mesh};

pub const SMOOTHING_RADIUS: f32 = 0.08;
const WORKGROUP_SIZE: u32 = 64;

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
    pub collision_radius: f32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    bounds_min: [f32; 3],
    count: u32,
    bounds_max: [f32; 3],
    viscosity: f32,
    density: f32,
    pressure_const: f32,
    mass: f32,
    dt: f32,
    restitution: f32,
    friction: f32,
    collision_radius: f32,
    max_speed: f32,
}

pub struct Fluid {
    particles: wgpu::Buffer,
    initial: Vec<Particle>,
    settings: FluidSettings,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Fluid {
    pub fn new(pipeline: &FluidComputePipeline, particles: Vec<Particle>, settings: FluidSettings) -> Self {
        let device = &pipeline.device;
        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let accelerations = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accelerations"),
            size: (particles.len() * size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid params"),
            size: size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid compute"),
            layout: &pipeline.fluid_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: accelerations.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        Fluid {
            particles: particle_buffer,
            initial: particles,
            settings,
            params_buffer,
            bind_group,
        }
    }

    pub fn particles(&self) -> &wgpu::Buffer {
        &self.particles
    }

    pub fn count(&self) -> u32 {
        self.initial.len() as u32
    }

    pub fn reset(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.particles, 0, bytemuck::cast_slice(&self.initial));
    }

    fn params(&self, dt: f32) -> Params {
        let s = &self.settings;
        Params {
            bounds_min: s.bounds_min.to_array(),
            count: self.count(),
            bounds_max: s.bounds_max.to_array(),
            viscosity: s.viscosity,
            density: s.density,
            pressure_const: s.pressure_const,
            mass: s.mass,
            dt,
            restitution: s.restitution,
            friction: s.friction,
            collision_radius: s.collision_radius,
            max_speed: 0.4 * SMOOTHING_RADIUS / dt,
        }
    }
}

pub struct FluidComputePipeline {
    device: wgpu::Device,
    queue: wgpu::Queue,

    density_pipeline: wgpu::ComputePipeline,
    force_pipeline: wgpu::ComputePipeline,
    integration_pipeline: wgpu::ComputePipeline,
    collision_pipeline: wgpu::ComputePipeline,

    fluid_layout: wgpu::BindGroupLayout,
    collider_layout: wgpu::BindGroupLayout,
}

impl FluidComputePipeline {
    pub fn new(gpu: &gpu::Gpu) -> Self {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fluid step"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compute_step.wgsl").into()),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<Params>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let collider_layout = Self::create_collider_layout(&device);

        let step_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fluid step layout"),
            bind_group_layouts: &[Some(&fluid_layout)],
            immediate_size: 0,
        });
        let collision_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fluid collision layout"),
            bind_group_layouts: &[Some(&fluid_layout), Some(&collider_layout)],
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

        FluidComputePipeline {
            density_pipeline: pipeline("density", &step_layout),
            force_pipeline: pipeline("pressure", &step_layout),
            integration_pipeline: pipeline("integration", &step_layout),
            collision_pipeline: pipeline("collide", &collision_layout),
            device,
            queue,
            fluid_layout,
            collider_layout,
        }
    }

    pub fn step(&self, fluid: &Fluid, colliders: &[&Mesh], dt: f32) {
        self.queue.write_buffer(&fluid.params_buffer, 0, bytemuck::bytes_of(&fluid.params(dt)));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fluid step") });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            let workgroups = fluid.count().div_ceil(WORKGROUP_SIZE);
            pass.set_bind_group(0, &fluid.bind_group, &[]);

            for pipeline in [&self.density_pipeline, &self.force_pipeline, &self.integration_pipeline] {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            pass.set_pipeline(&self.collision_pipeline);
            for sdf in colliders.iter().filter_map(|mesh| mesh.sdf()) {
                pass.set_bind_group(1, sdf.bind_group(), &[]);
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
            collision_radius: RADIUS,
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(2.0, 6.0, 2.0),
        }
    }

    fn read_particles(gpu: &gpu::Gpu, fluid: &Fluid) -> Vec<Particle> {
        let size = fluid.particles.size();
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&fluid.particles, 0, &staging, 0, size);
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
    fn lattice_starts_near_rest_density() {
        let gpu = pollster::block_on(gpu::Gpu::headless()).unwrap();
        let pipeline = FluidComputePipeline::new(&gpu);
        let particles = particle_cube(SPACING, Vec3::new(0.5, 0.5, 0.5), Vec3::ZERO, 12);
        let fluid = Fluid::new(&pipeline, particles, settings());
        pipeline.step(&fluid, &[], DT);

        let result = read_particles(&gpu, &fluid);
        let center = result[12 * 12 * 6 + 12 * 6 + 6].density;
        assert!((800.0..1250.0).contains(&center), "interior density {center}");
    }

    #[test]
    fn fluid_stays_bounded_and_outside_mesh() {
        let gpu = pollster::block_on(gpu::Gpu::headless()).unwrap();
        let pipeline = FluidComputePipeline::new(&gpu);
        let center = Vec3::new(1.0, 0.6, 1.0);
        let torus = torus_collider(&gpu, &pipeline, center);
        let particles = particle_cube(SPACING, Vec3::new(0.7, 1.4, 0.7), Vec3::ZERO, 16);
        let settings = settings();
        let fluid = Fluid::new(&pipeline, particles, settings);

        for _ in 0..1500 {
            pipeline.step(&fluid, &[&torus], DT);
        }

        let result = read_particles(&gpu, &fluid);
        for p in &result {
            let position = Vec3::from(p.position);
            assert!(position.is_finite() && Vec3::from(p.velocity).is_finite(), "{p:?}");
            assert!(position.cmpge(settings.bounds_min).all() && position.cmple(settings.bounds_max).all(), "{p:?}");
            assert!(torus_distance(position, center) > RADIUS * 0.5, "inside mesh: {p:?}");
        }
        assert!(result.iter().any(|p| Vec3::from(p.position).y < 0.2), "fluid never reached the floor");
    }

    #[test]
    fn reset_restores_initial_particles() {
        let gpu = pollster::block_on(gpu::Gpu::headless()).unwrap();
        let pipeline = FluidComputePipeline::new(&gpu);
        let particles = particle_cube(SPACING, Vec3::new(0.5, 1.0, 0.5), Vec3::ZERO, 6);
        let start = particles[0].position;
        let fluid = Fluid::new(&pipeline, particles, settings());

        for _ in 0..50 {
            pipeline.step(&fluid, &[], DT);
        }
        assert_ne!(read_particles(&gpu, &fluid)[0].position, start);

        fluid.reset(&gpu.queue);
        assert_eq!(read_particles(&gpu, &fluid)[0].position, start);
    }

    #[test]
    #[ignore = "full-scale run, prints stability and timing figures"]
    fn full_scene_report() {
        let gpu = pollster::block_on(gpu::Gpu::headless()).unwrap();
        let pipeline = FluidComputePipeline::new(&gpu);
        let center = Vec3::new(1.0, 0.85, 1.0);
        let torus = torus_collider(&gpu, &pipeline, center);
        let mut particles = particle_cube(SPACING, Vec3::new(0.5, 2.5, 0.5), Vec3::ZERO, 30);
        particles.extend(particle_cube(SPACING, Vec3::new(0.8, 5.0, 0.8), Vec3::ZERO, 10));
        let fluid = Fluid::new(&pipeline, particles, settings());

        for chunk in 0..8 {
            let start = Instant::now();
            for _ in 0..250 {
                pipeline.step(&fluid, &[&torus], DT);
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
