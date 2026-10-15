
use crate::gpu;
use crate::renderer::mesh::ColliderUniform;

pub struct Fluid {
    particles: wgpu::Buffer,
    count: u32,
    density: f32,
    viscosity: f32,
    params_buffer: wgpu::Buffer,
}

impl Fluid {
    pub fn new(particle_count: u32, density: f32, viscosity: f32, device: &wgpu::Device) -> Self {
        let particles = Fluid::create_particles(device, particle_count);
        todo!()
    }

    fn create_particles(device: &wgpu::Device, particle_count: u32) -> wgpu::Buffer {
        todo!()
    }
}

pub struct FluidComputePipeline {
    device: wgpu::Device,
    queue: wgpu::Queue,

    density_pipeline: wgpu::ComputePipeline,
    force_pipeline: wgpu::ComputePipeline,
    integration_pipeline: wgpu::ComputePipeline,

    collider_layout: wgpu::BindGroupLayout,
}

impl FluidComputePipeline {
    pub fn new(gpu: &gpu::Gpu) -> Self {
        todo!()
    }

    pub fn step(&self, fluid: &Fluid, dt: f32) {
        todo!()
    }

    pub fn collider_layout(&self) -> &wgpu::BindGroupLayout {
        &self.collider_layout
    }

    // Matches @group(1) in compute_step.wgsl.
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