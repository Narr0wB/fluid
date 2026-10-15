use glam::Vec3;

pub struct FluidMaterial {
    pub color: Vec3,
    pub particle_radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FluidParams {
    color: [f32; 3],
    particle_radius: f32,
}

pub struct FluidInstance {
    count: u32,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub struct FluidRenderer {
    fluid_layout: wgpu::BindGroupLayout,
    sphere_pipeline: wgpu::RenderPipeline,
}

impl FluidRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat, camera_layout: &wgpu::BindGroupLayout) -> Self {
        todo!()
    }

    pub fn create_instance(
        &self,
        device: &wgpu::Device,
        particles: &wgpu::Buffer,
        count: u32,
        material: FluidMaterial,
    ) -> FluidInstance {
        todo!()
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, camera: &wgpu::BindGroup, fluids: &[&FluidInstance]) {
        todo!()
    }
}
