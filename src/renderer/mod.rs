pub mod fluid;
pub mod mesh;

use std::sync::Arc;

use glam::Mat4;
use winit::window::Window;

use crate::camera::Camera;
use crate::gpu;

use fluid::{FluidInstance, FluidMaterial, FluidRenderer};
use mesh::{Mesh, MeshRenderer, Topology, Vertex};

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const SCENE_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    view_projection: [[f32; 4]; 4],
    position: [f32; 4],
}

impl CameraUniform {
    fn from_camera(camera: &Camera) -> Self {
        todo!()
    }
}

pub struct Renderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,

    camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    depth: wgpu::TextureView,
    scene_color: wgpu::TextureView,

    sampler: wgpu::Sampler,
    present_layout: wgpu::BindGroupLayout,
    present_bind_group: wgpu::BindGroup,
    present_pipeline: wgpu::RenderPipeline,

    meshes: MeshRenderer,
    fluids: FluidRenderer,
}

impl Renderer {
    pub fn new(gpu: &gpu::Gpu, surface: wgpu::Surface<'static>, window: Arc<Window>, camera: Camera) -> Self {
        todo!()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        todo!()
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    pub fn create_mesh(&self, vertices: &[Vertex], indices: &[u32], topology: Topology, model: Mat4) -> Mesh {
        todo!()
    }

    pub fn create_fluid(&self, particles: &wgpu::Buffer, count: u32, material: FluidMaterial) -> FluidInstance {
        todo!()
    }

    pub fn render(&mut self, meshes: &[&Mesh], fluids: &[&FluidInstance]) {
        todo!()
    }
}

fn create_depth_view(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    todo!()
}

fn create_scene_color_view(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    todo!()
}

fn create_present_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene_color: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    todo!()
}

fn create_present_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    todo!()
}
