use std::sync::Arc;
use winit::window::Window;

pub trait GpuMirror {
    type Gpu: bytemuck::Pod;
    type Context;

    fn to_gpu(&self, ctx: Self::Context) -> Self::Gpu;

    fn upload(&self, queue: &wgpu::Queue, buffer: &wgpu::Buffer, ctx: Self::Context) {
        queue.write_buffer(buffer, 0, bytemuck::bytes_of(&self.to_gpu(ctx)));
    }
}

pub struct Gpu {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<(Self, wgpu::Surface<'static>)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window)?;
        let gpu = Self::from_instance(&instance, Some(&surface)).await?;
        Ok((gpu, surface))
    }

    #[cfg(test)]
    pub async fn headless() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        Self::from_instance(&instance, None).await
    }

    async fn from_instance(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> anyhow::Result<Self> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                ..Default::default()
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await?;

        Ok(Gpu { adapter, device, queue })
    }
}
