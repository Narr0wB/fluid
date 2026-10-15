mod camera;
mod controller;
mod gpu;
mod renderer;
mod fluid;

use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

use crate::controller::CameraController;
use crate::gpu::gpu;
use crate::renderer::Renderer;
use crate::renderer::mesh::Mesh;

struct State {
    gpu: gpu,
    renderer: Renderer,
    bounding_box: Mesh,
}

struct App {
    state: Option<State>,
    controller: CameraController,
    last_frame: Instant,
}

impl App {
    fn new() -> Self {
        todo!()
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        todo!()
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        todo!()
    }

    fn device_event(&mut self, event_loop: &ActiveEventLoop, device_id: DeviceId, event: DeviceEvent) {
        todo!()
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        todo!()
    }
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new())?;
    Ok(())
}
