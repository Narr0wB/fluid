use glam::Vec2;
use winit::event::{DeviceEvent, WindowEvent};

use crate::camera::Camera;

pub struct CameraController {
    move_speed: f32,
    look_sensitivity: f32,
    zoom_step: f32,

    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,

    looking: bool,
    look_delta: Vec2,
    scroll: f32,
}

impl CameraController {
    pub fn new(move_speed: f32, look_sensitivity: f32, zoom_step: f32) -> Self {
        todo!()
    }

    pub fn window_event(&mut self, event: &WindowEvent) -> bool {
        todo!()
    }

    pub fn device_event(&mut self, event: &DeviceEvent) {
        todo!()
    }

    pub fn update(&mut self, camera: &mut Camera, dt: f32) {
        todo!()
    }
}
