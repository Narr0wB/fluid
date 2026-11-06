use glam::{Vec2, Vec3};
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::camera::Camera;

const PIXELS_PER_SCROLL_LINE: f32 = 40.0;

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
        CameraController {
            move_speed,
            look_sensitivity,
            zoom_step,
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            looking: false,
            look_delta: Vec2::ZERO,
            scroll: 0.0,
        }
    }

    pub fn is_looking(&self) -> bool {
        self.looking
    }

    pub fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                match code {
                    KeyCode::KeyW => self.forward = pressed,
                    KeyCode::KeyS => self.backward = pressed,
                    KeyCode::KeyA => self.left = pressed,
                    KeyCode::KeyD => self.right = pressed,
                    KeyCode::Space => self.up = pressed,
                    KeyCode::ShiftLeft => self.down = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.looking = *state == ElementState::Pressed;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, lines) => *lines,
                    MouseScrollDelta::PixelDelta(pixels) => pixels.y as f32 / PIXELS_PER_SCROLL_LINE,
                };
            }
            WindowEvent::Focused(false) => {
                self.forward = false;
                self.backward = false;
                self.left = false;
                self.right = false;
                self.up = false;
                self.down = false;
                self.looking = false;
            }
            _ => {}
        }
    }

    pub fn device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.looking
        {
            self.look_delta += Vec2::new(delta.0 as f32, delta.1 as f32);
        }
    }

    pub fn update(&mut self, camera: &mut Camera, dt: f32) {
        let axis = |positive: bool, negative: bool| positive as i32 as f32 - negative as i32 as f32;

        let forward = camera.forward().with_y(0.0).normalize_or_zero();
        let right = camera.right().with_y(0.0).normalize_or_zero();
        let direction = forward * axis(self.forward, self.backward)
            + right * axis(self.right, self.left)
            + Vec3::Y * axis(self.up, self.down);
        camera.translate(direction.normalize_or_zero() * self.move_speed * dt);

        camera.rotate(self.look_delta.x * self.look_sensitivity, self.look_delta.y * self.look_sensitivity, 0.0);
        self.look_delta = Vec2::ZERO;

        camera.set_fov(camera.fov() - self.scroll * self.zoom_step);
        self.scroll = 0.0;
    }
}
