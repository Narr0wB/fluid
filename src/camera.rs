use std::f32::consts::{FRAC_PI_2, PI, TAU};

use glam::camera::rh::{proj::directx, view};
use glam::{EulerRot, Mat4, Quat, Vec3};

const Z_NEAR: f32 = 0.1;
const Z_FAR: f32 = 100.0;
const MIN_FOV: f32 = 10.0 * PI / 180.0;
const MAX_FOV: f32 = 90.0 * PI / 180.0;
const MAX_PITCH: f32 = FRAC_PI_2 - 0.01;

/* 
 Right-handed, +Y up, looks down -Z at zero yaw/pitch/roll. Positive yaw turns left, positive 
 pitch looks up, positive roll lifts the right side. Clip space is +Y up with depth in 0..1.
*/ 
pub struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    roll: f32,

    fov_y: f32,
    aspect: f32,
    z_near: f32,
    z_far: f32,
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3, fov_y: f32, aspect: f32) -> Self {
        let mut camera = Camera {
            position,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            fov_y: fov_y.clamp(MIN_FOV, MAX_FOV),
            aspect,
            z_near: Z_NEAR,
            z_far: Z_FAR,
        };
        camera.look_at(target);
        return camera;
    }

    pub fn translate(&mut self, delta: Vec3) {
        self.position += delta;
    }

    pub fn look_at(&mut self, target: Vec3) {
        let Some(dir) = (target - self.position).try_normalize() else {
            return;
        };

        self.set_orientation((-dir.x).atan2(-dir.z), dir.y.asin(), self.roll);
    }

    pub fn set_orientation(&mut self, yaw: f32, pitch: f32, roll: f32) {
        self.yaw = yaw.rem_euclid(TAU);
        self.pitch = pitch.clamp(-MAX_PITCH, MAX_PITCH);
        self.roll = roll.rem_euclid(TAU);
    }

    pub fn rotate(&mut self, d_yaw: f32, d_pitch: f32, d_roll: f32) {
        self.set_orientation(self.yaw + d_yaw, self.pitch + d_pitch, self.roll + d_roll);
    }

    pub fn set_fov(&mut self, fov_y: f32) {
        self.fov_y = fov_y.clamp(MIN_FOV, MAX_FOV);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.aspect = width as f32 / height as f32;
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn fov(&self) -> f32 {
        self.fov_y
    }

    pub fn orientation(&self) -> Quat {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, self.roll)
    }

    pub fn forward(&self) -> Vec3 {
        self.orientation() * Vec3::NEG_Z
    }

    pub fn right(&self) -> Vec3 {
        self.orientation() * Vec3::X
    }

    pub fn up(&self) -> Vec3 {
        self.orientation() * Vec3::Y
    }

    pub fn view(&self) -> Mat4 {
        view::look_to_mat4(self.position, self.forward(), self.up())
    }

    pub fn projection(&self) -> Mat4 {
        directx::perspective(self.fov_y, self.aspect, self.z_near, self.z_far)
    }

    pub fn view_projection(&self) -> Mat4 {
        self.projection() * self.view()
    }
}
