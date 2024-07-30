extern crate nalgebra as na;

use na::{Matrix4, Perspective3, RowVector4, Unit, Vector3};

const UP: Vector3<f32> = Vector3::new(0.0, 1.0, 0.0);

pub struct Camera {
    position: Vector3<f32>,
    orientation: Vector3<f32>,

    aspect_ratio: f32,
    zfar: f32,
    znear: f32,
    fov: f32,
}

impl Camera {
    pub fn new(position: Vector3<f32>, orientation: Vector3<f32>, aspect_ratio: Option<f32>, fov: f32) -> Self {
        Camera { position, orientation, aspect_ratio: aspect_ratio.unwrap_or(16.0 / 9.0), zfar: 100.0, znear: 1.0, fov }
    }

    pub fn set_target(&mut self, camera_target: Vector3<f32>) {
        self.orientation = (camera_target - self.position).normalize();
    }

    pub fn set_position(&mut self, camera_position: Vector3<f32>) {
        self.position = camera_position;
    }

    pub fn set_orientation(&mut self, camera_orientation: Vector3<f32>) {
        if self.orientation.cross(&camera_orientation).norm() < 0.0001 { return; }

        self.orientation = camera_orientation;
    }

    pub fn set_fov(&mut self, fov: f32) {
        if fov >= 90.0 || fov <= 10.0 { return; }

        self.fov = fov;
    }

    pub fn set_aspect_ratio(&mut self, aspect: f32) {
        self.aspect_ratio = aspect;
    }

    pub fn get_position(&self) -> Vector3<f32> {
        self.position
    }

    pub fn get_orientation(&self) -> Vector3<f32> {
        self.orientation
    }

    pub fn get_fov(&self) -> f32 {
        self.fov
    }

    pub fn view(&self) -> Matrix4<f32> {
        let right_vector: Vector3<f32> = UP.cross(&self.orientation).normalize();
        let up_vector: Vector3<f32> = right_vector.cross(&self.orientation).normalize();

        let rotation = Matrix4::from_rows(&[
            RowVector4::new(right_vector.x,     right_vector.y,     right_vector.z,     0.0),
            RowVector4::new(up_vector.x,        up_vector.y,        up_vector.z,        0.0),
            RowVector4::new(self.orientation.x, self.orientation.y, self.orientation.z, 0.0),
            RowVector4::new(0.0, 0.0, 0.0, 1.0)
        ]);

        let translation = Matrix4::from_rows(&[
            RowVector4::new(1.0, 0.0, 0.0, -self.position.x),
            RowVector4::new(0.0, 1.0, 0.0, -self.position.y),
            RowVector4::new(0.0, 0.0, 1.0, -self.position.z),
            RowVector4::new(0.0, 0.0, 0.0, 1.0)
        ]);

        return rotation * translation;
    }

    pub fn projection(&self) -> Matrix4<f32> {
        let fov_factor: f32 = 1.0/self.fov.to_radians().tan();
        let lambda: f32 = self.zfar / (self.zfar - self.znear);
        let lambda2: f32 = -(self.zfar * self.znear) / (self.zfar - self.znear);

        let p = Matrix4::from_rows(&[
            RowVector4::new(self.aspect_ratio * fov_factor, 0.0, 0.0, 0.0),
            RowVector4::new(0.0, fov_factor, 0.0, 0.0),
            RowVector4::new(0.0, 0.0, lambda, lambda2),
            RowVector4::new(0.0, 0.0, 1.0, 0.0)
        ]);

        return p;
    }

    pub fn vp(&self) -> Matrix4<f32> {
        return self.projection() * self.view();
    }
}
