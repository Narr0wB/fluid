use nalgebra::{*};
use std::sync::Arc;
use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}};

pub const PARTICLE_RADIUS: f32  = 0.1;
pub const GRAVITY: Vector3<f32> = Vector3::new(0.0, -9.81, 0.0);

pub struct Particle {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub accel:    Vector3<f32>,
}

impl Particle {
    pub fn new(start_pos: Vector3<f32>, velocity: Option<Vector3<f32>>) -> Self{
        Particle { position: start_pos, velocity: velocity.unwrap_or(Vector3::repeat(0.0)), accel: Vector3::new(0.0, 0.0, 0.0) }
    }

    pub fn update(&mut self, dt: f32, bounds: &BoundingBox) {
        self.velocity += dt * self.accel;

        let mut step = dt * self.velocity;
        let mut new_position = self.position + step;

        if new_position.x <= bounds.x1 || new_position.x >= bounds.x2 || 
            new_position.z <= bounds.z1 || new_position.z >= bounds.z2 ||
            new_position.y <= bounds.y1 
        { 
            self.velocity *= -1.0 * (1.0 - bounds.damping_factor);

            step = dt * self.velocity;
            if step.x.abs() <= 0.05 { step.x = 0.0; }
            if step.y.abs() <= 0.05 { step.y = 0.0; }
            if step.z.abs() <= 0.05 { step.z = 0.0; }

            new_position = self.position + step;
        }

        self.position = new_position;
    }

    pub fn update_accel(&mut self, a: Vector3<f32>) {
        self.accel = a;
    }
}

pub struct BoundingBox {
    pub x1: f32,
    pub x2: f32,
    pub z1: f32,
    pub z2: f32,
    pub y1: f32,
    
    pub damping_factor: f32 // Between 0 and 1 (ideally 0.1 and 0.4)
}

pub struct Fluid {
    particles: Vec<Particle>,
    pressure_constant: f32,

    model_buffer: Subbuffer<[f32]>
}

impl Fluid {
    pub fn new(particles: Vec<Particle>, pressure: f32, memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        // Create the buffer for how many particles this fluid has
        
        let matrices_buffer = 
            Buffer::new_slice::<f32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                (particles.len() * (16 + 4)) as u64 
            ).unwrap();
        
        Fluid { particles, pressure_constant: pressure, model_buffer: matrices_buffer}
    }

    pub fn update(&mut self, dt: f32, bounds: &BoundingBox) -> &Subbuffer<[f32]> {
        // Calculate forces and accelerations
        
        // Propagate to each single particles
        for (i, particle) in self.particles.iter_mut().enumerate() {
            particle.update_accel(GRAVITY);
            particle.update(dt, bounds);
            
            let particle_velocity_uv = particle.velocity.norm() / 10.0;
            let offset = i * (16 + 4);

            self.model_buffer.write().unwrap()[offset..offset+16].copy_from_slice(Matrix4::new_translation(&particle.position).as_slice());
            self.model_buffer.write().unwrap()[offset+16..offset+20].copy_from_slice(Vector4::repeat(particle_velocity_uv).as_slice());
        }

        &self.model_buffer
    }
}
