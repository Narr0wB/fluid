use nalgebra::{*};
use std::{sync::Arc, time::Instant};
use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, descriptor_set::{allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator}, PersistentDescriptorSet, WriteDescriptorSet}, device::Device, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo}};

pub const PARTICLE_RADIUS: f32  = 0.1; // m
pub const PARTICLE_MASS: f32    = 0.5; // kg
pub const SMOOTHING_RADIUS: f32 = 1.0; // m
pub const GRAVITY: Vector3<f32> = Vector3::new(0.0, -9.81, 0.0);

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
            
        "
    }
}

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

pub fn particle_cube(start: Vector3<f32>, side: u32) -> Vec<Particle> {
    let mut p_cube = vec![];

    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                p_cube.push(Particle::new(Vector3::new(start.x + PARTICLE_RADIUS + i as f32 * (PARTICLE_RADIUS * 2.0), start.y + PARTICLE_RADIUS + k as f32 * (PARTICLE_RADIUS * 2.0), start.z + PARTICLE_RADIUS + j as f32 * (PARTICLE_RADIUS * 2.0)), None));
            }
        }    
    }

    p_cube
}

pub struct BoundingBox {
    pub x1: f32,
    pub x2: f32,
    pub z1: f32,
    pub z2: f32,
    pub y1: f32,
    pub y2: f32,
    
    pub damping_factor: f32 // Between 0 and 1 (ideally 0.1 and 0.4)
}

impl BoundingBox {
    pub fn edge_collision(&self, position: Vector3<f32>, threshold: f32) -> bool {
        return ((self.x1 - threshold <= position.x && position.x <= self.x1) ||
                (self.x2 <= position.x && position.x <= self.x2 + threshold) ||

                (self.z1 - threshold <= position.z && position.z <= self.z1) ||
                (self.z2 <= position.z && position.z <= self.z2 + threshold) ||

                (self.y1 - threshold <= position.y && position.y <= self.y1)) 
                && (position.y <= self.y2);
    }
}



pub struct Fluid {
    device: Arc<Device>,

    particles: Vec<Particle>,
    target_density: f32,
    pressure_constant: f32,

    model_buffer: Subbuffer<[f32]>,
    compute_particle_buffer: Option<Subbuffer<[f32]>>,
    compute_descriptor_set: Option<Arc<PersistentDescriptorSet>>
}

impl Fluid {
    pub fn new(particles: Vec<Particle>, target_density: f32, pressure: f32, device: Arc<Device>) -> Self {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(device.clone(), Default::default()));

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
        
        Fluid { device, particles, target_density, pressure_constant: pressure, model_buffer: matrices_buffer, compute_descriptor_set: None, compute_particle_buffer: None }
    }

    pub fn update(&mut self, dt: f32, bounds: &BoundingBox) -> (&Subbuffer<[f32]>, u32) {
        let len = self.particles.len();

        let mut densities = vec![];
        

        let before = Instant::now(); 
        for i in 0..len {
            densities.push(0.0);

            for j in 0..len {
                let particle = &self.particles[i];
                let other    = &self.particles[j];
                
                let dist = (particle.position - other.position).norm(); 

                if dist > SMOOTHING_RADIUS {continue;}

                densities[i] += PARTICLE_MASS * smoothing_kernel(dist, SMOOTHING_RADIUS);
            }
        }

        println!("{:?} {}", before.elapsed(), len); 
        for i in 0..len {
            let mut pressure_force = Vector3::repeat(0.0);

            for j in 0..len {
                if i == j {continue;}
                let particle = &self.particles[i];

                let mut dir = particle.position - self.particles[j].position;
                let dist = dir.norm();

                if dist > SMOOTHING_RADIUS {continue;}
                dir = dir / dist;

                let density = densities[j];

                pressure_force += -(self.convert_density_pressure(density) * smoothing_kernel_derivative(dist, SMOOTHING_RADIUS) * PARTICLE_MASS / density) * dir;  
            }

            let particle = &mut self.particles[i];

            particle.update_accel(GRAVITY + pressure_force);
            particle.update(dt, bounds);

            let particle_velocity_uv = particle.velocity.norm() / 20.0;
            let offset = i * (16 + 4);

            self.model_buffer.write().unwrap()[offset..offset+16].copy_from_slice(Matrix4::new_translation(&particle.position).as_slice());
            self.model_buffer.write().unwrap()[offset+16..offset+20].copy_from_slice(Vector4::repeat(particle_velocity_uv).as_slice());
        }

        (&self.model_buffer, len as u32)
    }

    pub fn bind_compute(&mut self, pipeline: &FluidComputePipeline) {
        // Create the compute particle buffer
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(self.device.clone(), Default::default()));

        let stride = 2 * 4;

        self.compute_particle_buffer = Some(
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
                (self.particles.len() * stride) as u64 
            ).unwrap()
        );
        
        for offset in 0..self.particles.len() {
            let particle = &self.particles[offset];

            let position = Vector4::new(particle.position.x, particle.position.y, particle.position.z, 0.0);
            let velocity = Vector4::new(particle.velocity.x, particle.velocity.y, particle.velocity.z, 0.0);

            let final_slice = [position.as_slice(), velocity.as_slice()].concat();

            self.compute_particle_buffer.as_mut().unwrap().write().unwrap()[offset * stride..(offset+1) * stride].copy_from_slice(&final_slice);
        }
        
        let layout = pipeline.layout().clone();

        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(self.device.clone(), Default::default());
        let descriptor_set_layout = layout
            .set_layouts()
            .get(0).unwrap();

        self.compute_descriptor_set = Some(
                PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                descriptor_set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, self.compute_particle_buffer.as_ref().unwrap().clone())
                ],
                []
            ).unwrap()
        );
    }

    fn convert_density_pressure(&self, density: f32) -> f32 {
        (density - self.target_density) * self.pressure_constant
    }
}

pub struct FluidComputePipeline {
    device: Arc<Device>,
    pipeline: Arc<ComputePipeline>,
    layout: Arc<PipelineLayout>
}

impl FluidComputePipeline {
    pub fn new(device: Arc<Device>) -> Self {
        let shader = cs::load(device.clone()).unwrap();
        
        let stage = PipelineShaderStageCreateInfo::new(shader.entry_point("main").unwrap());
        
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();

        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout.clone()) 
        ).unwrap();

        Self { device, pipeline, layout }
    }
     
    pub fn layout(&self) -> Arc<PipelineLayout> {
        self.layout.clone()
    }
}

pub fn smoothing_kernel(dist: f32, radius: f32) -> f32 {
    if dist < radius { 
        2.0 * (-dist.abs()).exp() / smoothing_sphere_volume(radius) 
    }
    else {
        0.0
    }
}

pub fn smoothing_kernel_derivative(dist: f32, radius: f32) -> f32 {
    if dist < radius { 
        ( smoothing_kernel(dist + 0.001, radius) - smoothing_kernel(dist, radius) ) / 0.001 
    }
    else {
        0.0
    }
}

pub fn smoothing_sphere_volume(radius: f32) -> f32 {
    2.0
}
