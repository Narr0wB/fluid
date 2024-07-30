use nalgebra::{*};
use vulkano::{command_buffer::PrimaryCommandBufferAbstract, sync::GpuFuture};
use std::{sync::Arc, time::Instant};
use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{allocator::{CommandBufferAllocator, StandardCommandBufferAllocator}, AutoCommandBufferBuilder, CommandBufferUsage}, descriptor_set::{allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator}, PersistentDescriptorSet, WriteDescriptorSet}, device::{Device, Queue}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo}, sync};
use rand::Rng;

pub const PARTICLE_RADIUS: f32  = 0.0375; // m
pub const PARTICLE_MASS: f32    = 0.02; // kg
pub const SMOOTHING_RADIUS: f32 = 0.15; // m
pub const GRAVITY: Vector3<f32> = Vector3::new(0.0, -9.81, 0.0);

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/compute.glsl" 
    }
}

mod density {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/compute_density.glsl"
    }
}

#[derive(Debug, Clone, Copy)]
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

        let step = dt * self.velocity;
        let mut new_position = self.position + step;

        if new_position.x <= bounds.x1 {
            self.velocity.x *= -1.0 * (1.0 - bounds.damping_factor);
            new_position.x = bounds.x1 + SMOOTHING_RADIUS;
        }

        if new_position.x >= bounds.x2 {
            self.velocity.x *= -1.0 * (1.0 - bounds.damping_factor);
            new_position.x = bounds.x2 - SMOOTHING_RADIUS;
        }

        if new_position.z <= bounds.z1 {
            self.velocity.z *= -1.0 * (1.0 - bounds.damping_factor);
            new_position.z = bounds.z1 + SMOOTHING_RADIUS;
        }

        if new_position.z >= bounds.z2 {
            self.velocity.z *= -1.0 * (1.0 - bounds.damping_factor);
            new_position.z = bounds.z2 - SMOOTHING_RADIUS;
        }

        if new_position.y <= bounds.y1 {
            self.velocity.y *= -1.0 * (1.0 - bounds.damping_factor);
            new_position.y = bounds.y1 + SMOOTHING_RADIUS;
        }

        self.position = new_position;
    }

    pub fn update_accel(&mut self, a: Vector3<f32>) {
        self.accel = a;
    }
}

pub fn particle_cube(start: Vector3<f32>, side: u32) -> Vec<Particle> {
    let mut p_cube = vec![];

    let particle_separation = SMOOTHING_RADIUS + 0.01;

    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                let randX: f32 = (rand::thread_rng().gen::<f32>() * 0.5 - 1.0) * SMOOTHING_RADIUS / 10.0; 
                let randY: f32 = (rand::thread_rng().gen::<f32>() * 0.5 - 1.0) * SMOOTHING_RADIUS / 10.0; 
                let randZ: f32 = (rand::thread_rng().gen::<f32>() * 0.5 - 1.0) * SMOOTHING_RADIUS / 10.0;

                // println!("{:?} {:?} {:?}", randX, randY, randZ);
                p_cube.push(
                    Particle::new(
                        Vector3::new(
                            start.x + (i as f32 * (particle_separation)) + randX - 1.5, 
                            start.y + (k as f32 * (particle_separation)) + randY + SMOOTHING_RADIUS + 0.1,
                            start.z + (j as f32 * (particle_separation)) + randZ - 1.5
                        ),
                        None
                    )
                );
            }
        }    
    }

    p_cube
}

#[derive(Default, Copy, Clone)]
pub struct BoundingBox {
    pub x1: f32,
    pub x2: f32,
    pub z1: f32,
    pub z2: f32,
    pub y1: f32,
    pub y2: f32,
    
    pub damping_factor: f32 // Between 0 and 1 (ideally 0.5 and 0.7)
}

pub struct Fluid {
    device: Arc<Device>,

    particles: Vec<Particle>,
    target_density: f32,
    pressure_constant: f32,
    viscosity_constant: f32,

    model_buffer: Subbuffer<[f32]>,

    compute_particle_buffer: Option<Subbuffer<[f32]>>,
    compute_density_set: Option<Arc<PersistentDescriptorSet>>,
    compute_update_set: Option<Arc<PersistentDescriptorSet>>
}

impl Fluid {
    pub fn new(particles: Vec<Particle>, target_density: f32, pressure_constant: f32, viscosity_constant: f32, device: Arc<Device>) -> Self {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

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
                (particles.len() * (16)) as u64 
            ).unwrap();
        
        Fluid { device, particles, target_density, pressure_constant, viscosity_constant, model_buffer: matrices_buffer, compute_density_set: None, compute_update_set: None, compute_particle_buffer: None }
    }

    pub fn update(&mut self, dt: f32, bounds: &BoundingBox) -> (&Subbuffer<[f32]>, u32) {
        let len = self.particles.len();

        let mut densities = vec![];
        
        for i in 0..len {
            densities.push(0.0);

            for j in 0..len {
                if i == j {continue;}

                let particle = &self.particles[i];
                let other    = &self.particles[j];
                
                let dist = (particle.position - other.position).norm(); 

                if dist > SMOOTHING_RADIUS {continue;}

                densities[i] += PARTICLE_MASS * smoothing_kernel(dist, SMOOTHING_RADIUS);
            }

            densities[i] += PARTICLE_MASS * smoothing_kernel(0.0, SMOOTHING_RADIUS);
        }

        for i in 0..len {
            let mut pressure_force = Vector3::repeat(0.0);

            for j in 0..len {
                if i == j {continue;}
                let particle = &self.particles[i];
                let other    = &self.particles[j];

                let dir = (other.position - particle.position).normalize(); 
                let dist = (other.position - particle.position).norm();

                if dist > SMOOTHING_RADIUS || dist == 0.0 {continue;}

                let density = densities[i];
                let other_density = densities[j];

                let contribution = -dir * PARTICLE_MASS * (self.convert_density_pressure(density) + self.convert_density_pressure(other_density)) / (2.0 * other_density) * smoothing_kernel_derivative(dist, SMOOTHING_RADIUS);

                pressure_force += contribution;  
            }

            let particle = &mut self.particles[i];
            let accel = pressure_force / densities[i];

            particle.update_accel(GRAVITY + accel);
            particle.update(dt, bounds);

            let particle_velocity_uv = particle.velocity.norm() / 20.0;
            let offset = i * (16);

            let mut translation_uv = Matrix4::new_translation(&particle.position);
            translation_uv[(0, 0)] = particle_velocity_uv;

            self.model_buffer.write().unwrap()[offset..offset+16].copy_from_slice(translation_uv.as_slice());
        }

        (&self.model_buffer, len as u32)
    }

    pub fn bind_compute(&mut self, pipeline: &FluidComputePipeline) {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));

        let stride = 2 * 4;
        let len = self.particles.len();

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
                (len * stride) as u64 
            ).unwrap()
        );
        
        for i in 0..len {
            let particle = &self.particles[i];

            let position = Vector4::new(particle.position.x, particle.position.y, particle.position.z, 0.0);
            let velocity = Vector4::new(particle.velocity.x, particle.velocity.y, particle.velocity.z, 0.0);

            let final_slice = [position.as_slice(), velocity.as_slice()].concat();

            self.compute_particle_buffer.as_mut().unwrap().write().unwrap()[i * stride..(i+1) * stride].copy_from_slice(&final_slice);
        }
        
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(self.device.clone(), Default::default());
        
        let update_layout = pipeline.update_layout().clone();
        let density_layout = pipeline.density_layout().clone();

        let density_descriptor_set_layout = density_layout
            .set_layouts()
            .get(0).unwrap();
        let update_descriptor_set_layout = update_layout
            .set_layouts()
            .get(0).unwrap();

        self.compute_density_set = Some(
                PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                density_descriptor_set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, self.compute_particle_buffer.as_ref().unwrap().clone()),
                ],
                []
            ).unwrap()
        );

        self.compute_update_set = Some(
            PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                update_descriptor_set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, self.compute_particle_buffer.as_ref().unwrap().clone()),
                    WriteDescriptorSet::buffer(1, self.model_buffer.clone())
                ],
                []
            ).unwrap()
        );
    }

    pub fn reset(&mut self, particles: Vec<Particle>) {
        self.particles = particles;
    }
    
    pub fn density(&self) -> f32 {
        self.target_density
    }

    pub fn pressure(&self) -> f32 {
        self.pressure_constant
    }

    pub fn viscosity(&self) -> f32 {
        self.viscosity_constant
    }

    pub fn update_descriptor(&self) -> Arc<PersistentDescriptorSet> {
        self.compute_update_set.as_ref().unwrap().clone()
    }

    pub fn density_descriptor(&self) -> Arc<PersistentDescriptorSet> {
        self.compute_density_set.as_ref().unwrap().clone()
    }

    pub fn models(&self) -> &Subbuffer<[f32]> {
        &self.model_buffer
    }

    pub fn len(&self) -> u32 {
        self.particles.len() as u32
    }

    fn convert_density_pressure(&self, density: f32) -> f32 {
        (density - self.target_density) * self.pressure_constant
    }
}

pub struct FluidComputePipeline {
    device: Arc<Device>,
    density_pipeline: Arc<ComputePipeline>,
    update_pipeline: Arc<ComputePipeline>,
    
    density_layout: Arc<PipelineLayout>,
    update_layout: Arc<PipelineLayout>,
}

impl FluidComputePipeline {
    pub fn new(device: Arc<Device>) -> Self {
        let density_shader = density::load(device.clone()).unwrap();
        let update_shader = cs::load(device.clone()).unwrap();
     
        let density_stage = PipelineShaderStageCreateInfo::new(density_shader.entry_point("main").unwrap());
        let update_stage = PipelineShaderStageCreateInfo::new(update_shader.entry_point("main").unwrap());
        
        let density_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&density_stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();
        let update_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&update_stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();

        let density_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(density_stage, density_layout.clone())
        ).unwrap();
        let update_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(update_stage, update_layout.clone()) 
        ).unwrap();

        Self { device, density_pipeline, update_pipeline, density_layout, update_layout }
    }
     
    pub fn update_layout(&self) -> Arc<PipelineLayout> {
        self.update_layout.clone()
    }

    pub fn density_layout(&self) -> Arc<PipelineLayout> {
        self.density_layout.clone()
    }

    pub fn compute<'a>(&'a self, dt: f32, fluid: &'a Fluid, bounds: &BoundingBox, compute_queue: Arc<Queue>) -> (&Subbuffer<[f32]>, u32) {
        let allocator = StandardCommandBufferAllocator::new(self.device.clone(), Default::default());

        let mut density_builder = AutoCommandBufferBuilder::primary(
            &allocator,
            compute_queue.queue_family_index(), 
            CommandBufferUsage::OneTimeSubmit 
        ).unwrap();

        let mut update_builder = AutoCommandBufferBuilder::primary(
            &allocator, 
            compute_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        ).unwrap();

        let density_push_constants = density::Constants { 
            size: fluid.len(), 
            particle_mass: PARTICLE_MASS, 
            smoothing_radius: SMOOTHING_RADIUS,
        };

        let compute_push_constants = cs::Constants { 
            size: fluid.len(), 
            particle_mass: PARTICLE_MASS, 
            smoothing_radius: SMOOTHING_RADIUS, 
            target_density: fluid.density(), 
            pressure_constant: fluid.pressure(),
            viscosity_constant: fluid.viscosity(),
            dt: dt.into(), 
            bounds: cs::BoundingBox { x1: bounds.x1, x2: bounds.x2, z1: bounds.z1, z2: bounds.z2, y1: bounds.y1, y2: bounds.y2, damping_factor: bounds.damping_factor }
        };

        density_builder
            .bind_pipeline_compute(self.density_pipeline.clone()).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute, 
                self.density_layout.clone(), 
                0, 
                vec![fluid.density_descriptor()]
            ).unwrap()
            .push_constants(self.density_layout.clone(), 0, density_push_constants).unwrap()
            .dispatch([3000, 1, 1]).unwrap();

        let cmd_buffer = density_builder.build().unwrap();

        update_builder
            .bind_pipeline_compute(self.update_pipeline.clone()).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute, 
                self.update_layout.clone(), 
                0, 
                vec![fluid.update_descriptor()]
            ).unwrap()
            .push_constants(self.update_layout.clone(), 0, compute_push_constants).unwrap()
            .dispatch([2048, 1, 1]).unwrap();

        let cmd_buffer_2 = update_builder.build().unwrap();

        let finished = sync::now(self.device.clone())
            .then_execute(compute_queue.clone(), cmd_buffer).unwrap()
            .then_execute(compute_queue.clone(), cmd_buffer_2).unwrap()
            .then_signal_fence_and_flush().unwrap();

        finished.wait(None).unwrap();

        (fluid.models(), fluid.len())
    }
}

pub fn smoothing_kernel(dist: f32, radius: f32) -> f32 {
    if (dist >= radius) {
        return 0.0;
    }

    return 315.0 / (64.0 * std::f32::consts::PI * radius.powi(9)) * (radius * radius - dist * dist).powi(3);
}

pub fn smoothing_kernel_derivative(dist: f32, radius: f32) -> f32 {
    if (dist >= radius) {
        return 0.0;
    } 

    return -45.0 / (std::f32::consts::PI * radius.powi(6)) * (radius - dist) * (radius - dist);
}
