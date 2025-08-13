
#![allow(non_snake_case)]

use nalgebra::{*};
use vulkano::sync::GpuFuture;
use std::sync::Arc;
use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{allocator::{CommandBufferAllocator, StandardCommandBufferAllocator}, AutoCommandBufferBuilder, CommandBufferUsage}, descriptor_set::{allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator}, PersistentDescriptorSet, WriteDescriptorSet}, device::{Device, Queue}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo}, sync};
use rand::Rng;


pub const PARTICLE_MASS: f32    = 0.02; // kg
pub const SMOOTHING_RADIUS: f32 = 0.08; // m
pub const GRAVITY: Vector3<f32> = Vector3::new(0.0, -9.81, 0.0);

mod pressure {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/compute_pressure.glsl" 
    }
}

mod density {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/compute_density.glsl"
    }
}

mod positions {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/compute_positions.glsl"
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Particle {
    pub position: Vector3<f32>,
    pub density:  f32,
    pub velocity: Vector3<f32>,
    pub pressure: f32
}

impl Particle {
    pub fn new(start_pos: Vector3<f32>, velocity: Option<Vector3<f32>>) -> Self{
        Particle {position: start_pos, density: 0.0, velocity: velocity.unwrap_or(Vector3::repeat(0.0)), pressure: 0.0}
    }

    pub fn update(&mut self, dt: f32, acceleration: Vector3<f32>, bounds: &BoundingBox) {
        self.velocity += dt * acceleration;

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
}

pub fn particle_cube(dist: f32, start: Vector3<f32>, start_velocity: Option<Vector3<f32>>, side: u32) -> Vec<Particle> {
    let mut p_cube = vec![];
    let particle_separation = dist;

    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                let randX: f32 = (rand::thread_rng().gen::<f32>() * 0.5 - 1.0) * particle_separation; 
                let randY: f32 = (rand::thread_rng().gen::<f32>() * 0.5 - 1.0) * particle_separation; 
                let randZ: f32 = (rand::thread_rng().gen::<f32>() * 0.5 - 1.0) * particle_separation;

                // println!("{:?} {:?} {:?}", randX, randY, randZ);
                p_cube.push(
                    Particle::new(
                        Vector3::new(
                            start.x + (i as f32 * (particle_separation)) + randX - 1.5, 
                            start.y + (k as f32 * (particle_separation)) + randY - 1.5,
                            start.z + (j as f32 * (particle_separation)) + randZ - 1.5,
                            // start.x + (i as f32 * (particle_separation)), 
                            // start.y + (k as f32 * (particle_separation)),
                            // start.z + (j as f32 * (particle_separation))
                        ),
                        start_velocity 
                    )
                );
            }
        }    
    }

    p_cube
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
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
    compute_force_buffer: Option<Subbuffer<[f32]>>,
    compute_density_descriptor_set: Option<Arc<PersistentDescriptorSet>>,
    compute_pressure_descriptor_set: Option<Arc<PersistentDescriptorSet>>,
    compute_positions_descriptor_set: Option<Arc<PersistentDescriptorSet>>
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
        
        return Fluid {
            device, 
            particles, 
            target_density, 
            pressure_constant, 
            viscosity_constant, 
            model_buffer: matrices_buffer, 
            compute_particle_buffer: None, 
            compute_force_buffer: None, 
            compute_density_descriptor_set: None, 
            compute_pressure_descriptor_set: None,
            compute_positions_descriptor_set: None
        };
    }

    pub fn update(&mut self, dt: f32, bounds: &BoundingBox) -> (&Subbuffer<[f32]>, u32) {
        let len = self.particles.len();
        
        // Density compute pass
        for i in 0..len {
            self.particles[i].density = 0.0;

            for j in 0..len {
                if i == j {continue;}

                let particle = &self.particles[i];
                let other    = &self.particles[j];
                let dist = (particle.position - other.position).norm(); 
                if dist > SMOOTHING_RADIUS {continue;}
                self.particles[i].density += PARTICLE_MASS * smoothing_kernel(dist, SMOOTHING_RADIUS);
            }

            self.particles[i].density += PARTICLE_MASS * smoothing_kernel(0.0, SMOOTHING_RADIUS);
        }

        // Pressure compute pass
        let mut pressure_forces = vec![Vector3::<f32>::repeat(0.0); self.particles.len()];
        for i in 0..len {
            let mut pressure_force = pressure_forces[i];
            pressure_force = Vector3::repeat(0.0);

            for j in 0..len {
                if i == j {continue;}
                let particle = &self.particles[i];
                let other    = &self.particles[j];
                let dir = (other.position - particle.position).normalize(); 
                let dist = (other.position - particle.position).norm();
                if dist > SMOOTHING_RADIUS || dist == 0.0 {continue;}

                let contribution = -dir * PARTICLE_MASS * 
                    (self.convert_density_pressure(particle.density) + self.convert_density_pressure(other.density)) / (2.0 * other.density) * 
                    smoothing_kernel_derivative(dist, SMOOTHING_RADIUS);

                pressure_force += contribution;  
            }
        }

        for i in 0..len {
            let pressure_force = pressure_forces[i];
            let particle = &mut self.particles[i];
            let accel = pressure_force / particle.density;

            particle.update(dt, GRAVITY + accel, bounds);

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

        let len = self.particles.len();

        // 2 vec3 (2 * 3) and 2 floats per particle
        let particle_stride = 2 * 3 + 2;
        // 1 vec4
        let force_stride = 4;

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
                (len * particle_stride) as u64 
            ).unwrap()
        );

        self.compute_force_buffer = Some(
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
                (len * force_stride) as u64
            ).unwrap()
        );
        
        for i in 0..len {
            let particle = &self.particles[i];
            let mut particle_buf = self.compute_particle_buffer.as_mut().unwrap().write().unwrap();
            let mut force_buf = self.compute_force_buffer.as_mut().unwrap().write().unwrap();
            let pbase = i * particle_stride;
            let fbase = i * force_stride;

            particle_buf[pbase + 0..pbase + 3].copy_from_slice(particle.position.as_slice());
            particle_buf[pbase + 3] = particle.density;
            particle_buf[pbase + 4..pbase + 7].copy_from_slice(particle.velocity.as_slice());
            particle_buf[pbase + 7] = particle.pressure;

            force_buf[fbase + 0..fbase + 4].copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        }
        
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(self.device.clone(), Default::default());
        
        let pressure_layout = pipeline.pressure_layout();
        let density_layout = pipeline.density_layout();
        let positions_layout = pipeline.positions_layout();

        let density_descriptor_set_layout = density_layout
            .set_layouts()
            .get(0).unwrap();
        let pressure_descriptor_set_layout = pressure_layout
            .set_layouts()
            .get(0).unwrap();
        let positions_descriptor_set_layout = positions_layout
            .set_layouts()
            .get(0).unwrap();

        self.compute_density_descriptor_set = Some(
                PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                density_descriptor_set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, self.compute_particle_buffer.as_ref().unwrap().clone()),
                ],
                []
            ).unwrap()
        );

        self.compute_pressure_descriptor_set = Some(
            PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                pressure_descriptor_set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, self.compute_particle_buffer.as_ref().unwrap().clone()),
                    WriteDescriptorSet::buffer(1, self.compute_force_buffer.as_ref().unwrap().clone())
                ],
                []
            ).unwrap()
        );

        self.compute_positions_descriptor_set = Some(
            PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                positions_descriptor_set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, self.compute_particle_buffer.as_ref().unwrap().clone()),
                    WriteDescriptorSet::buffer(1, self.compute_force_buffer.as_ref().unwrap().clone()),
                    WriteDescriptorSet::buffer(2, self.model_buffer.clone())
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

    pub fn density_descriptor(&self) -> Option<Arc<PersistentDescriptorSet>> {
        self.compute_density_descriptor_set.clone()
    }

    pub fn pressure_descriptor(&self) -> Option<Arc<PersistentDescriptorSet>> {
        self.compute_pressure_descriptor_set.clone()
    }

    pub fn positions_descriptor(&self) -> Option<Arc<PersistentDescriptorSet>> {
        self.compute_positions_descriptor_set.clone()
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
    pressure_pipeline: Arc<ComputePipeline>,
    positions_pipeline: Arc<ComputePipeline>,
    
    density_layout: Arc<PipelineLayout>,
    pressure_layout: Arc<PipelineLayout>,
    positions_layout: Arc<PipelineLayout>,
}

impl FluidComputePipeline {
    pub fn new(device: Arc<Device>) -> Self {
        let density_shader = density::load(device.clone()).unwrap();
        let pressure_shader = pressure::load(device.clone()).unwrap();
        let positions_shader = positions::load(device.clone()).unwrap();
     
        let density_stage = PipelineShaderStageCreateInfo::new(density_shader.entry_point("main").unwrap());
        let pressure_stage = PipelineShaderStageCreateInfo::new(pressure_shader.entry_point("main").unwrap());
        let positions_stage = PipelineShaderStageCreateInfo::new(positions_shader.entry_point("main").unwrap());
        
        let density_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&density_stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();
        let pressure_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&pressure_stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();
        let positions_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&positions_stage])
                .into_pipeline_layout_create_info(device.clone()).unwrap()
        ).unwrap();


        let density_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(density_stage, density_layout.clone())
        ).unwrap();
        let pressure_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(pressure_stage, pressure_layout.clone()) 
        ).unwrap();
        let positions_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(positions_stage, positions_layout.clone()) 
        ).unwrap();

        return Self { 
            device, 
            density_pipeline, pressure_pipeline, positions_pipeline,
            density_layout, pressure_layout, positions_layout
        };
    }
     
    pub fn pressure_layout(&self) -> Arc<PipelineLayout> {
        self.pressure_layout.clone()
    }

    pub fn density_layout(&self) -> Arc<PipelineLayout> {
        self.density_layout.clone()
    }

    pub fn positions_layout(&self) -> Arc<PipelineLayout> {
        self.positions_layout.clone()
    }

    pub fn compute_step<'a>(&'a self, dt: f32, fluid: &'a Fluid, bounds: &BoundingBox, compute_queue: Arc<Queue>) -> (&Subbuffer<[f32]>, u32) {
        let allocator = StandardCommandBufferAllocator::new(self.device.clone(), Default::default());

        let mut density_builder = AutoCommandBufferBuilder::primary(
            &allocator,
            compute_queue.queue_family_index(), 
            CommandBufferUsage::OneTimeSubmit 
        ).unwrap();

        let mut pressure_builder = AutoCommandBufferBuilder::primary(
            &allocator, 
            compute_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        ).unwrap();

        let mut positions_builder = AutoCommandBufferBuilder::primary(
            &allocator, 
            compute_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        ).unwrap();

        let density_push_constants = density::Constants { 
            size: fluid.len(), 
            particle_mass: PARTICLE_MASS, 
            smoothing_radius: SMOOTHING_RADIUS,
        };

        let pressure_push_constants = pressure::Constants { 
            size: fluid.len(), 
            particle_mass: PARTICLE_MASS, 
            smoothing_radius: SMOOTHING_RADIUS, 
            target_density: fluid.density(), 
            pressure_constant: fluid.pressure(),
            viscosity_constant: fluid.viscosity(),
        };

        let positions_push_constants = positions::Constants {
            size: fluid.len(), 
            particle_mass: PARTICLE_MASS, 
            smoothing_radius: SMOOTHING_RADIUS, 
            dt: dt.into(), 
            bounds: positions::BoundingBox { x1: bounds.x1, x2: bounds.x2, z1: bounds.z1, z2: bounds.z2, y1: bounds.y1, y2: bounds.y2, damping_factor: bounds.damping_factor }
        };

        let density_dispatch_layout = [3000, 1, 1];
        let pressure_dispatch_layout = [3000, 1, 1];
        let positions_dispatch_layout = [3000, 1, 1];

        density_builder
            .bind_pipeline_compute(self.density_pipeline.clone()).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute, 
                self.density_layout.clone(), 
                0, 
                vec![fluid.density_descriptor().unwrap()]
            ).unwrap()
            .push_constants(self.density_layout.clone(), 0, density_push_constants).unwrap()
            .dispatch(density_dispatch_layout).unwrap();
        let cmd_buffer = density_builder.build().unwrap();

        pressure_builder
            .bind_pipeline_compute(self.pressure_pipeline.clone()).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute, 
                self.pressure_layout.clone(), 
                0, 
                vec![fluid.pressure_descriptor().unwrap()]
            ).unwrap()
            .push_constants(self.pressure_layout.clone(), 0, pressure_push_constants).unwrap()
            .dispatch(pressure_dispatch_layout).unwrap();
        let cmd_buffer_2 = pressure_builder.build().unwrap();

        positions_builder
            .bind_pipeline_compute(self.positions_pipeline.clone()).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute, 
                self.positions_layout.clone(), 
                0, 
                vec![fluid.positions_descriptor().unwrap()]
            ).unwrap()
            .push_constants(self.positions_layout.clone(), 0, positions_push_constants).unwrap()
            .dispatch(positions_dispatch_layout).unwrap();
        let cmd_buffer_3 = positions_builder.build().unwrap();

        let finished = sync::now(self.device.clone())
            .then_execute(compute_queue.clone(), cmd_buffer).unwrap()
            .then_execute(compute_queue.clone(), cmd_buffer_2).unwrap()
            .then_execute(compute_queue.clone(), cmd_buffer_3).unwrap()
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
