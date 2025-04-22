
use vulkano::device::{Device, Features, DeviceExtensions, physical::PhysicalDeviceType, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use vulkano::VulkanLibrary;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyboardInput, MouseButton, MouseScrollDelta};

use nalgebra::{*};

mod mesh;
mod camera;
mod renderer;
mod fluid;

use crate::camera::Camera;
use crate::renderer::Renderer;
use crate::mesh::{*};
use crate::fluid::{FluidComputePipeline, particle_cube, BoundingBox, Fluid, SMOOTHING_RADIUS};

use winit::{event::{Event, WindowEvent}, event_loop::{ControlFlow, EventLoop}};

use std::time::Instant;

fn main() {
    let event_loop = EventLoop::new();
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = Surface::required_extensions(&event_loop);
    
    // Initialize the vulkan instance with the current event loop
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        }
    )
    .unwrap();

    // Create new device
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ext_shader_atomic_float: true,
        ..DeviceExtensions::empty()
    };

    // Select physical device
    let (phys_device, queue_family_graphics) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)|  {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        // && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("Could not find a suitable device");

    if cfg!(DEBUG) {
        println!(
            "Using device: {} (type: {:?})",
            phys_device.properties().device_name,
            phys_device.properties().device_type
        );
    }

    let (device, mut queues) = Device::new(
        phys_device,
        DeviceCreateInfo {
            enabled_features: Features {
                fill_mode_non_solid: true,
                shader_buffer_float32_atomic_add: true,
                ..Default::default()
            },
            enabled_extensions: device_extensions,
            queue_create_infos: vec![
                QueueCreateInfo {
                    queue_family_index: queue_family_graphics,
                    ..Default::default()
                }
            ],
            ..Default::default()
        }
    )
    .unwrap();

    let graphics_queue = queues.next().unwrap();

    let mut camera = Camera::new(
        Vector3::new(8.0, 3.0, -5.0),                      // position
        Vector3::new(0.0, 0.0, 1.0),                        // orientation
        None,                                               // aspect_ratio
        30.0                                                // FOV
    );

    camera.set_target(Vector3::new(2.5, 2.5, 2.5));

    let mut renderer = Renderer::init(instance, &event_loop, camera, device.clone(), graphics_queue.clone()); 
    
    let vertices: Vec<MVertex> = vec![
        MVertex {position: [0.0, 0.0, 0.0]}, MVertex {position: [0.0, 1.0, 0.0]},
        MVertex {position: [0.0, 0.0, 1.0]}, MVertex {position: [0.0, 1.0, 1.0]},

        MVertex {position: [1.0, 0.0, 0.0]}, MVertex {position: [1.0, 1.0, 0.0]},
        MVertex {position: [1.0, 0.0, 1.0]}, MVertex {position: [1.0, 1.0, 1.0]},    
    ];
    
    let indices: Vec<u32> = vec![
        0, 1, 3, 3, 2, 0,

        2, 6, 3, 3, 7, 6,

        4, 5, 7, 7, 6, 4,
        
        0, 4, 1, 1, 5, 4
    ]; 
    
    let model_matrix = Matrix4::new_scaling(3.0);
    let bounding_box = Mesh::new(vertices, indices, model_matrix, device.clone()); 

    let pressure_constant = 3.0;
    let density = 998.29;
    let viscosity = 3.5;
    let particle_radius = 2.0 * ((3.0 * fluid::PARTICLE_MASS) / (4.0*std::f32::consts::PI*998.29)).cbrt(); // m

    let sphere = create_sphere(particle_radius, Vector3::new(0.0, 0.0, 0.0), 10, device.clone());
    let initial_particle_distance = f32::powf((4.0*particle_radius.powi(3)*f32::pi())/(3.0*50.0), 1.0/3.0) + 0.07;
    println!("{} pr: {} sr: {}", initial_particle_distance, particle_radius, SMOOTHING_RADIUS);
    let particles = particle_cube(initial_particle_distance, Vector3::new(2.0, 5.0, 2.0), Some(Vector3::new(0.5, 0.0, 0.5)), 30);


    let mut fluid = Fluid::new(particles.clone(), density, pressure_constant, viscosity, device.clone());

    // Bind the fluid to the compute pipeline
    let compute_pipeline = FluidComputePipeline::new(device.clone());
    fluid.bind_compute(&compute_pipeline);
   
    let mut frame_time = 0.0;
    let mut capturing_mouse_input = false;
    let mut last_position_cursor = None::<PhysicalPosition<f64>>;
    let timestep = 0.01; 
    
    event_loop.run(move |event, _, control_flow| {
        let before = Instant::now();
        
        if renderer.reset_flag {
            fluid.reset(particles.clone());
            fluid.bind_compute(&compute_pipeline);

            compute_pipeline.compute(timestep, &fluid, &BoundingBox { x1: 0.0, x2: 5.0, z1: 0.0, z2: 5.0, y1: 0.0, y2: 10.0, damping_factor: 0.5 }, graphics_queue.clone());
            
            renderer.reset_flag = false;
        }

        match event {
            // Handle keyboard movement (WASD)
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput{
                    input: KeyboardInput{ virtual_keycode, state: ElementState::Pressed, .. },
                    ..
                },
                ..
            } => {
                renderer.handle_keyboard_events(virtual_keycode); 
            }
            
            // Handle mouse movement
            Event::WindowEvent { 
                event: WindowEvent::CursorLeft { .. }, 
                ..
            } => {last_position_cursor = None;}
            Event::WindowEvent { 
                event: WindowEvent::MouseInput { button: MouseButton::Left, state: ElementState::Pressed, .. }, 
                ..
            } => {capturing_mouse_input = true;}
            Event::WindowEvent { 
                event: WindowEvent::MouseInput { button: MouseButton::Left, state: ElementState::Released, .. }, 
                ..
            } => {capturing_mouse_input = false;}
            Event::WindowEvent { 
                event: WindowEvent::CursorMoved { position, .. }, 
                ..
            } => {
                if !capturing_mouse_input { return; }

                if let Some(prev) = last_position_cursor {
                    let dxdy = PhysicalPosition{ x: position.x - prev.x, y: position.y - prev.y };
                    
                    renderer.handle_cursor_events(dxdy);
                }
                
                last_position_cursor = Some(position);
            }

            // Handle mouse scroll wheel
            Event::WindowEvent {
                event: WindowEvent::MouseWheel { delta, .. },
                ..
            } => {
                match delta {
                    MouseScrollDelta::LineDelta(_, dw) => {
                        renderer.handle_mwheel_events(dw);
                    }

                    _ => { return; }
                }
            }

            Event::WindowEvent { 
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                renderer.update_viewport();
                renderer.recreate_swapchain();
            }
            Event::RedrawEventsCleared => {
                let mut buffer = fluid.models(); 
                let mut num_particles = fluid.len();

                if !renderer.stop_flag {    
                    (buffer, num_particles) = compute_pipeline.compute(timestep, &fluid, &BoundingBox { x1: 0.0, x2: 3.0, z1: 0.0, z2: 3.0, y1: 0.0, y2: 10.0, damping_factor: 0.5 }, graphics_queue.clone());
                }

                let first = renderer.begin();
                
                renderer.draw(&bounding_box);
                renderer.draw_particles(&sphere, buffer, num_particles);
                renderer.end(first);

                frame_time = before.elapsed().as_micros() as f32;

                // println!("fps {:?} {:?}", 1.0 / (frame_time / 1_000_000_f32), before.elapsed()); 
            }
            _ => ()
        }
    });
}
