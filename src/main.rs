#[allow(unused_imports)]

use std::sync::Arc;
use fluid::{particle_cube, BoundingBox, Fluid, PARTICLE_RADIUS};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::format::Format;
use vulkano::descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Features, DeviceExtensions, physical::PhysicalDeviceType, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents};
use vulkano::half::vec::HalfFloatVecExt;
use vulkano::image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::pipeline::graphics::{color_blend::{ColorBlendAttachmentState, ColorBlendState}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::RasterizationState,
    vertex_input::{VertexDefinition}, viewport::{Viewport, ViewportState}, GraphicsPipelineCreateInfo, depth_stencil::{DepthState, DepthStencilState}};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::{layout::PipelineDescriptorSetLayoutCreateInfo, DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, Subpass};
use vulkano::swapchain::{acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::{self, GpuFuture};
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorType, DescriptorSetLayoutBinding};
use vulkano::{memory, Validated, VulkanError, VulkanLibrary};
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyboardInput, MouseButton, MouseScrollDelta, ScanCode, VirtualKeyCode};

use nalgebra::{*};

mod mesh;
mod camera;
mod renderer;
mod fluid;

use crate::camera::Camera;
use crate::renderer::Renderer;
use crate::mesh::{*};
use crate::fluid::Particle;

use winit::{event::{Event, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::*};

use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

    let queue_family_compute = phys_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(i, q)| {
            q.queue_flags.intersects(QueueFlags::COMPUTE)
        }).unwrap() as u32;

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
                ..Default::default()
            },
            enabled_extensions: device_extensions,
            queue_create_infos: vec![
                QueueCreateInfo {
                    queue_family_index: queue_family_graphics,
                    ..Default::default()
                },
                QueueCreateInfo {
                    queue_family_index: queue_family_compute,
                    ..Default::default()
                }
            ],
            ..Default::default()
        }
    )
    .unwrap();

    let graphics_queue = queues
        .find(|q| {
            q.queue_family_index() == queue_family_graphics
        }).unwrap();

    let compute_queue = queues
        .find(|q| {
            q.queue_family_index() == queue_family_graphics
        }).unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let mut camera = Camera::new(
        Vector3::new(7.0, 5.0, 3.0),                       // position
        Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),   // orientation
        None,                                               // aspect_ratio
        50.0                                                // FOV
    );

    camera.set_target(Vector3::new(0.0, 0.0, 0.0));

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
    
    let model_matrix = Matrix4::new_scaling(5.0);
     
    let mesh = Mesh::new(memory_allocator.clone(), vertices, indices, model_matrix); 

    // let mut current_angle: f32 = 0.0;
    // let rotation_speed: f33 = 2.0 * std::f32::consts::PI / 1000.0;
    let mut last_position_cursor = None::<PhysicalPosition<f64>>;

    let mut capturing_mouse_input = false;
    let sphere = create_sphere(PARTICLE_RADIUS, Vector3::new(0.0, 0.0, 0.0), 20, memory_allocator.clone());
    
    let particles = particle_cube(Vector3::new(2.0, 4.0, 2.0), 6);
    let mut fluid = Fluid::new(particles, 2.0, 0.5, memory_allocator.clone());

    event_loop.run(move |event, _, control_flow| {
        // let now = Instant::now();
        // let frame_duration = now - start;
        // start = Instant::now();
        // 
        // current_angle += rotation_speed * ((frame_duration.as_millis() as f32 / 10.0) as f32);

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
                // let before = Instant::now();
                let (buffer, num_particles) = fluid.update(0.01, &BoundingBox { x1: 0.0, x2: 5.0, z1: 0.0, z2: 5.0, y1: 0.0, y2: 5.0, damping_factor: 0.7 });
                // println!("it took {:.2?}", before.elapsed());

                let first = renderer.begin();
                
                renderer.draw(&mesh);
                // renderer.draw(&sphere);
                renderer.draw_particles(&sphere, buffer, num_particles);
                
                renderer.end(first);
            }
            _ => ()
        }

    });

    // println!("odio gli zingari!");
    // println!("eddu gaming");
    
}
