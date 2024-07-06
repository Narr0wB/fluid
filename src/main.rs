#[allow(unused_imports)]

use std::sync::Arc;
use nalgebra::{*};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::format::Format;
use vulkano::descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Features, DeviceExtensions, physical::PhysicalDeviceType, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents};
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
use vulkano::{Validated, VulkanError, VulkanLibrary};

mod mesh;
mod camera;
mod renderer;
use crate::camera::Camera;
use crate::renderer::Renderer;
use crate::mesh::{MVertex, Mesh};

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
        ..DeviceExtensions::empty()
    };

    // Select physical device
    let (phys_device, queue_family_index) = instance
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
                ..Default::default()
            },
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo{
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        }
    )
    .unwrap();

    let mut camera = Camera::new(
        Vector3::new(2.0, 4.0, -4.0),                       // position
        Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),   // orientation
        None,                                               // aspect_ratio
        50.0                                                // FOV
    );

    camera.set_target(Vector3::new(0.0, 0.0, 0.0));

    let mut renderer = Renderer::init(instance, &event_loop, camera, device.clone(), queues.next().unwrap().clone()); 
    
    let vertices: Vec<MVertex> = vec![
        MVertex {position: [0.0, 0.0, 0.0]}, MVertex {position: [0.0, 1.5, 0.0]},
        MVertex {position: [0.0, 0.0, 1.0]}, MVertex {position: [0.0, 1.5, 1.0]},

        MVertex {position: [2.0, 0.0, 0.0]}, MVertex {position: [2.0, 1.5, 0.0]},
        MVertex {position: [2.0, 0.0, 1.0]}, MVertex {position: [2.0, 1.5, 1.0]},    
    ];
    
    let indices: Vec<u32> = vec![
        0, 1, 3, 3, 2, 0,

        2, 6, 3, 3, 7, 6,

        4, 5, 7, 7, 6, 4,
        
        0, 4, 1, 1, 5, 4
    ]; 
    
    let model_matrix = Matrix4::new_scaling(2.0);
     
    let mesh = Mesh::new(renderer.get_memory_allocator().try_into().unwrap(), vertices, indices, model_matrix); 


    // let command_buffer_allocator = 
    //     StandardCommandBufferAllocator::new(device.clone(), Default::default());
    //
    // let mut recreate_swapchain = false;
    // let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut start = Instant::now();
    let mut current_angle: f32 = 0.0;
    let rotation_speed: f32 = 2.0 * std::f32::consts::PI / 1000.0;

    
    event_loop.run(move |event, _, control_flow| {
        let now = Instant::now();
        let frame_duration = now - start;
        start = Instant::now();
        
        current_angle += rotation_speed * ((frame_duration.as_millis() as f32 / 10.0) as f32);

        match event {
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
                renderer.scene_camera.set_position(renderer.scene_camera.get_position() + Vector3::new(0.0, 0.00, 0.0));
                let first = renderer.begin();
                renderer.draw(&mesh); 
                renderer.end(first);
            }
            _ => ()
        }

    });

    // println!("odio gli zingari!");
    // println!("eddu gaming");
    
}
