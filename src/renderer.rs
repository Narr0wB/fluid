#[allow(unused_imports)]

use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Features, DeviceExtensions, physical::PhysicalDeviceType, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateFlags, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents};
use vulkano::image::ImageLayout;
use vulkano::image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::pipeline::graphics::color_blend::ColorComponents;
use vulkano::pipeline::graphics::rasterization::PolygonMode;
use vulkano::pipeline::graphics::{color_blend::{ColorBlendAttachmentState, ColorBlendState}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::RasterizationState,
    vertex_input::{Vertex, VertexDefinition}, viewport::{Viewport, ViewportState}, GraphicsPipelineCreateInfo, depth_stencil::{DepthState, DepthStencilState}};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::{layout::PipelineDescriptorSetLayoutCreateInfo, DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo, Pipeline};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::{self, GpuFuture};
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorType, DescriptorSetLayoutBinding};
use vulkano::{Validated, VulkanError, VulkanLibrary};

use std::any::Any;
use std::sync::Arc;

use winit::{event::{Event, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::*};

use nalgebra::{Vector3, Unit};

use crate::camera::Camera;
use crate::mesh::MVertex;
use crate::mesh::Mesh;


mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450

            layout(set = 0, binding = 0) uniform view {
                mat4 mat;
            } v;
            
            layout(location = 0) in vec3 position;
            
            void main() {
                gl_Position = v.mat * vec4(position, 1.0);
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450

            layout(location = 0) out vec4 f_color;
            
            void main() {
                vec4 modified_color = vec4(1.0);
                f_color = modified_color;
            }
        "
    }
}

pub struct Renderer {
    // Window and device specific stuff
    instance: Arc<Instance>,
    device: Arc<Device>,
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    
    // Camera and scene
    pub scene_camera: Camera,
    camera_buffer: Arc<Subbuffer<[[f32; 4]; 4]>>,
    
    // Rendering specific stuff
    render_pass: Arc<RenderPass>,
    depth_buffer: Arc<Image>,
    framebuffers: Vec<Arc<Framebuffer>>,
    viewport: Viewport,
    pipeline: Arc<GraphicsPipeline>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    render_queue: Arc<Queue>,
    cmd_buffer_allocator: StandardCommandBufferAllocator
}

impl Renderer {
    pub fn init(instance: Arc<Instance>, event_loop: &EventLoop<()>, mut camera: Camera) -> Self {
        // Create a new window and the window abstraction (surface)
        let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        // Set up the camera scene with the correct aspect ratio
        let (width, height): (u32, u32) = window.inner_size().into();
        camera.set_aspect_ratio((width / height) as f32);
        
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
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
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
            .expect("eddu");

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

        let render_queue = queues.next().unwrap();

        let mem_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        
        let (swapchain, images) = {
            let surface_capabilites = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let image_format = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0;
        
            Swapchain::new(
                device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: 2,
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilites
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let depth_buffer = Image::new(
            mem_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D16_UNORM,
                extent: images[0].extent(),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        ).unwrap();

        let depth_buffer_view = ImageView::new_default(depth_buffer.clone()).unwrap();

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments:  {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                    final_layout: ImageLayout::PresentSrc
                },
                depth: {
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        )
        .unwrap();
         
        let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        view,
                        depth_buffer_view.clone()
                    ],
                    ..Default::default()
                }
            ).unwrap()
        })
        .collect::<Vec<_>>();
        
        let vs = vs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let fs = fs::load(device.clone()).unwrap().entry_point("main").unwrap();

        let vertex_input_state = MVertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();

        let shader_stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs)
        ].into_iter().collect();

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&shader_stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap()
        ).unwrap();

        let mut rasterizer_state = RasterizationState::default();
        rasterizer_state.polygon_mode = PolygonMode::Line;
        rasterizer_state.line_width = 1.0f32;
        
        let mut color_attachment_state = ColorBlendAttachmentState::default();
        color_attachment_state.color_write_mask = ColorComponents::all();
        color_attachment_state.blend = None;

        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: shader_stages,
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(rasterizer_state), 
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    Subpass::from(render_pass.clone(), 0).unwrap().num_color_attachments(), 
                    color_attachment_state 
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(Subpass::from(render_pass.clone(), 0).unwrap().into()),
                ..GraphicsPipelineCreateInfo::layout(layout.clone())
            }
        ).unwrap();

        let camera_buffer = Arc::new(
            Buffer::new_sized(
                mem_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                }
            ).unwrap()
        );

        let descriptor_set_allocator = 
            StandardDescriptorSetAllocator::new(device.clone(), Default::default());

        let vs_layout = layout.set_layouts().get(0).unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            vs_layout.clone(),
            [WriteDescriptorSet::buffer(0, (*camera_buffer).clone())],
            []
        ).unwrap();

        // Create the command buffer
        let cmd_buffer_allocator = 
            StandardCommandBufferAllocator::new(device.clone(), Default::default()); 

        let mut viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [0.0, 0.0],
            depth_range: 0.0..=1.0,
        };

        let extent = images[0].extent();
        viewport.extent = [extent[0] as f32, extent[1] as f32];

        return Renderer {
            instance,
            device,
            window,
            
            scene_camera: camera,
            
            render_pass,
            swapchain,
            images,
            depth_buffer,
            framebuffers,
            viewport,

            pipeline,
            render_queue,
            camera_buffer,
            cmd_buffer_allocator,
            descriptor_set
        }
    }

    pub fn get_memory_allocator(&self) -> StandardMemoryAllocator {
        return StandardMemoryAllocator::new_default(self.device.clone());
    }

    pub fn recreate_swapchain(&mut self) {
        let extent = [self.viewport.extent[0] as u32, self.viewport.extent[1] as u32];

        let (new_swapchain, new_images) = self.swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: extent,
                ..self.swapchain.create_info()
            }).unwrap();

        let new_framebuffers = new_images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();

                Framebuffer::new(
                    self.render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![
                            view,
                            ImageView::new_default(self.depth_buffer.clone()).unwrap() 
                        ],
                        ..Default::default()
                    }
                ).unwrap()
            })
            .collect::<Vec<_>>();

        self.framebuffers = new_framebuffers;
        self.swapchain = new_swapchain;
        self.images = new_images;
    }
    
    pub fn update_viewport(&mut self) {
        let new_window_extent: [f32; 2] = self.window.inner_size().into();
        
        self.viewport = Viewport {
            offset: [0.0, 0.0],
            extent: new_window_extent,
            depth_range: 0.0..=1.0,
        };
    }

    pub fn draw(&mut self, obj: &Mesh) {
        // Set up camera matrices and data
        let vp = self.scene_camera.vp().transpose();

        let data = [
            [vp[(0, 0)], vp[(0, 1)], vp[(0, 2)], vp[(0, 3)]],
            [vp[(1, 0)], vp[(1, 1)], vp[(1, 2)], vp[(1, 3)]],
            [vp[(2, 0)], vp[(2, 1)], vp[(2, 2)], vp[(2, 3)]],
            [vp[(3, 0)], vp[(3, 1)], vp[(3, 2)], vp[(3, 3)]]
        ];

        *self.camera_buffer.write().unwrap() = data;

        
        // Acquire the next image in the swapchain
        let (image_index, _, future) = acquire_next_image(self.swapchain.clone(), None).expect("eddu failed :(");
        
        // Create the command builder
        let mut cmd_builder = AutoCommandBufferBuilder::primary(
            &self.cmd_buffer_allocator,
            self.render_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        ).unwrap();
        
        // Actually build the command
        cmd_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.0, 0.0, 0.0, 1.0].into()),
                        Some(1f32.into())
                    ],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone()
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                }
            ).unwrap()
            .set_viewport(0, [self.viewport.clone()].into_iter().collect()).unwrap()
            .bind_pipeline_graphics(self.pipeline.clone()).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                vec![self.descriptor_set.clone()]
            ).unwrap()
            .bind_vertex_buffers(0, obj.get_vertex_buffer().clone()).unwrap()
            .bind_index_buffer(obj.get_index_buffer().clone()).unwrap()
            .draw_indexed(obj.get_index_buffer().len() as u32, 1, 0, 0, 0).unwrap()
            .end_render_pass(Default::default()).unwrap();

        let cmd_buf = cmd_builder.build().unwrap();

        let super_future = sync::now(self.device.clone())
            .join(future)
            .then_execute(self.render_queue.clone(), cmd_buf).unwrap()
            .then_swapchain_present(
                self.render_queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index)
            )
            .then_signal_fence_and_flush()
            .expect("Could not send the command buffer to the GPU!");

    } 
}
