#[allow(unused_imports)]

use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocatorCreateInfo;
use vulkano::command_buffer::{CommandBufferInheritanceRenderPassType, PrimaryAutoCommandBuffer, SecondaryAutoCommandBuffer, SecondaryCommandBufferAbstract};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Features, DeviceExtensions, physical::PhysicalDeviceType, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateFlags, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{allocator::StandardCommandBufferAllocator, CommandBufferInheritanceInfo, AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents};
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
                mat4 model;
                mat4 vp;
            } v;
            
            layout(location = 0) in vec3 position;
            
            void main() {
                gl_Position = v.model * v.vp * vec4(position, 1.0);
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
    // Window 
    instance: Arc<Instance>,
    window: Arc<Window>,
    
    // Device specific stuff
    device: Arc<Device>,
    render_queue: Arc<Queue>,
    
    // Camera and scene
    pub scene_camera: Camera,
    descriptor_set: Arc<PersistentDescriptorSet>,
    mvp_buffer: Arc<Subbuffer<[f32]>>,

    // Render buffers
    depth_buffer: Arc<Image>,
    framebuffers: Vec<Arc<Framebuffer>>,
    images: Vec<Arc<Image>>,
    acquired_image: u32,
    swapchain: Arc<Swapchain>,

    // Rendering specific stuff
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
    pipeline: Arc<GraphicsPipeline>,

    // Command buffer stuff
    cmd_buffer_allocator: StandardCommandBufferAllocator,
    secondary_cmd_buffers: Vec<Arc<dyn SecondaryCommandBufferAbstract>>
}

impl Renderer {
    pub fn init(instance: Arc<Instance>, event_loop: &EventLoop<()>, mut camera: Camera, device: Arc<Device>, render_queue: Arc<Queue>) -> Self {
        // Create a new window and the window abstraction (surface)
        let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        
        // Set up the camera scene with the correct aspect ratio
        let (width, height): (u32, u32) = window.inner_size().into();
        camera.set_aspect_ratio(height as f32 / width as f32);

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

        let mvp_buffer = Arc::new(
            Buffer::new_slice::<f32>(
                mem_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                4 * 4 * 2 * std::mem::size_of::<f32>() as u64 
            ).unwrap()
        );

        let descriptor_set_allocator = 
            StandardDescriptorSetAllocator::new(device.clone(), Default::default());

        let vs_layout = layout.set_layouts().get(0).unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            vs_layout.clone(),
            [WriteDescriptorSet::buffer(0, (*mvp_buffer).clone())],
            []
        ).unwrap();

        // Create the command buffer
        let cmd_buffer_allocator = 
            StandardCommandBufferAllocator::new(device.clone(), StandardCommandBufferAllocatorCreateInfo {secondary_buffer_count: 32, ..Default::default()}); 

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
            acquired_image: 0,
            depth_buffer,
            framebuffers,
            viewport,
            pipeline,
            render_queue,
            mvp_buffer,
            cmd_buffer_allocator,
            descriptor_set,
            secondary_cmd_buffers: vec![]
        }
    }

    pub fn get_memory_allocator(&self) -> Arc<StandardMemoryAllocator> {
        return Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));
    }

    pub fn recreate_swapchain(&mut self) {
        let extent = [self.viewport.extent[0] as u32, self.viewport.extent[1] as u32];

        let (new_swapchain, new_images) = self.swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: extent,
                ..self.swapchain.create_info()
            }).unwrap();

        let new_depth_buffer = Image::new(
            self.get_memory_allocator(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D16_UNORM,
                extent: new_images[0].extent(),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        ).unwrap();

        let new_framebuffers = new_images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                
                Framebuffer::new(
                    self.render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![
                            view,
                            ImageView::new_default(new_depth_buffer.clone()).unwrap() 
                        ],
                        ..Default::default()
                    }
                ).unwrap()
            })
            .collect::<Vec<_>>();
        
        self.framebuffers = new_framebuffers;
        self.depth_buffer = new_depth_buffer; 
        self.swapchain = new_swapchain;
        self.images = new_images;
    }
    
    pub fn update_viewport(&mut self) {
        let new_window_extent: [f32; 2] = self.window.inner_size().into();

        self.scene_camera.set_aspect_ratio(new_window_extent[1] / new_window_extent[0]);
        
        self.viewport = Viewport {
            offset: [0.0, 0.0],
            extent: new_window_extent,
            depth_range: 0.0..=1.0,
        };
    }

    pub fn begin(&mut self) -> Box<dyn GpuFuture> {
        
        
        // Acquire the next image in the swapchain
        let (image_index, _, future) = acquire_next_image(self.swapchain.clone(), None).expect("eddu failed :(");
        
        self.acquired_image = image_index;

        Box::new(future)
    }

    pub fn draw_particles(&mut self, particle: Vec<MVertex>) {
        // let mut cmd_builder = AutoCommandBufferBuilder::primary(
        //     &self.cmd_buffer_allocator,
        //     self.render_queue.queue_family_index(),
        //     CommandBufferUsage::OneTimeSubmit
        // ).unwrap();
        //
        // cmd_builder
        //     .begin_render_pass(
        //         RenderPassBeginInfo {
        //             clear_values: vec![
        //                 Some([0.0, 0.0, 0.0, 1.0].into()),
        //                 Some(1f32.into())
        //             ],
        //             ..RenderPassBeginInfo::framebuffer(
        //                 self.framebuffers[self.acquired_image as usize].clone()
        //             )
        //         },
        //         SubpassBeginInfo {
        //             contents: SubpassContents::Inline,
        //             ..Default::default()
        //         }
        //     ).unwrap()
        //     .set_viewport(0, [self.viewport.clone()].into_iter().collect()).unwrap();
        //
        // let cmd_buf = cmd_builder.build().unwrap();
        //
        // let super_future = prev_future
        //     .then_execute(self.render_queue.clone(), cmd_buf).unwrap()
        //     .then_signal_fence_and_flush().unwrap();
        //
        // Box::new(super_future)
    }

    pub fn draw(&mut self, obj: &Mesh) {
        // Set up camera matrices and data
        let vp = self.scene_camera.vp();
        let model = obj.get_model_matrix();

        let data = [
            [vp[(0, 0)], vp[(1, 0)], vp[(2, 0)], vp[(3, 0)]],
            [vp[(0, 1)], vp[(1, 1)], vp[(2, 1)], vp[(3, 1)]],
            [vp[(0, 2)], vp[(1, 2)], vp[(2, 2)], vp[(3, 2)]],
            [vp[(0, 3)], vp[(1, 3)], vp[(2, 3)], vp[(3, 3)]],

            [model[(0, 0)], model[(1, 0)], model[(2, 0)], model[(3, 0)]],
            [model[(0, 1)], model[(1, 1)], model[(2, 1)], model[(3, 1)]],
            [model[(0, 2)], model[(1, 2)], model[(2, 2)], model[(3, 2)]],
            [model[(0, 3)], model[(1, 3)], model[(2, 3)], model[(3, 3)]],
        ];

        *self.mvp_buffer.write().unwrap() = vp.as_slice();

        // Create the command builder
        let mut cmd_builder = AutoCommandBufferBuilder::secondary(
            &self.cmd_buffer_allocator,
            self.render_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(CommandBufferInheritanceRenderPassType::BeginRenderPass( (Subpass::from(self.render_pass.clone(), 0).unwrap()).into() )),
                ..Default::default()
            }
        ).unwrap();

        // Record commands
        cmd_builder
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
            .draw_indexed(obj.get_index_buffer().len() as u32, 1, 0, 0, 0).unwrap();
            

        let cmd_buf = cmd_builder.build().unwrap();
        
        self.secondary_cmd_buffers.push(cmd_buf);
    }

    pub fn end(&mut self, last_future: Box<dyn GpuFuture>) {
        let mut cmd_builder = AutoCommandBufferBuilder::primary(
            &self.cmd_buffer_allocator,
            self.render_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        ).unwrap();

        cmd_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.0, 0.0, 0.0, 1.0].into()),
                        Some(1f32.into())
                    ],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[self.acquired_image as usize].clone()
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..Default::default()
                }
            ).unwrap();

        let _ = cmd_builder
            .execute_commands_from_vec(self.secondary_cmd_buffers.clone()).unwrap();

        cmd_builder
            .end_render_pass(Default::default()).unwrap();

        let cmd = cmd_builder.build().unwrap();
        
        self.secondary_cmd_buffers.clear();
        let _ = last_future
            .then_execute(self.render_queue.clone(), cmd).unwrap()
            .then_swapchain_present(
                    self.render_queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), self.acquired_image)
                )
            .then_signal_fence_and_flush()
            .expect("Could not send the command buffer to the GPU!");
    }
}
