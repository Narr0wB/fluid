#[allow(unused_imports)]

use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocatorCreateInfo;
use vulkano::command_buffer::{CommandBufferInheritanceRenderPassType, PrimaryAutoCommandBuffer, SecondaryAutoCommandBuffer, SecondaryCommandBufferAbstract};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateFlags, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{allocator::StandardCommandBufferAllocator, CommandBufferInheritanceInfo, AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents};
use vulkano::image::ImageLayout;
use vulkano::image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::pipeline::graphics::color_blend::ColorComponents;
use vulkano::pipeline::graphics::input_assembly::PrimitiveTopology;
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
use winit::dpi::PhysicalPosition;
use winit::event::{KeyboardInput, VirtualKeyCode};

use std::any::Any;
use std::convert::identity;
use std::sync::Arc;
use std::time::Instant;

use winit::{event::{Event, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::*};

use nalgebra::{*};

use crate::camera::Camera;
use crate::mesh::{Mesh, MVertex};
use crate::fluid::Particle;


mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450

            layout(std140, push_constant) uniform view {
                mat4 vp;
                mat4 model;
            } v;
            
            layout(location = 0) in vec3 position;
            
            void main() {
                gl_Position = v.vp * v.model * vec4(position, 1.0);
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

mod pvs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450

            struct Particle {
                mat4 model;
            };
            
            layout(std140, push_constant) uniform view {
                mat4 vp; 
            } v;
 
            layout(set = 0, binding = 0) buffer particles {
                Particle particles[];
            } p;
            
            layout(location = 0) in vec3 position;
            layout(location = 0) out vec4 param;

            void main() {
                mat4 model = p.particles[gl_InstanceIndex].model;
                float color = model[0][1];
                model[0][1] = 0.0;
                gl_Position = v.vp * model * vec4(position, 1.0);
                param = vec4(color);
            }
        "
    }
}

mod pfs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450

            layout(location = 0) out vec4 frag_color;
            layout(location = 0) in vec4 param;

            vec3 blue   = vec3(0.,0.114,1.);
            vec3 teal   = vec3(0.,1.,0.612);
            vec3 yellow = vec3(0.953,1.,0.);
            vec3 red    = vec3(1.,0.,0.);

            vec3 grad(float t) {
                t = clamp(t, 0.0, 1.0);

                float segment1 = 0.54;
                float segment2 = 0.22;
                float segment3 = 0.24;

                if (t < segment1) {
                    return mix(blue, teal, t / segment1);
                }
                else if (t < segment1 + segment2) {
                    return mix(teal, yellow, (t - segment1) / segment2);
                }
                
                return mix(yellow, red, (t - (segment1 + segment2)) / segment3);
            }

            void main() {
                frag_color = vec4(grad(param.x), 1.0);
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
    move_sensitivity: f32,
    mouse_sensitivity: f32,
    fov_step: f32,

    // Render buffers
    std_memory_allocator: Arc<StandardMemoryAllocator>,
    depth_buffer: Arc<Image>,
    framebuffers: Vec<Arc<Framebuffer>>,
    images: Vec<Arc<Image>>,
    acquired_image: u32,
    swapchain: Arc<Swapchain>,

    // Rendering specific stuff
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
    obj_pipeline: Arc<GraphicsPipeline>,
    particle_pipeline: Arc<GraphicsPipeline>,

    // Command buffer stuff
    cmd_buffer_allocator: StandardCommandBufferAllocator,
    secondary_cmd_buffers: Vec<Arc<dyn SecondaryCommandBufferAbstract>>,

    // Command flags
    pub reset_flag: bool,
    pub stop_flag: bool,
    pub showcase_flag: bool
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
                    min_image_count: 4,
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

        let obj_pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: shader_stages,
                vertex_input_state: Some(vertex_input_state.clone()),
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
                    color_attachment_state.clone() 
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(Subpass::from(render_pass.clone(), 0).unwrap().into()),
                ..GraphicsPipelineCreateInfo::layout(layout.clone())
            }
        ).unwrap();

        let pvs = pvs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let pfs = pfs::load(device.clone()).unwrap().entry_point("main").unwrap();
        
        let p_shader_stages = vec![
            PipelineShaderStageCreateInfo::new(pvs),
            PipelineShaderStageCreateInfo::new(pfs)
        ].into();

        let p_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&p_shader_stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap()
        ).unwrap();

        let particle_pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: p_shader_stages,
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState { topology: PrimitiveTopology::TriangleList, ..InputAssemblyState::default() }),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState { polygon_mode: PolygonMode::Fill, ..Default::default() }),
                depth_stencil_state: Some(DepthStencilState {depth: Some(DepthState::simple()), ..Default::default()}),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    Subpass::from(render_pass.clone(), 0).unwrap().num_color_attachments(), 
                    color_attachment_state
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(Subpass::from(render_pass.clone(), 0).unwrap().into()),
                ..GraphicsPipelineCreateInfo::layout(p_layout.clone())
            }
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
            obj_pipeline,
            particle_pipeline,
            render_queue,
            cmd_buffer_allocator,
            std_memory_allocator: mem_allocator,
            secondary_cmd_buffers: vec![],
            move_sensitivity: 0.5,
            mouse_sensitivity: 0.01,
            fov_step: 5.0,
            reset_flag: false,
            stop_flag: false,
            showcase_flag: false
        }
    }

    pub fn handle_keyboard_events(&mut self, input: Option<VirtualKeyCode>) {
        match input {

            Some(winit::event::VirtualKeyCode::W) => {
                let current_position = self.scene_camera.get_position();

                let aheadXZ = { 
                    let mut tmp = self.scene_camera.get_orientation(); 
                    tmp.y = 0.0;
                    tmp
                };

                self.scene_camera.set_position(current_position + (self.move_sensitivity * aheadXZ));
            }

            Some(winit::event::VirtualKeyCode::A) => {
                let current_position = self.scene_camera.get_position();
                let ahead = self.scene_camera.get_orientation();
                let up = Vector3::new(0.0, 1.0, 0.0);

                let right = up.cross(&ahead).normalize();

                self.scene_camera.set_position(current_position - (self.move_sensitivity * right));
            }

            Some(winit::event::VirtualKeyCode::S) => {
                let current_position = self.scene_camera.get_position();
                
                let aheadXZ = { 
                    let mut tmp = self.scene_camera.get_orientation(); 
                    tmp.y = 0.0;
                    tmp
                };
                
                self.scene_camera.set_position(current_position - (self.move_sensitivity * aheadXZ)); 
            }

            Some(winit::event::VirtualKeyCode::D) => {
                let current_position = self.scene_camera.get_position();
                let ahead = self.scene_camera.get_orientation();
                let up = Vector3::new(0.0, 1.0, 0.0);

                let right = up.cross(&ahead).normalize();

                self.scene_camera.set_position(current_position + (self.move_sensitivity * right));
            }

            Some(winit::event::VirtualKeyCode::Space) => {
                let current_position = self.scene_camera.get_position();
                let up = Vector3::new(0.0, 1.0, 0.0);

                self.scene_camera.set_position(current_position + (self.move_sensitivity * up));
            }

            Some(winit::event::VirtualKeyCode::LShift) => {
                let current_position = self.scene_camera.get_position();
                let up = Vector3::new(0.0, 1.0, 0.0);

                self.scene_camera.set_position(current_position - (self.move_sensitivity * up));
            }

            Some(winit::event::VirtualKeyCode::R) => {
                self.reset_flag = true;
            }

            Some(winit::event::VirtualKeyCode::C) => {
                self.stop_flag ^= true;
            }

            Some(winit::event::VirtualKeyCode::T) => {
                self.showcase_flag ^= true;
            }
            
            _ => {
                return;
            }
        }
    }

    pub fn handle_cursor_events(&mut self, dxdy: PhysicalPosition<f64>) {
        let ahead = self.scene_camera.get_orientation();
        let up = Vector3::new(0.0, 1.0, 0.0);

        let right = up.cross(&ahead).normalize();
        let real_up = right.cross(&ahead).normalize();

        let scaling_factor = (ahead.x.powf(2.0) + ahead.z.powf(2.0)).sqrt() + 0.01;
       
        let dx = if dxdy.x == 0.0 {
                Vector3::new(0.0, 0.0, 0.0)
            }
            else { 
                (right * -dxdy.x as f32).normalize()
            };
        
        let dy = if dxdy.y == 0.0 {
                Vector3::new(0.0, 0.0, 0.0)
            }
            else { 
                (real_up * -dxdy.y as f32).normalize()
            }; 

        let change = self.mouse_sensitivity * scaling_factor * (dx + dy);

        self.scene_camera.set_orientation((ahead + change).normalize());
    }

    pub fn handle_mwheel_events(&mut self, dw: f32) {
        let current_fov = self.scene_camera.get_fov();

        self.scene_camera.set_fov(current_fov - self.fov_step * dw);
    }

    pub fn recreate_swapchain(&mut self) {
        let extent = [self.viewport.extent[0] as u32, self.viewport.extent[1] as u32];

        let (new_swapchain, new_images) = self.swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: extent,
                ..self.swapchain.create_info()
            }).unwrap();

        let new_depth_buffer = Image::new(
            self.std_memory_allocator.clone(),
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

    pub fn draw_particles(&mut self, sphere: &Mesh, particle_models: &Subbuffer<[f32]>, particles: u32) {
        let descriptor_set_allocator = 
            StandardDescriptorSetAllocator::new(self.device.clone(), Default::default());

        let pvs_layout = self.particle_pipeline.layout().set_layouts().get(0).unwrap();

        let p_descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            pvs_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, (*particle_models).clone())
            ],
            []
        ).unwrap();
        
        #[repr(C)]
        #[derive(BufferContents)]
        struct PushConstants {
            vp: [f32; 16]
        }

        let push_constants = PushConstants {
            vp: self.scene_camera.vp().as_slice().try_into().unwrap()
        };

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
            .bind_pipeline_graphics(self.particle_pipeline.clone()).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.particle_pipeline.layout().clone(),
                0,
                vec![p_descriptor_set.clone()]
            ).unwrap()
            .push_constants(self.particle_pipeline.layout().clone(), 0, push_constants).unwrap()
            .bind_vertex_buffers(0, sphere.get_vertex_buffer().clone()).unwrap()
            .bind_index_buffer(sphere.get_index_buffer().clone()).unwrap()
            .draw_indexed(sphere.len() as u32, particles as u32, 0, 0, 0).unwrap();

        let cmd_buf = cmd_builder.build().unwrap();
        
        self.secondary_cmd_buffers.push(cmd_buf);
    }

    pub fn draw(&mut self, obj: &Mesh) {
        let model = obj.get_model_matrix();

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
        
        #[repr(C)]
        #[derive(BufferContents)]
        struct PushConstants {
            vp: [f32; 16],
            model: [f32; 16]
        }

        let push_constants = PushConstants {
            vp: self.scene_camera.vp().as_slice().try_into().unwrap(),
            model: model.as_slice().try_into().unwrap()
        };

        // Record commands
        cmd_builder
            .set_viewport(0, [self.viewport.clone()].into_iter().collect()).unwrap()
            .bind_pipeline_graphics(self.obj_pipeline.clone()).unwrap()
            .push_constants(self.obj_pipeline.layout().clone(), 0, push_constants).unwrap() 
            .bind_vertex_buffers(0, obj.get_vertex_buffer().clone()).unwrap()
            .bind_index_buffer(obj.get_index_buffer().clone()).unwrap()
            .draw_indexed(obj.len() as u32, 1, 0, 0, 0).unwrap();
            

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
                        Some([0.749,0.749,0.749, 1.0].into()),
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

        self.secondary_cmd_buffers.clear();
        
        cmd_builder
            .end_render_pass(Default::default()).unwrap();

        let cmd = cmd_builder.build().unwrap(); 

        let wait_future = sync::now(self.device.clone()) 
            .then_execute(self.render_queue.clone(), cmd).unwrap()
            .then_signal_fence_and_flush().unwrap();

        wait_future.wait(None).unwrap();
        
        let _ = last_future 
            .then_swapchain_present(
                self.render_queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), self.acquired_image)
            )
            .then_signal_fence_and_flush().unwrap();
    }
}
