pub mod fluid;
pub mod mesh;

use std::sync::Arc;

use glam::Mat4;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::camera::Camera;
use crate::gpu;

use fluid::{FluidInstance, FluidMaterial, FluidRenderer};
use mesh::{Mesh, MeshMaterial, MeshRenderer, Topology, Vertex};

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const SCENE_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const COMMON_WGSL: &str = include_str!("../shaders/common.wgsl");

pub(crate) fn shader_module(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}\n{source}").into()),
    })
}

pub(crate) fn create_target(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

pub(crate) fn fullscreen_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    vertex: &str,
    fragment: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fragment),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vertex),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fragment),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    view_projection: [[f32; 4]; 4],
    position: [f32; 4],
}

impl CameraUniform {
    fn from_camera(camera: &Camera) -> Self {
        CameraUniform {
            view: camera.view().to_cols_array_2d(),
            projection: camera.projection().to_cols_array_2d(),
            view_projection: camera.view_projection().to_cols_array_2d(),
            position: camera.position().extend(1.0).to_array(),
        }
    }
}

struct RenderTargets {
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    scene_color: wgpu::TextureView,
    shaded_color: wgpu::TextureView,
}

impl RenderTargets {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let depth = create_target(
            device,
            "scene depth",
            width,
            height,
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );
        let depth_view = depth.create_view(&Default::default());
        let color_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let scene_color = create_target(device, "scene color", width, height, SCENE_COLOR_FORMAT, color_usage)
            .create_view(&Default::default());
        let shaded_color = create_target(device, "shaded color", width, height, SCENE_COLOR_FORMAT, color_usage)
            .create_view(&Default::default());
        RenderTargets { depth, depth_view, scene_color, shaded_color }
    }
}

pub struct Renderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,

    camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    targets: RenderTargets,

    sampler: wgpu::Sampler,
    present_layout: wgpu::BindGroupLayout,
    present_bind_group: wgpu::BindGroup,
    present_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,

    meshes: MeshRenderer,
    fluids: FluidRenderer,
}

impl Renderer {
    pub fn new(gpu: &gpu::Gpu, surface: wgpu::Surface<'static>, window: Arc<Window>, mut camera: Camera) -> anyhow::Result<Self> {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();

        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));
        let capabilities = surface.get_capabilities(&gpu.adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .or(capabilities.formats.first().copied())
            .ok_or_else(|| anyhow::anyhow!("surface reports no supported formats"))?;
        let mut config = surface
            .get_default_config(&gpu.adapter, width, height)
            .ok_or_else(|| anyhow::anyhow!("surface is not supported by the adapter"))?;
        config.format = format;
        surface.configure(&device, &config);

        camera.resize(width, height);

        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera uniform"),
            contents: bytemuck::bytes_of(&CameraUniform::from_camera(&camera)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let targets = RenderTargets::new(&device, width, height);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("present sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let present_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let present_bind_group = create_present_bind_group(&device, &present_layout, &targets.shaded_color, &sampler);
        let present_pipeline = create_present_pipeline(&device, &present_layout, config.format);
        let sky_pipeline = create_sky_pipeline(&device, &camera_layout);

        let meshes = MeshRenderer::new(&device, SCENE_COLOR_FORMAT, &camera_layout);
        let fluids = FluidRenderer::new(&device, &camera_layout, width, height, &targets.scene_color, &targets.depth_view);

        Ok(Renderer {
            window,
            surface,
            config,
            device,
            queue,
            camera,
            camera_buffer,
            camera_bind_group,
            targets,
            sampler,
            present_layout,
            present_bind_group,
            present_pipeline,
            sky_pipeline,
            meshes,
            fluids,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.camera.resize(width, height);

        self.targets = RenderTargets::new(&self.device, width, height);
        self.present_bind_group = create_present_bind_group(&self.device, &self.present_layout, &self.targets.shaded_color, &self.sampler);
        self.fluids.resize(&self.device, width, height, &self.targets.scene_color, &self.targets.depth_view);
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    pub fn create_mesh(&self, vertices: &[Vertex], indices: &[u32], topology: Topology, model: Mat4, material: MeshMaterial) -> Mesh {
        self.meshes.create_mesh(&self.device, vertices, indices, topology, model, material)
    }

    pub fn model_layout(&self) -> &wgpu::BindGroupLayout {
        self.meshes.model_layout()
    }

    pub fn create_fluid(&self, particles: &wgpu::Buffer, count: u32, material: FluidMaterial) -> FluidInstance {
        self.fluids.create_instance(&self.device, particles, count, material)
    }

    pub fn render(&mut self, meshes: &[&Mesh], fluid: &FluidInstance, shaded: bool) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };
        let frame_view = frame.texture.create_view(&Default::default());

        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&CameraUniform::from_camera(&self.camera)));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.scene_color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.draw(0..3, 0..1);

            self.meshes.draw(&mut pass, &self.camera_bind_group, meshes);

            if !shaded {
                self.fluids.draw_speed_spheres(&mut pass, &self.camera_bind_group, fluid);
            }
        }

        self.fluids.shade(
            &mut encoder,
            &self.camera_bind_group,
            fluid,
            &self.targets.depth,
            &self.targets.depth_view,
            &self.targets.shaded_color,
            shaded,
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.present_pipeline);
            pass.set_bind_group(0, &self.present_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.queue.present(frame);
    }
}

fn create_present_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    color: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("present"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(color),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn create_present_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let module = shader_module(device, "present shader", include_str!("../shaders/present.wgsl"));
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("present pipeline layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    fullscreen_pipeline(device, &pipeline_layout, &module, "vs_main", "fs_main", surface_format)
}

fn create_sky_pipeline(device: &wgpu::Device, camera_layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let module = shader_module(device, "sky shader", include_str!("../shaders/sky.wgsl"));
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sky pipeline layout"),
        bind_group_layouts: &[Some(camera_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sky"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_COLOR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
