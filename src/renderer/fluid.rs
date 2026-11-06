use glam::Vec3;
use wgpu::util::DeviceExt;

use super::{DEPTH_FORMAT, SCENE_COLOR_FORMAT, create_target, fullscreen_pipeline, shader_module};

const LINEAR_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const THICKNESS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const DEPTH_BLUR_ITERATIONS: usize = 3;

pub struct FluidMaterial {
    pub color: Vec3,
    pub particle_radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FluidParams {
    color: [f32; 3],
    particle_radius: f32,
}

pub struct FluidInstance {
    count: u32,
    bind_group: wgpu::BindGroup,
}

struct BlurChain {
    front: wgpu::TextureView,
    front_bind_group: wgpu::BindGroup,
    back: wgpu::TextureView,
    back_bind_group: wgpu::BindGroup,
}

impl BlurChain {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let front = create_target(device, label, width, height, format, usage).create_view(&Default::default());
        let back = create_target(device, label, width, height, format, usage).create_view(&Default::default());
        let bind = |view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                }],
            })
        };
        let front_bind_group = bind(&front);
        let back_bind_group = bind(&back);
        BlurChain { front, front_bind_group, back, back_bind_group }
    }
}

struct FluidTargets {
    depth_buffer: wgpu::Texture,
    depth_buffer_view: wgpu::TextureView,
    linear_depth: BlurChain,
    thickness: BlurChain,
    composite_bind_group: wgpu::BindGroup,
}

pub struct FluidRenderer {
    fluid_layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,

    sphere_pipeline: wgpu::RenderPipeline,
    depth_pipeline: wgpu::RenderPipeline,
    thickness_pipeline: wgpu::RenderPipeline,
    blur_depth_x: wgpu::RenderPipeline,
    blur_depth_y: wgpu::RenderPipeline,
    blur_thickness_x: wgpu::RenderPipeline,
    blur_thickness_y: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,

    targets: FluidTargets,
}

impl FluidRenderer {
    pub fn new(
        device: &wgpu::Device,
        camera_layout: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
        scene_color: &wgpu::TextureView,
        scene_depth: &wgpu::TextureView,
    ) -> Self {
        let module = shader_module(device, "fluid shader", include_str!("../shaders/fluid_render.wgsl"));

        let fluid_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let unfilterable = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let filterable = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let fragment_entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        };

        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid blur layout"),
            entries: &[fragment_entry(0, unfilterable)],
        });

        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid composite layout"),
            entries: &[
                fragment_entry(1, unfilterable),
                fragment_entry(2, filterable),
                fragment_entry(3, filterable),
                fragment_entry(
                    4,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                fragment_entry(5, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fluid sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sphere_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fluid sphere pipeline layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&fluid_layout)],
            immediate_size: 0,
        });
        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fluid blur pipeline layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&fluid_layout), Some(&blur_layout)],
            immediate_size: 0,
        });
        let composite_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fluid composite pipeline layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&fluid_layout), Some(&composite_layout)],
            immediate_size: 0,
        });

        let sphere = |label: &str, entry: &str, format, blend, depth_write| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&sphere_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_sphere"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };

        let sphere_pipeline = sphere("fluid speed spheres", "fs_sphere_speed", SCENE_COLOR_FORMAT, None, true);
        let depth_pipeline = sphere("fluid depth", "fs_sphere_depth", LINEAR_DEPTH_FORMAT, None, true);
        let thickness_pipeline = sphere("fluid thickness", "fs_thickness", THICKNESS_FORMAT, Some(additive), false);

        let vertex = "vs_fullscreen";
        let blur_depth_x = fullscreen_pipeline(device, &blur_pipeline_layout, &module, vertex, "fs_blur_depth_x", LINEAR_DEPTH_FORMAT);
        let blur_depth_y = fullscreen_pipeline(device, &blur_pipeline_layout, &module, vertex, "fs_blur_depth_y", LINEAR_DEPTH_FORMAT);
        let blur_thickness_x = fullscreen_pipeline(device, &blur_pipeline_layout, &module, vertex, "fs_blur_thickness_x", THICKNESS_FORMAT);
        let blur_thickness_y = fullscreen_pipeline(device, &blur_pipeline_layout, &module, vertex, "fs_blur_thickness_y", THICKNESS_FORMAT);
        let composite_pipeline = fullscreen_pipeline(device, &composite_pipeline_layout, &module, vertex, "fs_composite", SCENE_COLOR_FORMAT);

        let targets = create_targets(device, &blur_layout, &composite_layout, &sampler, width, height, scene_color, scene_depth);

        FluidRenderer {
            fluid_layout,
            blur_layout,
            composite_layout,
            sampler,
            sphere_pipeline,
            depth_pipeline,
            thickness_pipeline,
            blur_depth_x,
            blur_depth_y,
            blur_thickness_x,
            blur_thickness_y,
            composite_pipeline,
            targets,
        }
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        scene_color: &wgpu::TextureView,
        scene_depth: &wgpu::TextureView,
    ) {
        self.targets = create_targets(
            device,
            &self.blur_layout,
            &self.composite_layout,
            &self.sampler,
            width,
            height,
            scene_color,
            scene_depth,
        );
    }

    pub fn create_instance(
        &self,
        device: &wgpu::Device,
        particles: &wgpu::Buffer,
        count: u32,
        material: FluidMaterial,
    ) -> FluidInstance {
        let params = FluidParams {
            color: material.color.to_array(),
            particle_radius: material.particle_radius,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fluid material"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid instance"),
            layout: &self.fluid_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles.as_entire_binding(),
                },
            ],
        });
        FluidInstance { count, bind_group }
    }

    pub fn draw_speed_spheres(&self, pass: &mut wgpu::RenderPass<'_>, camera: &wgpu::BindGroup, fluid: &FluidInstance) {
        pass.set_pipeline(&self.sphere_pipeline);
        pass.set_bind_group(0, camera, &[]);
        pass.set_bind_group(1, &fluid.bind_group, &[]);
        pass.draw(0..6, 0..fluid.count);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn shade(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera: &wgpu::BindGroup,
        fluid: &FluidInstance,
        scene_depth: &wgpu::Texture,
        scene_depth_view: &wgpu::TextureView,
        output: &wgpu::TextureView,
        draw_fluid: bool,
    ) {
        let targets = &self.targets;

        {
            let mut pass = begin_color_pass(encoder, "fluid thickness", &targets.thickness.front, wgpu::Color::TRANSPARENT, Some(wgpu::RenderPassDepthStencilAttachment {
                view: scene_depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }));
            if draw_fluid {
                pass.set_pipeline(&self.thickness_pipeline);
                pass.set_bind_group(0, camera, &[]);
                pass.set_bind_group(1, &fluid.bind_group, &[]);
                pass.draw(0..6, 0..fluid.count);
            }
        }

        encoder.copy_texture_to_texture(
            scene_depth.as_image_copy(),
            targets.depth_buffer.as_image_copy(),
            scene_depth.size(),
        );

        {
            let mut pass = begin_color_pass(encoder, "fluid depth", &targets.linear_depth.front, wgpu::Color::TRANSPARENT, Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth_buffer_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }));
            if draw_fluid {
                pass.set_pipeline(&self.depth_pipeline);
                pass.set_bind_group(0, camera, &[]);
                pass.set_bind_group(1, &fluid.bind_group, &[]);
                pass.draw(0..6, 0..fluid.count);
            }
        }

        if draw_fluid {
            for _ in 0..DEPTH_BLUR_ITERATIONS {
                self.blur(encoder, camera, fluid, &targets.linear_depth, &self.blur_depth_x, &self.blur_depth_y);
            }
            self.blur(encoder, camera, fluid, &targets.thickness, &self.blur_thickness_x, &self.blur_thickness_y);
        }

        let mut pass = begin_color_pass(encoder, "fluid composite", output, wgpu::Color::BLACK, None);
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, camera, &[]);
        pass.set_bind_group(1, &fluid.bind_group, &[]);
        pass.set_bind_group(2, &targets.composite_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn blur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera: &wgpu::BindGroup,
        fluid: &FluidInstance,
        chain: &BlurChain,
        horizontal: &wgpu::RenderPipeline,
        vertical: &wgpu::RenderPipeline,
    ) {
        let steps = [
            (horizontal, &chain.back, &chain.front_bind_group),
            (vertical, &chain.front, &chain.back_bind_group),
        ];
        for (pipeline, target, source) in steps {
            let mut pass = begin_color_pass(encoder, "fluid blur", target, wgpu::Color::TRANSPARENT, None);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, &fluid.bind_group, &[]);
            pass.set_bind_group(2, source, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn begin_color_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    view: &wgpu::TextureView,
    clear: wgpu::Color,
    depth: Option<wgpu::RenderPassDepthStencilAttachment<'_>>,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: depth,
        ..Default::default()
    })
}

#[allow(clippy::too_many_arguments)]
fn create_targets(
    device: &wgpu::Device,
    blur_layout: &wgpu::BindGroupLayout,
    composite_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    width: u32,
    height: u32,
    scene_color: &wgpu::TextureView,
    scene_depth: &wgpu::TextureView,
) -> FluidTargets {
    let depth_buffer = create_target(
        device,
        "fluid depth buffer",
        width,
        height,
        DEPTH_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
    );
    let depth_buffer_view = depth_buffer.create_view(&Default::default());
    let linear_depth = BlurChain::new(device, blur_layout, width, height, LINEAR_DEPTH_FORMAT, "fluid linear depth");
    let thickness = BlurChain::new(device, blur_layout, width, height, THICKNESS_FORMAT, "fluid thickness");

    let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fluid composite"),
        layout: composite_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&linear_depth.front) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&thickness.front) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(scene_color) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(scene_depth) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    });

    FluidTargets { depth_buffer, depth_buffer_view, linear_depth, thickness, composite_bind_group }
}
