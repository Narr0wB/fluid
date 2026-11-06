use std::path::Path;

use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use super::{DEPTH_FORMAT, shader_module};

const SDF_RESOLUTION: f32 = 64.0;
const SDF_PADDING_CELLS: u32 = 10;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl Vertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    Triangles,
    Lines,
}

#[derive(Clone, Copy)]
pub struct MeshMaterial {
    pub color: Vec3,
    pub checker: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ModelUniform {
    model: [[f32; 4]; 4],
    color: [f32; 3],
    checker: f32,
}

impl ModelUniform {
    fn new(model: Mat4, material: MeshMaterial) -> Self {
        ModelUniform {
            model: model.to_cols_array_2d(),
            color: material.color.to_array(),
            checker: if material.checker { 1.0 } else { 0.0 },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ColliderUniform {
    pub inv_model: [[f32; 4]; 4],
    pub bounds_min: [f32; 3],
    pub scale: f32,
    pub bounds_max: [f32; 3],
    pub cell_size: f32,
    pub velocity: [f32; 3],
    pub _pad: f32,
}

pub struct Sdf {
    collider_uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    min_bound: Vec3,
    max_bound: Vec3,
    cell_size: f32,
}

impl Sdf {
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    fn uniform(&self, model: Mat4, velocity: Vec3) -> ColliderUniform {
        ColliderUniform {
            inv_model: model.inverse().to_cols_array_2d(),
            bounds_min: self.min_bound.to_array(),
            scale: model.x_axis.truncate().length(),
            bounds_max: self.max_bound.to_array(),
            cell_size: self.cell_size,
            velocity: velocity.to_array(),
            _pad: 0.0,
        }
    }
}

pub struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    count: u32,
    topology: Topology,
    model_uniform: wgpu::Buffer,
    model_bind_group: wgpu::BindGroup,
    material: MeshMaterial,
    local_min: Vec3,
    local_max: Vec3,
    sdf: Option<Sdf>,
}

impl Mesh {
    pub fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model_layout: &wgpu::BindGroupLayout,
        sdf_layout: &wgpu::BindGroupLayout,
        path: &Path,
        material: MeshMaterial,
    ) -> anyhow::Result<Mesh> {
        let options = tobj::LoadOptions {
            triangulate: true,
            single_index: false,
            ignore_points: true,
            ignore_lines: true,
        };

        let (models, _) = tobj::load_obj(path, &options).with_context(|| format!("loading {}", path.display()))?;
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for model in models {
            let base = positions.len() as u32;
            positions.extend_from_slice(model.mesh.positions.as_chunks::<3>().0);
            indices.extend(model.mesh.indices.into_iter().map(|i| i + base));
        }

        anyhow::ensure!(!indices.is_empty(), "{} contains no triangles", path.display());

        let vertices = with_smooth_normals(&positions, &indices);
        let mut mesh = create_mesh(device, model_layout, &vertices, &indices, Topology::Triangles, Mat4::IDENTITY, material);
        mesh.sdf = Some(bake_sdf(device, queue, sdf_layout, &positions, &indices));
        Ok(mesh)
    }

    pub fn local_bounds(&self) -> (Vec3, Vec3) {
        (self.local_min, self.local_max)
    }

    pub fn sdf(&self) -> Option<&Sdf> {
        self.sdf.as_ref()
    }

    pub fn set_model(&self, queue: &wgpu::Queue, model: Mat4, velocity: Vec3) {
        queue.write_buffer(&self.model_uniform, 0, bytemuck::bytes_of(&ModelUniform::new(model, self.material)));
        if let Some(sdf) = &self.sdf {
            queue.write_buffer(&sdf.collider_uniform, 0, bytemuck::bytes_of(&sdf.uniform(model, velocity)));
        }
    }
}

fn with_smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<Vertex> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for triangle in indices.as_chunks::<3>().0 {
        let [a, b, c] = triangle.map(|index| index as usize);
        let (pa, pb, pc) = (Vec3::from(positions[a]), Vec3::from(positions[b]), Vec3::from(positions[c]));
        let area_weighted = (pb - pa).cross(pc - pa);
        normals[a] += area_weighted;
        normals[b] += area_weighted;
        normals[c] += area_weighted;
    }
    positions
        .iter()
        .zip(normals)
        .map(|(position, normal)| Vertex {
            position: *position,
            normal: normal.normalize_or_zero().to_array(),
        })
        .collect()
}

fn bounds_of<'a>(positions: impl Iterator<Item = &'a [f32; 3]>) -> (Vec3, Vec3) {
    positions.fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), p| {
        let p = Vec3::from(*p);
        (lo.min(p), hi.max(p))
    })
}

fn create_mesh(
    device: &wgpu::Device,
    model_layout: &wgpu::BindGroupLayout,
    vertices: &[Vertex],
    indices: &[u32],
    topology: Topology,
    model: Mat4,
    material: MeshMaterial,
) -> Mesh {
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vertices"),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("indices"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let model_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("model uniform"),
        contents: bytemuck::bytes_of(&ModelUniform::new(model, material)),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("model bind group"),
        layout: model_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: model_uniform.as_entire_binding(),
        }],
    });
    let (local_min, local_max) = bounds_of(vertices.iter().map(|v| &v.position));

    Mesh {
        vertex_buffer,
        index_buffer,
        count: indices.len() as u32,
        topology,
        model_uniform,
        model_bind_group,
        material,
        local_min,
        local_max,
        sdf: None,
    }
}

fn bake_sdf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Sdf {
    let (mut lo, mut hi) = bounds_of(positions.iter());
    let cell_size = (hi - lo).max_element() / SDF_RESOLUTION;
    let padding = cell_size * SDF_PADDING_CELLS as f32;
    lo -= Vec3::splat(padding);
    hi += Vec3::splat(padding);

    let n = ((hi - lo) / cell_size).ceil().as_usizevec3().max(glam::USizeVec3::splat(2));
    hi = lo + n.as_vec3() * cell_size;

    let grid = mesh_to_sdf::Grid::from_bounding_box(&lo.to_array(), &hi.to_array(), n.to_array());
    let topology = mesh_to_sdf::Topology::TriangleList(Some(indices));
    let distances = mesh_to_sdf::generate_grid_sdf(positions, topology, &grid, mesh_to_sdf::SignMethod::Raycast);

    let mut texels = Vec::with_capacity(n.x * n.y * n.z);
    for z in 0..n.z {
        for y in 0..n.y {
            for x in 0..n.x {
                texels.push(half::f16::from_f32(distances[grid.get_cell_idx(&[x, y, z])]));
            }
        }
    }

    let dims = n.as_uvec3();
    let extent = wgpu::Extent3d { width: dims.x, height: dims.y, depth_or_array_layers: dims.z };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sdf"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(dims.x * size_of::<half::f16>() as u32),
            rows_per_image: Some(dims.y),
        },
        extent,
    );

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("sdf sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let collider = ColliderUniform {
        inv_model: Mat4::IDENTITY.to_cols_array_2d(),
        bounds_min: lo.to_array(),
        scale: 1.0,
        bounds_max: hi.to_array(),
        cell_size,
        velocity: [0.0; 3],
        _pad: 0.0,
    };
    let collider_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("collider uniform"),
        contents: bytemuck::bytes_of(&collider),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let view = texture.create_view(&Default::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sdf bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: collider_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    Sdf { collider_uniform, bind_group, min_bound: lo, max_bound: hi, cell_size }
}

pub struct MeshRenderer {
    triangles: wgpu::RenderPipeline,
    lines: wgpu::RenderPipeline,
    model_layout: wgpu::BindGroupLayout,
}

impl MeshRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat, camera_layout: &wgpu::BindGroupLayout) -> Self {
        let model_layout = create_model_layout(device);
        let module = shader_module(device, "mesh shader", include_str!("../shaders/mesh.wgsl"));
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh pipeline layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&model_layout)],
            immediate_size: 0,
        });

        let pipeline = |label: &str, entry: &str, topology| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(Vertex::layout())],
                },
                primitive: wgpu::PrimitiveState { topology, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
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
                        format: color_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        MeshRenderer {
            triangles: pipeline("mesh triangles", "fs_lit", wgpu::PrimitiveTopology::TriangleList),
            lines: pipeline("mesh lines", "fs_line", wgpu::PrimitiveTopology::LineList),
            model_layout,
        }
    }

    pub fn model_layout(&self) -> &wgpu::BindGroupLayout {
        &self.model_layout
    }

    pub fn create_mesh(
        &self,
        device: &wgpu::Device,
        vertices: &[Vertex],
        indices: &[u32],
        topology: Topology,
        model: Mat4,
        material: MeshMaterial,
    ) -> Mesh {
        create_mesh(device, &self.model_layout, vertices, indices, topology, model, material)
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, camera: &wgpu::BindGroup, meshes: &[&Mesh]) {
        pass.set_bind_group(0, camera, &[]);
        for mesh in meshes {
            pass.set_pipeline(match mesh.topology {
                Topology::Triangles => &self.triangles,
                Topology::Lines => &self.lines,
            });
            pass.set_bind_group(1, &mesh.model_bind_group, &[]);
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.count, 0, 0..1);
        }
    }
}

pub(crate) fn create_model_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("model layout"),
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
    })
}

pub fn wireframe_box(min: Vec3, max: Vec3) -> (Vec<Vertex>, Vec<u32>) {
    let vertices = (0..8)
        .map(|i| Vertex {
            position: [
                if i & 1 == 0 { min.x } else { max.x },
                if i & 2 == 0 { min.y } else { max.y },
                if i & 4 == 0 { min.z } else { max.z },
            ],
            normal: [0.0; 3],
        })
        .collect();
    let indices = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 1, 3, 4, 6, 5, 7, 0, 4, 1, 5, 2, 6, 3, 7];
    (vertices, indices)
}

pub fn floor_quad(min: Vec3, max: Vec3) -> (Vec<Vertex>, Vec<u32>) {
    let normal = [0.0, 1.0, 0.0];
    let vertices = vec![
        Vertex { position: [min.x, min.y, min.z], normal },
        Vertex { position: [max.x, min.y, min.z], normal },
        Vertex { position: [max.x, min.y, max.z], normal },
        Vertex { position: [min.x, min.y, max.z], normal },
    ];
    (vertices, vec![0, 2, 1, 0, 3, 2])
}
