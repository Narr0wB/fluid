use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use bytemuck::{self, Zeroable, Pod};
use wgpu;
use tobj;
use mesh_to_sdf;
use half;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
}

impl Vertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        todo!()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    Triangles,
    Lines,
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
    texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    collider_uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    dims: [u32; 3],
    min_bound: Vec3,
    max_bound: Vec3,
    cell_size: f32
}

pub struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    count: usize,
    topology: Topology,
    model_uniform: wgpu::Buffer,
    model_bind_group: wgpu::BindGroup,
    sdf: Sdf,
}


impl Mesh {
    pub fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model_layout: &wgpu::BindGroupLayout,
        sdf_layout: &wgpu::BindGroupLayout,
        path: &str,
    ) -> Result<Mesh, tobj::LoadError> {
        let opts = tobj::LoadOptions {
            triangulate: true,
            single_index: false,
            ..Default::default()
        };

        let (models, _mats) = tobj::load_obj(path, &opts)?;
        let mut vbuffer = Vec::new();
        let mut ibuffer = Vec::new();

        for m in models {
            // Indices refer to vertices, and each vertex is 3 floats.
            let base = (vbuffer.len() / 3) as u32;
            vbuffer.extend(m.mesh.positions.into_iter());
            ibuffer.extend(m.mesh.indices.into_iter().map(|i| i + base));
        }

        // draw_indexed takes the number of indices.
        let count = ibuffer.len();

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertices"),
            size: (vbuffer.len() * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indices"),
            size: (ibuffer.len() * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vbuffer));
        queue.write_buffer(&index_buffer, 0, bytemuck::cast_slice(&ibuffer));

        let (model_uniform, model_bind_group) = Self::create_model_binding(device, model_layout, Mat4::IDENTITY);
        let sdf = Self::bake_sdf(device, queue, sdf_layout, vbuffer.as_slice(), ibuffer.as_slice(), 0.01, 10);

        Ok(Mesh {
            vertex_buffer,
            index_buffer,
            count,
            topology: Topology::Triangles,
            model_uniform,
            model_bind_group,
            sdf
        })
    }

    pub fn bake_sdf(device: &wgpu::Device, queue: &wgpu::Queue, layout: &wgpu::BindGroupLayout, vertex_buffer: &[f32], index_buffer: &[u32], cell_size: f32, padding_cells: u32) -> Sdf {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in vertex_buffer.chunks_exact(3).map(Vec3::from_slice) {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let padding = cell_size * padding_cells as f32;
        lo -= Vec3::splat(padding);
        hi += Vec3::splat(padding);

        let n = ((hi - lo) / cell_size).ceil().as_usizevec3().max(glam::USizeVec3::splat(2));
        hi = lo + n.as_vec3() * cell_size;
        let dims = n.to_array();

        let grid = mesh_to_sdf::Grid::from_bounding_box(&lo.to_array(), &hi.to_array(), dims);

        let (vertices, _) = vertex_buffer.as_chunks::<3>();
        let topology = mesh_to_sdf::Topology::TriangleList(Some(index_buffer));
        let sdf = mesh_to_sdf::generate_grid_sdf(vertices, topology, &grid, mesh_to_sdf::SignMethod::Raycast);

        let mut reordered = Vec::with_capacity(n.x * n.y * n.z);
        for z in 0..n.z {
            for y in 0..n.y {
                for x in 0..n.x {
                    reordered.push(half::f16::from_f32(sdf[grid.get_cell_idx(&[x, y, z])]));
                }
            }
        }

        let dims = n.as_uvec3().to_array();
        let extent = wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] };
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
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&reordered),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // One f16 per texel.
                bytes_per_row: Some(dims[0] * size_of::<half::f16>() as u32),
                rows_per_image: Some(dims[1]),
            },
            extent,
        );

        // Trilinear filtering, clamped so samples outside the grid read the padded border.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sdf sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Identity transform until set_model moves the collider.
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

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

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

        Sdf { texture, sampler, collider_uniform, bind_group, dims, min_bound: lo, max_bound: hi, cell_size }
    }

    fn create_model_binding(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, model: Mat4) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("model uniform"),
            contents: bytemuck::bytes_of(&model.to_cols_array_2d()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model bind group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        (buffer, bind_group)
    }

    pub fn set_model(&self, queue: &wgpu::Queue, model: Mat4) {
        todo!()
    }
}

pub struct MeshRenderer {
    triangles: wgpu::RenderPipeline,
    lines: wgpu::RenderPipeline,
    model_layout: wgpu::BindGroupLayout,
}

impl MeshRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat, camera_layout: &wgpu::BindGroupLayout) -> Self {
        todo!()
    }

    pub fn create_mesh(
        &self,
        device: &wgpu::Device,
        vertices: &[Vertex],
        indices: &[u32],
        topology: Topology,
        model: Mat4,
    ) -> Mesh {
        todo!()
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, camera: &wgpu::BindGroup, meshes: &[&Mesh]) {
        todo!()
    }
}

pub fn wireframe_box(min: Vec3, max: Vec3) -> (Vec<Vertex>, Vec<u32>) {
    todo!()
}
