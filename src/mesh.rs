use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::device::Device;
use vulkano::pipeline::graphics::vertex_input::Vertex as VertexTrait;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use nalgebra::{*};

#[repr(C)]
#[derive(BufferContents, VertexTrait, Default, Clone, Copy)]
pub struct MVertex {
    #[format(R32G32B32_SFLOAT)] 
    pub position: [f32; 3]
}

impl From<Vector3<f32>> for MVertex {
    fn from(vec: Vector3<f32>) -> Self {
        MVertex {position: [vec.x, vec.y, vec.z]}
    }
}

pub struct Mesh {
    vertex_buffer: Subbuffer<[MVertex]>, 
    index_buffer: Subbuffer<[u32]>,
    model_matrix: Matrix4<f32>,
}

impl Mesh {
    pub fn new(vertices: Vec<MVertex>, indices: Vec<u32>, model_matrix: Matrix4<f32>, device: Arc<Device>) -> Self {

        let mem_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        
        let vertex_buffer = Buffer::from_iter(
            mem_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices.into_iter()
        ).unwrap();
        
        let index_buffer = Buffer::from_iter(
                mem_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                indices.into_iter()
            ).unwrap();
        
       Mesh {vertex_buffer, model_matrix, index_buffer}
    }
    
    pub fn get_vertex_buffer(&self) -> &Subbuffer<[MVertex]> {
        &self.vertex_buffer
    }

    pub fn get_index_buffer(&self) -> &Subbuffer<[u32]> {
        &self.index_buffer
    }

    pub fn get_model_matrix(&self) -> &Matrix4<f32> {
        &self.model_matrix
    }

    pub fn len(&self) -> u64 {
        self.index_buffer.len()
    }
    
}

pub fn create_sphere(radius: f32, center: Vector3<f32>, resolution: u32, device: Arc<Device>) -> Mesh {
    let mut vertices = vec![];
    let mut indices  = vec![];

    for i in 0..resolution {
        let theta = (i as f32 / (resolution-1) as f32) * std::f32::consts::PI;
        
        for j in 0..resolution {
            let phi = (j as f32 / (resolution-1) as f32) * 2.0 * std::f32::consts::PI;

            indices.push(i * resolution + j);
            indices.push(i * resolution + j + 1);
            indices.push((i+1) * resolution + j);
            indices.push(i * resolution + j + 1);
            indices.push((i+1) * resolution + j);
            indices.push((i+1) * resolution + j + 1);

            vertices.push(Vector3::<f32>::new(center.x + radius * theta.sin() * phi.cos(), center.y + radius * theta.sin() * phi.sin(), center.z + radius * theta.cos()));
        }
    }

    let mvertices: Vec<MVertex> = vertices
        .into_iter()
        .map(MVertex::from)
        .collect(); 

    Mesh::new(mvertices, indices, Matrix4::identity(), device)
}
