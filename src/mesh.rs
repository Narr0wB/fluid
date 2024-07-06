use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::pipeline::graphics::vertex_input::Vertex as VertexTrait;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use nalgebra::{*};


#[derive(BufferContents, VertexTrait, Default, Clone, Copy)]
#[repr(C)]
pub struct MVertex {
    #[format(R32G32B32_SFLOAT)] 
    pub position: [f32; 3]
}


pub struct Mesh {
    vertex_buffer: Subbuffer<[MVertex]>,
    model_matrix: Matrix4<f32>,
    index_buffer: Subbuffer<[u32]>,
}

impl Mesh {
    pub fn new(mem_allocator: Arc<StandardMemoryAllocator>, vertices: Vec<MVertex>, indices: Vec<u32>, model_matrix: Matrix4<f32>) -> Self {
        
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
    
}
