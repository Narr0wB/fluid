#version 460
    
#extension GL_EXT_shader_atomic_float: enable

layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

struct Particle {
    vec4 position;
    vec4 velocity;
};

layout(std140, binding = 0) buffer Particles {
    Particle particles[];
} p;

const vec3 min_grid = vec3(-1.0, -1.0, -1.0);

uint hash_func(uint idx_x, uint idx_y, uint idx_z) {
    return idx_x + idx_y * 5741 + idx_z * 6547;
}
