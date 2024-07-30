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
