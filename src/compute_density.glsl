#version 460
    
#extension GL_EXT_shader_atomic_float: enable
#extension GL_EXT_debug_printf: enable
#define M_PI 3.1415926535897932384626433832795

layout(local_size_x=64, local_size_y=1, local_size_z=1) in;

struct Particle {
    vec3 position;
    float density;

    vec3 velocity;
    float pressure;
};

layout(std430, binding = 0) buffer Particles {
    Particle particles[];
} p;

layout(push_constant) uniform Constants {
    uint size;
    float particle_mass;
    float smoothing_radius;
} c;

void main() {
    uint stride = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
    float h = c.smoothing_radius;
    float h2 = h*h;
    float poly6 = 315.0 / (64.0 * M_PI * pow(h, 9.0));

    for (uint i = gl_GlobalInvocationID.x; i < c.size; i += stride) {
        vec3 xi = p.particles[i].position;
        float rho = 0.0;
        
        for (uint j = 0; j < c.size; ++j) {
            vec3 xj = p.particles[j].position;
            float r2 = dot(xi - xj, xi - xj);
            if (r2 > h2) continue;

            float diff = h2 - r2;
            rho += c.particle_mass * poly6 * diff * diff * diff; 
        }
        
        p.particles[i].density = rho;
    }
}