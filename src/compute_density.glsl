#version 460
    
#extension GL_EXT_shader_atomic_float: enable
#define M_PI 3.1415926535897932384626433832795

layout(local_size_x=64, local_size_y=1, local_size_z=1) in;

struct Particle {
    vec3 position;
    float density;

    vec3 velocity;
    float pressure;
};

struct BoundingBox {
    float x1;
    float x2;
    float z1;
    float z2;
    float y1;
    float y2;

    float damping_factor;
};

layout(std140, binding = 0) buffer Particles {
    Particle particles[];
} p;

layout(std140, push_constant) uniform Constants {
    uint size;
    float particle_mass;
    float smoothing_radius;
} c;

float smoothing_kernel(float dist, float radius) {
    if (dist > radius) return 0.0;
    
    float h2 = radius * radius;
    float r2 = dist * dist;
    float diff = h2 - r2;
    float volume = 315.0 / (64.0 * M_PI * pow(radius, 9));
    return volume * diff * diff * diff;  
}

void main() {
    for (uint i = gl_GlobalInvocationID.x; i < c.size; i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
        Particle pi = p.particles[i];
        
        if (gl_LocalInvocationID.x == 0) {
            p.particles[i].density = 0.0;
        }
        
        barrier();

        for (uint j = 0; j < c.size; ++j) {
            Particle pj = p.particles[j];

            float dist = length(pi.position.xyz - pj.position.xyz);
            if (dist > c.smoothing_radius) continue;

            float contribution = c.particle_mass * smoothing_kernel(dist, c.smoothing_radius);
            p.particles[i].density += contribution; 
        }
    }
}
