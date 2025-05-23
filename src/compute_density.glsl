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




// spiky-6 smoothing kernel
float smoothing_kernel(float dist, float radius) {
    if (dist > radius) {
        return 0.0;
    }
    
    float volume = 15.0 / (M_PI * pow(radius, 6));
    return pow((radius - dist), 3) * volume;
}




void main() {
    for (uint i = gl_WorkGroupID.x; i < c.size; i += gl_NumWorkGroups.x) {
        Particle pi = p.particles[i];
        
        if (gl_LocalInvocationID.x == 0) {
            p.particles[i].density = 0.0;
        }
        
        barrier();

        for (uint j = gl_LocalInvocationID.x; j < c.size; j += gl_WorkGroupSize.x) {
            Particle pj = p.particles[j];

            float dist = length(pi.position.xyz - pj.position.xyz);
            if (dist >= c.smoothing_radius) {continue;}

            float contribution = c.particle_mass * smoothing_kernel(dist, c.smoothing_radius);
            atomicAdd(p.particles[i].density, contribution); 
        }
    }
}
