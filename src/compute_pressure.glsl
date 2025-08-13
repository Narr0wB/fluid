#version 450
#define M_PI 3.1415926535897932384626433832795
#define DISTANCE_EPSILON 1e-5
#define DENSITY_EPSILON 1e-5

layout(local_size_x=64, local_size_y=1, local_size_z=1) in;

struct Particle {
    vec3 position;
    float density;

    vec3 velocity;
    float pressure;
};

layout(std430, binding = 0) readonly buffer Particles {
    Particle particles[];
} p;

layout(std430, binding = 1) writeonly buffer Forces {
    vec4 forces[];
} f;

layout(push_constant) uniform Constants {
    uint size;
    float particle_mass;
    float smoothing_radius;
    float target_density;
    float pressure_constant;
    float viscosity_constant;
} c;

// float density_to_pressure(float density) {
//     if (density < c.target_density) return 0.0;
//     return c.pressure_constant * (density - c.target_density);
// }

// Tait EOS
float density_to_pressure(float rho) {
    float rho0 = c.target_density;
    float gamma = 7.0;
    float B = c.pressure_constant;
    float ratio = max(rho, 0.5*rho0) / rho0;
    return B * (pow(ratio, gamma) - 1.0);
}


const vec3 gravity = vec3(0.0, -9.81, 0.0);

void main() {
    uint stride = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
    float h = c.smoothing_radius; 
    float smoothing_kernel_scale = -45.0 / (M_PI * pow(h, 6));
    float laplacian_kernel_scale = 45.0 / (M_PI * pow(h, 6));

    for (uint i = gl_GlobalInvocationID.x; i < c.size; i += stride) {
        vec3 total_force = vec3(0.0);
        float rho_i = max(p.particles[i].density, DENSITY_EPSILON);
        float pressure_i = density_to_pressure(rho_i);
        
        for (uint j = 0; j < c.size; j++) {
            if (i == j) continue;
            
            float rho_j = max(p.particles[j].density, DENSITY_EPSILON);
            vec3 dir = p.particles[i].position - p.particles[j].position;
            float dist = length(dir);
            
            if (dist < DISTANCE_EPSILON || dist > c.smoothing_radius) continue;
            
            vec3 norm_dir = dir / max(dist, DISTANCE_EPSILON);
            float pressure_j = density_to_pressure(rho_j);
            
            vec3 pressure_force = -norm_dir * c.particle_mass * 
                                (pressure_i/(rho_i*rho_i) + pressure_j/(rho_j*rho_j)) *
                                smoothing_kernel_scale * (h - dist) * (h - dist);
            
            vec3 viscosity_force = c.viscosity_constant * c.particle_mass *
                                (p.particles[j].velocity - p.particles[i].velocity) / rho_j *
                                laplacian_kernel_scale * (h - dist);
            
            total_force += pressure_force + viscosity_force;
        }
        total_force += gravity;
        f.forces[i].xyz = total_force;
    }
}