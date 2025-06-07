#version 450  // Downgrade to 450 for better compatibility
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

layout(std140, binding = 1) buffer TranslationBuffers {
    mat4 translations[];
} t;

layout(std140, push_constant) uniform Constants {
    uint size;
    float particle_mass;
    float smoothing_radius;
    float target_density;
    float pressure_constant;
    float viscosity_constant;
    float dt;
    BoundingBox bounds;
} c;

// Shared memory per work item
shared vec3 shared_pressure[64];  // Matches local_size_x

float smoothing_kernel(float dist, float radius) {
    if (dist > radius) return 0.0;
    float r2 = radius * radius;
    float d2 = dist * dist;
    float diff = r2 - d2;
    float volume = 315.0 / (64.0 * M_PI * pow(radius, 9));
    return diff * diff * diff * volume;  // Optimized pow(diff,3)
}

// viscosity kernel
float laplacian_kernel(float dist, float radius)  {
    if (dist > radius) return 0.0;
    float scale = 45.0 / (M_PI * pow(radius, 6));
    return scale * (radius - dist);
}

float smoothing_kernel_derivative(float dist, float radius) {
    if (dist > radius) return 0.0;
    float diff = radius - dist;
    float scale = -45.0 / (M_PI * pow(radius, 6));
    return scale * diff * diff * diff;  // Optimized pow(diff,3)
}

float density_to_pressure(float density) {
    return (density >= c.target_density) 
        ? c.pressure_constant * (density - c.target_density) 
        : 0.0;
}

mat4 translation(vec3 pos) {
    return mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        pos.x, pos.y, pos.z, 1.0
    );
}

const vec3 gravity = vec3(0.0, -9.81, 0.0);

const vec3 NORM_X1 = vec3(1, 0, 0);
const vec3 NORM_X2 = vec3(-1, 0, 0);
const vec3 NORM_Z1 = vec3(0, 0, 1);
const vec3 NORM_Z2 = vec3(0, 0, -1);
const vec3 NORM_Y1 = vec3(0, 1, 0);

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= c.size) return;

    Particle pi = p.particles[i];
    float di = pi.position.w;
    vec3 pressure_force = vec3(0.0);

    // Each thread calculates its own contribution
    for (uint j = 0; j < c.size; j++) {
        if (i == j) continue;
        
        Particle pj = p.particles[j];
        float dj = pj.position.w;
        vec3 delta = pi.position.xyz - pj.position.xyz;
        float dist = length(delta);
        
        if (dist >= c.smoothing_radius || dist < 0.001) continue;
        
        vec3 dir = delta / dist;  // Normalize
        float deriv = smoothing_kernel_derivative(dist, c.smoothing_radius);
        float pressure_j = density_to_pressure(dj);
        float pressure_i = density_to_pressure(di);
        
        // Pressure force
        vec3 contribution = -dir * c.particle_mass * 
                           ((pressure_j + pressure_i) / (2.0 * dj)) * 
                           deriv;
        
        // Viscosity force
        vec3 viscosity = c.viscosity_constant * 
                        ((pj.velocity.xyz - pi.velocity.xyz) / dj) * 
                        c.particle_mass * 
                        laplacian_kernel(dist, c.smoothing_radius);
        
        pressure_force += contribution + viscosity;
    }

    // Store in shared memory for reduction
    shared_pressure[gl_LocalInvocationID.x] = pressure_force;
    barrier();
    
    // Reduction in shared memory (first thread sums)
    if (gl_LocalInvocationID.x == 0) {
        vec3 total_pressure = vec3(0.0);
        for (uint idx = 0; idx < 64; idx++) {
            total_pressure += shared_pressure[idx];
        }
        
        // Physics integration
        vec3 accel = gravity + (total_pressure / di);
        if (any(isnan(accel))) accel = vec3(0.0);
        
        p.particles[i].velocity.xyz += accel * c.dt;
        vec3 new_position = pi.position.xyz + p.particles[i].velocity.xyz * c.dt;
        
        // Boundary collision
        float half_radius = c.smoothing_radius * 0.5;
        vec3 vel = p.particles[i].velocity.xyz;
        
        if (new_position.x - half_radius <= c.bounds.x1) {
            vel.x *= -1.0 * (1.0 - c.bounds.damping_factor);
            new_position.x = c.bounds.x1 + half_radius;
        }
        else if (new_position.x + half_radius >= c.bounds.x2) {
            vel.x *= -1.0 * (1.0 - c.bounds.damping_factor);
            new_position.x = c.bounds.x2 - half_radius;
        }
        
        if (new_position.z - half_radius <= c.bounds.z1) {
            vel.z *= -1.0 * (1.0 - c.bounds.damping_factor);
            new_position.z = c.bounds.z1 + half_radius;
        }
        else if (new_position.z + half_radius >= c.bounds.z2) {
            vel.z *= -1.0 * (1.0 - c.bounds.damping_factor);
            new_position.z = c.bounds.z2 - half_radius;
        }
        
        if (new_position.y - half_radius <= c.bounds.y1) {
            vel.y *= -1.0 * (1.0 - c.bounds.damping_factor);
            new_position.y = c.bounds.y1 + half_radius;
        }
        
        p.particles[i].velocity.xyz = vel;
        p.particles[i].position.xyz = new_position;
        t.translations[i] = translation(new_position);
        t.translations[i][0][1] = length(vel) * 0.5;  // Velocity magnitude
    }
}