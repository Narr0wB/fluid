#version 450
#extension GL_EXT_shader_atomic_float : enable
#define M_PI 3.1415926535897932384626433832795
#define EPSILON 1e-5

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

float smoothing_kernel_derivative(float dist, float radius) {
    if (dist > radius) return 0.0;
    float diff = radius - dist;
    float scale = -45.0 / (M_PI * pow(radius, 6));
    return scale * diff * diff;  
}

// Viscosity kernel (corrected)
float laplacian_kernel(float dist, float radius)  {
    if (dist > radius) return 0.0;
    float scale = 45.0 / (M_PI * pow(radius, 6));
    return scale * (radius - dist);
}

float density_to_pressure(float density) {
    if (density < c.target_density) return 0.0;
    return c.pressure_constant * (density - c.target_density);
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

void handle_collision(inout vec3 position, inout vec3 velocity) {
    vec3 min_bound = vec3(c.bounds.x1, c.bounds.y1, c.bounds.z1);
    vec3 max_bound = vec3(c.bounds.x2, c.bounds.y2, c.bounds.z2);
    float r = c.smoothing_radius * 0.5;
    
    if (position.x < min_bound.x + r) {
        position.x = min_bound.x + r;
        velocity.x *= -c.bounds.damping_factor;
    } else if (position.x > max_bound.x - r) {
        position.x = max_bound.x - r;
        velocity.x *= -c.bounds.damping_factor;
    }
    
    if (position.y < min_bound.y + r) {
        position.y = min_bound.y + r;
        velocity.y *= -c.bounds.damping_factor;
    } else if (position.y > max_bound.y - r) {
        position.y = max_bound.y - r;
        velocity.y *= -c.bounds.damping_factor;
    }
    
    if (position.z < min_bound.z + r) {
        position.z = min_bound.z + r;
        velocity.z *= -c.bounds.damping_factor;
    } else if (position.z > max_bound.z - r) {
        position.z = max_bound.z - r;
        velocity.z *= -c.bounds.damping_factor;
    }
}

void main() {
    for (uint i = gl_GlobalInvocationID.x; i < c.size; i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
        Particle pi = p.particles[i];
        vec3 total_force = vec3(0.0);
        
        for (uint j = 0; j < c.size; j++) {
            if (i == j) continue;
            
            Particle pj = p.particles[j];
            vec3 dir = pi.position - pj.position;
            float dist = length(dir);
            
            if (dist < EPSILON || dist > c.smoothing_radius) continue;
            
            vec3 norm_dir = dir / max(dist, EPSILON);
            float pressure_i = density_to_pressure(pi.density);
            float pressure_j = density_to_pressure(pj.density);
            
            vec3 pressure_force = -norm_dir * c.particle_mass * 
                                (pressure_i/(pi.density*pi.density) + 
                                pressure_j/(pj.density*pj.density)) *
                                smoothing_kernel_derivative(dist, c.smoothing_radius);
            
            vec3 viscosity_force = c.viscosity_constant * c.particle_mass *
                                (pj.velocity - pi.velocity) / pj.density *
                                laplacian_kernel(dist, c.smoothing_radius);
            
            total_force += pressure_force + viscosity_force;
        }
        
        total_force += c.particle_mass * gravity;
        
        vec3 acceleration = total_force / c.particle_mass;
        vec3 new_velocity = pi.velocity + acceleration * c.dt;
        vec3 new_position = pi.position + new_velocity * c.dt;
        
        handle_collision(new_position, new_velocity);
        
        p.particles[i].velocity = new_velocity;
        p.particles[i].position = new_position;
        
        t.translations[i] = translation(new_position);
        t.translations[i][0][1] = length(new_velocity) / 10.0;
    }
}