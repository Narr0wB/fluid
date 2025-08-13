#version 450

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


layout(std430, binding = 0) buffer Particles {
    Particle particles[];
} p;

layout(std430, binding = 1) readonly buffer Forces {
    vec4 forces[];
} f; 

layout(std430, binding = 2) writeonly buffer Translations {
    mat4 translations[];
} t;

layout(push_constant) uniform Constants {
    uint size;
    float particle_mass;
    float dt;
    float smoothing_radius;
    BoundingBox bounds;
} c;

mat4 translation(vec3 pos) {
    return mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        pos.x, pos.y, pos.z, 1.0
    );
}

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
    uint stride = gl_WorkGroupSize.x * gl_NumWorkGroups.x;

    for (uint i = gl_GlobalInvocationID.x; i < c.size; i += stride) {
        vec3 acceleration = f.forces[i].xyz;
        vec3 new_velocity = p.particles[i].velocity + acceleration * c.dt;
        vec3 new_position = p.particles[i].position + new_velocity * c.dt;
        
        handle_collision(new_position, new_velocity);
        
        p.particles[i].velocity = new_velocity;
        p.particles[i].position = new_position;
        
        t.translations[i] = translation(new_position);
        t.translations[i][0][1] = length(new_velocity) / 10.0;
    }
}