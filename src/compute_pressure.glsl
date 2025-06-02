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



// spiky kernel
float smoothing_kernel_derivative(float dist, float radius) {
    if (dist > radius) {
        return 0.0;
    }
    
    float scale = -45.0 / (M_PI * pow(radius, 6));
    return scale * (radius - dist) * (radius - dist);
}

// viscosity kernel
float laplacian_kernel(float dist, float radius)  {
    if (dist > radius) {
        return 0.0;
    }

    float scale = 45.0 / (M_PI * pow(radius, 6));
    return scale * (radius - dist);
}

float density_to_pressure(float density) {
    if (density < c.target_density) return 0;

    return c.pressure_constant * (density - c.target_density);
}

mat4 translation(vec3 pos) {
    mat4 translation = mat4(1.0);

    translation[3][0] = pos.x;
    translation[3][1] = pos.y;
    translation[3][2] = pos.z;

    return translation;
}




shared vec3 forces;
const vec3 gravity = vec3(0.0, -9.81, 0.0);

const vec3 NORM_X1 = vec3(1, 0, 0);
const vec3 NORM_X2 = vec3(-1, 0, 0);
const vec3 NORM_Z1 = vec3(0, 0, 1);
const vec3 NORM_Z2 = vec3(0, 0, -1);
const vec3 NORM_Y1 = vec3(0, 1, 0);

void main() {
    for (uint i = gl_WorkGroupID.x; i < c.size; i += gl_NumWorkGroups.x) {
        Particle pi = p.particles[i];
        float di    = pi.density;
        
        if (gl_LocalInvocationID.x == 0) {
            forces = vec3(0.0);
        }

        barrier();

        for (uint j = gl_LocalInvocationID.x; j < c.size; j += gl_WorkGroupSize.x) {
            if (i == j) {continue;}

            Particle pj = p.particles[j];
            float dj = pj.density;

            vec3 dir = (pi.position - pj.position);
            float dist = length(dir);
            
            if (dist == 0) continue;
            if (dist >= c.smoothing_radius) {continue;}

            vec3 pressure_force = -(dir/dist) * c.particle_mass * c.particle_mass * ((density_to_pressure(dj)/(dj*dj) + density_to_pressure(di)/(di*di))) * smoothing_kernel_derivative(dist, c.smoothing_radius);
            vec3 viscosity_force = c.viscosity_constant * ((pi.velocity.xyz - pj.velocity.xyz) / dj) * c.particle_mass * laplacian_kernel(dist, c.smoothing_radius);

            pressure_force += viscosity_force;

            atomicAdd(forces.x, pressure_force.x);
            atomicAdd(forces.y, pressure_force.y);
            atomicAdd(forces.z, pressure_force.z);
        }

        barrier();
        
        if (gl_LocalInvocationID.x == 0)  {
            vec3 accel = (forces / di) + gravity;

            p.particles[i].velocity += (accel) * c.dt;

            vec3 step = p.particles[i].velocity * c.dt;
            vec3 new_position = p.particles[i].position + step;
            
            // Check if particle out of bounds
            if (new_position.x - (c.smoothing_radius / 2.0) <= c.bounds.x1) {
                p.particles[i].velocity.x *= -1.0 * (1.0 - c.bounds.damping_factor);
                // p.particles[i].velocity -= -1.0*(dot(p.particles[i].velocity, NORM_X1))*NORM_X1;
                new_position.x = c.bounds.x1 + (c.smoothing_radius / 2.0);
            }

            if (new_position.x + (c.smoothing_radius / 2.0) >= c.bounds.x2) {
                p.particles[i].velocity.x *= -1.0 * (1.0 - c.bounds.damping_factor);
                // p.particles[i].velocity -= -1.0*(dot(p.particles[i].velocity, NORM_X2))*NORM_X2;
                new_position.x = c.bounds.x2 - (c.smoothing_radius / 2.0);
            }

            if (new_position.z - (c.smoothing_radius / 2.0) <= c.bounds.z1) {
                p.particles[i].velocity.z *= -1.0 * (1.0 - c.bounds.damping_factor);
                // p.particles[i].velocity -= -1.0*(dot(p.particles[i].velocity, NORM_Z1))*NORM_Z1;
                new_position.z = c.bounds.z1 + (c.smoothing_radius / 2.0);
            }

            if (new_position.z + (c.smoothing_radius / 2.0) >= c.bounds.z2) {
                p.particles[i].velocity.z *= -1.0 * (1.0 - c.bounds.damping_factor);
                // p.particles[i].velocity -= -1.0*(dot(p.particles[i].velocity, NORM_Z2))*NORM_Z2;
                new_position.z = c.bounds.z2 - (c.smoothing_radius / 2.0);
            }

            if (new_position.y - (c.smoothing_radius / 2.0) <= c.bounds.y1) {
                p.particles[i].velocity.y *= -1.0 * (1.0 - c.bounds.damping_factor);
                // p.particles[i].velocity -= -1.0*(dot(p.particles[i].velocity, NORM_Y1))*NORM_Y1;
                new_position.y = c.bounds.y1 + (c.smoothing_radius / 2.0);
            }

            p.particles[i].position = new_position;
            t.translations[i] = translation(new_position);

            // We are going to use one of the cells of the translation matrix to propagate to the fragment shader
            // the velocity, normalized to the range [0, 1] in order to give a specific color to each particle based on the velocity
            t.translations[i][0][1] = length(pi.velocity) / 10.0;
        }
    }
}
