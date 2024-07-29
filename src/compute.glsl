#version 460
    
#extension GL_EXT_shader_atomic_float: enable
#define M_PI 3.1415926535897932384626433832795

layout(local_size_x=64, local_size_y=1, local_size_z=1) in;

struct Particle {
    vec4 position;
    vec4 velocity;
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
    float pressure_multiplier;
    float dt;
    BoundingBox bounds;
} c;

float smoothing_kernel(float dist, float radius) {
    // if (dist > radius) {
    //     return 0.0;
    // }
    
    float volume = 315.0 / (64.0 * M_PI * pow(radius, 9));
    return pow((radius * radius - dist * dist), 3) * volume;
}

float laplacian_kernel(float dist, float radius)  {
    if (dist > radius) {
        return 0.0;
    }
    float scale = 45.0 / (M_PI * pow(radius, 6));
    return scale * (radius - dist);
}

float smoothing_kernel_derivative(float dist, float radius) {
    if (dist > radius) {
        return 0.0;
    } 
    
    float scale = -45.0 / (M_PI * pow(radius, 6));
    return scale * (radius - dist) * (radius - dist);
}

float density_to_pressure(float density) {
    if (density < c.target_density) return 0;

    return (density - c.target_density) * c.pressure_multiplier;
}

mat4 translation(vec3 pos) {
    mat4 translation = mat4(1.0);

    translation[3][0] = pos.x;
    translation[3][1] = pos.y;
    translation[3][2] = pos.z;

    return translation;
}

shared vec3 pressure;
const vec3 gravity = vec3(0.0, -9.81, 0.0);

void main() {
    for (uint i = gl_WorkGroupID.x; i < c.size; i += gl_NumWorkGroups.x) {
        Particle pi = p.particles[i];
        float di    = p.particles[i].position.w;
        
        if (gl_LocalInvocationID.x == 0) {
            pressure = vec3(0.0);
        }

        barrier();

        for (uint j = gl_LocalInvocationID.x; j < c.size; j += gl_WorkGroupSize.x) {
            if (i == j) {continue;}
            Particle pj = p.particles[j];
            
            float dj = pj.position.w;
            float dist = length(pi.position.xyz - pj.position.xyz);
            vec3 dir = normalize(pi.position.xyz - pj.position.xyz);

            if (dist >= c.smoothing_radius) {continue;}

            vec3 contribution = -dir * c.particle_mass * (density_to_pressure(dj) + density_to_pressure(di)) / (2 * dj) * smoothing_kernel_derivative(dist, c.smoothing_radius);
            
            vec3 viscosity = (pj.velocity.xyz - pi.velocity.xyz) / dj * c.particle_mass * 1.5 * laplacian_kernel(dist, c.smoothing_radius);

            contribution += viscosity;

            atomicAdd(pressure.x, contribution.x);
            atomicAdd(pressure.y, contribution.y);
            atomicAdd(pressure.z, contribution.z);
        }

        barrier();
        
        if (gl_LocalInvocationID.x == 0)  {
            vec3 accel = pressure / di + gravity / di;

            // if (di == 0) {accel = vec3(100.0);}
            // if (isnan(accel.y) || isnan(accel.z) || isnan(accel.z)) {accel = vec3(0.0);}
            p.particles[i].velocity.xyz += (accel) * c.dt;

            vec3 step = p.particles[i].velocity.xyz * c.dt;
            vec3 new_position = p.particles[i].position.xyz + step;

            if (new_position.x <= c.bounds.x1) {
                p.particles[i].velocity.x *= -1.0 * (1.0 - c.bounds.damping_factor);
                new_position.x = c.bounds.x1 + (c.smoothing_radius / 2.0);
            }

            if (new_position.x >= c.bounds.x2) {
                p.particles[i].velocity.x *= -1.0 * (1.0 - c.bounds.damping_factor);
                new_position.x = c.bounds.x2 - (c.smoothing_radius / 2.0);
            }

            if (new_position.z <= c.bounds.z1) {
                p.particles[i].velocity.z *= -1.0 * (1.0 - c.bounds.damping_factor);
                new_position.z = c.bounds.z1 + (c.smoothing_radius / 2.0);
            }

            if (new_position.z >= c.bounds.z2) {
                p.particles[i].velocity.z *= -1.0 * (1.0 - c.bounds.damping_factor);
                new_position.z = c.bounds.z2 - (c.smoothing_radius / 2.0);
            }

            if (new_position.y <= c.bounds.y1) {
                p.particles[i].velocity.y *= -1.0 * (1.0 - c.bounds.damping_factor);
                new_position.y = c.bounds.y1 + c.smoothing_radius;
            }

            // if (new_position.x <= c.bounds.x1 || new_position.x >= c.bounds.x2 ||
            //     new_position.z <= c.bounds.z1 || new_position.z >= c.bounds.z2 ||
            //     new_position.y <= c.bounds.y1) 
            // {
            //     p.particles[i].velocity.xyz *= -1.0 * (1.0 - c.bounds.damping_factor);
            //     
            //     step = p.particles[i].velocity.xyz * c.dt;
            //     // if (abs(step.x) <= 0.05) {step.x = 0.0;}
            //     // if (abs(step.y) <= 0.05) {step.y = 0.0;}
            //     // if (abs(step.z) <= 0.05) {step.z = 0.0;}
            //
            //     new_position = p.particles[i].position.xyz + step; 
            // }
            
            p.particles[i].position.xyz = new_position;
            t.translations[i] = translation(new_position);
            t.translations[i][0][0] = length(pi.velocity.xyz) / 2.0;
        }
    }
}
