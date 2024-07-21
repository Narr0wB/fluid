#version 460
            
            layout(local_size_x=32, local_size_y=1, local_size_z=1) in;

            struct Particle {
                vec4 position;
                vec4 velocity;
            };

            layout(std140, binding = 0) buffer Particles {
                Particle particles[];
            } p;

            layout(push_constant) uniform Constants {
                uint size;
                float particle_mass;
                float smoothing_radius;
                float pressure_multiplier;
            } c;

            float smoothing_kernel(float dist, float radius) {
                if (dist > radius) {
                    return 0.0;
                }
                
                return exp(-abs(dist)) / 2.0;
            }

            void main() {
                for (uint i = gl_WorkGroupID.x; i < c.size; i += gl_NumWorkGroups.x) {
                    Particle local_particle = p.particles[i];
                    for (uint j = gl_LocalInvocationID.x; i < c.size; i += gl_WorkGroupSize.x) {
                        Particle current_particle = p.particles[j];

                        float dist = length(local_particle.position.xyz - current_particle.position.xyz);
                        float contribution = c.particle_mass * smoothing_kernel(dist, 3.0); 

                        atomicAdd(p.particles[i].position.w, contribution); 
                    } 
                }

                for (uint i = gl_WorkGroupID.x; i < c.size; ++i) {
                    i = 1;
                }
            }
