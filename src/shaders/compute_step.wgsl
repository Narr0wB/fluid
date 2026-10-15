
struct Particle {
    position: vec3<f32>,
    density: f32,
    velocity: vec3<f32>,
    pressure: f32,
}

struct Params {
    count: u32,
    viscosity: f32,
    density: f32,
    pressure_const: f32,
    mass: f32,
    dt: f32,
}

struct Collider {
    inv_model: mat4<f32>,
    bounds_min: vec3<f32>,
    scale: f32,
    bounds_max: vec3<f32>,
    cell_size: f32,
    velocity: vec3<f32>,
    _pad: f32,
}

override H: f32 = 0.04;
override H2: f32 = H*H;
const PI: f32 = 3.14159265;
override POLY6: f32 = 315.0 / (64.0  * PI * H2*H2*H2*H2*H);
override SPIKY: f32 = -45.0 / (PI * H2*H2*H2);
override LAPLACIAN: f32 = -SPIKY;
const GRAVITY: vec3<f32> = vec3(0.0, -9.81, 0.0);

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> forces: array<vec3>;
@group(0) @binding(2) var<uniform> params: Params;

@group(1) @binding(0) var<uniform> collider: Collider;
@group(1) @binding(1) var sdf_scene: texture_3d<f32>;
@group(1) @binding(2) var sdf_sampler: sampler;

fn density_to_pressure(density: f32) -> f32 {
    let rho0 = params.density;
    let B = params.pressure_const;
    let r = density / rho0;
    let r2 = r * r;
    let r7 = r2 * r2 * r2 * r;
    return max(B * (r7 - 1.0), 0.0);
}

fn handle_collistion() {

}

@compute @workgroup_size(64)
fn density(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count { 
        return; 
    }

    let xi  = particles[i].position;
    var sum = 0.0;

    for (var j = 0u; j < params.count; ++j) {
        let dist = xi - paricles[j].position;
        let r2   = dot(dist, dist);

        if r2 > H2 { continue; }

        let w = H2 - r2;
        sum  += w * w * w;
    }

    particles[i].density = sum * params.mass * POLY6;
}

@compute @workgroup_size(64)
fn pressure(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count {
        return;
    }

    let xi    = particles[i].position;
    let rhoi  = particles[i].density;
    let pi    = density_to_pressure(rhoi);
    let force = vec3(0.0);

    for (var j = 0; j < params.count; ++j) {
        if j == i { continue; } 

        let dist = xi - particles[j].position;
        let r2   = dot(dist, dist);

        if r2 > H2 { continue; }

        let norm = dist / r2;
        let rhoj = particles[j].density;
        let pj   = density_to_pressure(rhoj);

        let pressure_force  = -norm * params.mass * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj)) * SPIKY * (H - r2) * (H - r2);
        let viscosity_force = params.viscosity * params.mass * ((particles[j].velocity - particles[i].velocity) / rhoj) * LAPLACIAN * (H - r2);

        force += pressure_force * viscosity_force;
    }

    forces[i] = force + GRAVITY;
}

@compute @workgroup_size(64)
fn integration(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count {
        return;
    }

    let accel = forces[i] / particles[i].density;
    let vel   = particles[i].velocity + accel * params.dt;
    let pos   = particles[i].position + vel * params.dt;

    handle_collisions(pos, vel);

    particles[i].velocity = vel;
    particles[i].position = pos;
}