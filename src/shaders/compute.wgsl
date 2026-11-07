struct Particle {
    position: vec3<f32>,
    density: f32,
    velocity: vec3<f32>,
    pressure: f32,
}

struct Params {
    count: u32,
    active_buffer: u32,
    viscosity: f32,
    density: f32,
    pressure_const: f32,
    mass: f32,
    restitution: f32,
    friction: f32,
}

struct Collider {
    inv_model: mat4x4<f32>,
    bounds_min: vec3<f32>,
    scale: f32,
    bounds_max: vec3<f32>,
    cell_size: f32,
    velocity: vec3<f32>,
    _pad: f32,
}

// One spatial grid cell: its particles occupy [start, start + count) in the sorted particle buffer.
struct Cell {
    start: atomic<u32>,
    count: atomic<u32>,
}

struct Environment {
    bounds_min: vec3<f32>,
    dt: f32,
    bounds_max: vec3<f32>,
    spatial_cell_size: f32,
    grid_dims: vec3<u32>,
    max_speed: f32,
    collision_radius: f32,
}

override H: f32 = 0.08;
override H2: f32 = H * H;
const PI: f32 = 3.14159265;
override POLY6: f32 = 315.0 / (64.0 * PI * H2 * H2 * H2 * H2 * H);
override SPIKY: f32 = -45.0 / (PI * H2 * H2 * H2);
override LAPLACIAN: f32 = -SPIKY;
const GRAVITY: vec3<f32> = vec3(0.0, -9.81, 0.0);

// The 3x3x3 block of cells around (and including) a particle's cell, x varying fastest.
const NEIGHBOR_OFFSETS = array<vec3<i32>, 27>(
    vec3(-1, -1, -1), vec3( 0, -1, -1), vec3( 1, -1, -1),
    vec3(-1,  0, -1), vec3( 0,  0, -1), vec3( 1,  0, -1),
    vec3(-1,  1, -1), vec3( 0,  1, -1), vec3( 1,  1, -1),

    vec3(-1, -1,  0), vec3( 0, -1,  0), vec3( 1, -1,  0),
    vec3(-1,  0,  0), vec3( 0,  0,  0), vec3( 1,  0,  0),
    vec3(-1,  1,  0), vec3( 0,  1,  0), vec3( 1,  1,  0),

    vec3(-1, -1,  1), vec3( 0, -1,  1), vec3( 1, -1,  1),
    vec3(-1,  0,  1), vec3( 0,  0,  1), vec3( 1,  0,  1),
    vec3(-1,  1,  1), vec3( 0,  1,  1), vec3( 1,  1,  1),
);

@group(0) @binding(0) var<uniform>             env: Environment;
@group(0) @binding(1) var<storage, read_write> indices: array<Cell>;

@group(1) @binding(0) var<storage, read_write> particles_1: array<Particle>;
@group(1) @binding(1) var<storage, read_write> particles_2: array<Particle>;
@group(1) @binding(2) var<storage, read_write> accelerations: array<vec4<f32>>;
@group(1) @binding(3) var<uniform>             params: Params;

@group(2) @binding(0) var<uniform> collider: Collider;
@group(2) @binding(1) var          sdf_scene: texture_3d<f32>;
@group(2) @binding(2) var          sdf_sampler: sampler;


fn density_to_pressure(density: f32) -> f32 {
    let r = density / params.density;
    let r2 = r * r;
    let r7 = r2 * r2 * r2 * r;
    return max(params.pressure_const * (r7 - 1.0), 0.0);
}

fn cell_index(position: vec3<f32>) -> vec3<u32> {
    let cell = floor((position - env.bounds_min) / env.spatial_cell_size);
    return vec3<u32>(clamp(cell, vec3(0.0), vec3<f32>(env.grid_dims - 1u)));
}

fn cell_idx_to_key(index: vec3<u32>) -> u32 {
    return index.x + index.y * env.grid_dims.x + index.z * env.grid_dims.x * env.grid_dims.y;
}

fn load(i: u32) -> Particle {
    if (params.active_buffer == 0u) {
        return particles_1[i];
    }

    return particles_2[i];
}

fn store(i: u32, p: Particle) {
    if (params.active_buffer == 0u) {
        particles_1[i] = p;
    } else {
        particles_2[i] = p;
    }
}

@compute @workgroup_size(64)
fn count(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count {
        return;
    }

    let particle = load(i);
    let key = cell_idx_to_key(cell_index(particle.position));
    atomicAdd(&indices[key].count, 1u);
}


const SCAN_THREADS: u32 = 256u;
var<workgroup> partial: array<u32, SCAN_THREADS>;

@compute @workgroup_size(SCAN_THREADS)
fn scan(@builtin(local_invocation_index) id: u32) {
    let cell_count = env.grid_dims.x * env.grid_dims.y * env.grid_dims.z;
    let step = (cell_count / SCAN_THREADS) + 1u;

    // Local scan
    var slice_total = 0u;
    for (var i = id * step; i < cell_count && i < (id + 1u) * step; i++) {
        slice_total += atomicLoad(&indices[i].count);
    }

    // Publish
    partial[id] = slice_total;

    workgroupBarrier();

    // Local running total reconstruction
    var running_total = 0u;
    for (var i = 0u; i < id; i++) {
        running_total += partial[i];
    }

    // Write each cell's start
    var accumulator = 0u;
    for (var i = id * step; i < cell_count && i < (id + 1u) * step; i++) {
        atomicStore(&indices[i].start, running_total + accumulator);
        accumulator += atomicLoad(&indices[i].count);
    }
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count {
        return;
    }

    let particle = load(i);
    let key = cell_idx_to_key(cell_index(particle.position));
    let slot = atomicAdd(&indices[key].start, 1u);

    if (params.active_buffer == 0u) {
        particles_2[slot] = particle;
    } else {
        particles_1[slot] = particle;
    }
}

@compute @workgroup_size(64)
fn density(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count {
        return;
    }

    var pi = load(i);
    var sum = 0.0;

    let base = vec3<i32>(cell_index(pi.position));
    for (var j = 0u; j < 27u; j++) {
        let cell  = base + NEIGHBOR_OFFSETS[j];
        if any(cell < vec3(0)) || any(cell >= vec3<i32>(env.grid_dims)) {
            continue;
        }

        let key   = cell_idx_to_key(vec3<u32>(cell));
        let count = atomicLoad(&indices[key].count);
        let start = atomicLoad(&indices[key].start) - count;

        for (var k = start; k < start + count; k++) {
            let dist = pi.position - load(k).position;
            let r2 = dot(dist, dist);

            if r2 > H2 {
                continue;
            }

            let w = H2 - r2;
            sum += w * w * w;
        }
    }

    let rho = sum * params.mass * POLY6;
    pi.density = rho;
    pi.pressure = density_to_pressure(rho);
    store(i, pi);
}

@compute @workgroup_size(64)
fn pressure(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count {
        return;
    }

    let pi = load(i);
    let pi_term = pi.pressure / (pi.density * pi.density);
    var acceleration = vec3(0.0);

    let base = vec3<i32>(cell_index(pi.position));
    for (var j = 0u; j < 27u; j++) {
        let cell  = base + NEIGHBOR_OFFSETS[j];
        if any(cell < vec3(0)) || any(cell >= vec3<i32>(env.grid_dims)) {
            continue;
        }

        let key   = cell_idx_to_key(vec3<u32>(cell));
        let count = atomicLoad(&indices[key].count);
        let start = atomicLoad(&indices[key].start) - count;

        for (var k = start; k < start + count; k++) {
            if k == i {
                continue;
            }

            let pk = load(k);
            let dist = pi.position - pk.position;
            let r2 = dot(dist, dist);

            if r2 > H2 || r2 < 1e-10 {
                continue;
            }

            let r = sqrt(r2);
            let q = H - r;
            let pk_term = pk.pressure / (pk.density * pk.density);

            acceleration += -(dist / r) * params.mass * (pi_term + pk_term) * SPIKY * q * q;
            acceleration += params.viscosity * params.mass * ((pk.velocity - pi.velocity) / pk.density) * LAPLACIAN * q;
        }
    }

    accelerations[i] = vec4(acceleration + GRAVITY, 0.0);
}

@compute @workgroup_size(64)
fn integration(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count {
        return;
    }

    var p = load(i);

    var vel = p.velocity + accelerations[i].xyz * env.dt;
    let speed = length(vel);
    if speed > env.max_speed {
        vel *= env.max_speed / speed;
    }

    var pos = p.position + vel * env.dt;

    let lo = env.bounds_min + env.collision_radius;
    let hi = env.bounds_max - env.collision_radius;
    for (var axis = 0; axis < 3; axis++) {
        if pos[axis] < lo[axis] {
            pos[axis] = lo[axis];
            vel[axis] *= -params.restitution;
        } else if pos[axis] > hi[axis] {
            pos[axis] = hi[axis];
            vel[axis] *= -params.restitution;
        }
    }

    p.velocity = vel;
    p.position = pos;
    store(i, p);
}

fn sample_sdf(local: vec3<f32>) -> f32 {
    let uvw = (local - collider.bounds_min) / (collider.bounds_max - collider.bounds_min);
    return textureSampleLevel(sdf_scene, sdf_sampler, uvw, 0.0).r;
}

@compute @workgroup_size(64)
fn collide(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count {
        return;
    }

    var p = load(i);
    let world = p.position;
    let local = (collider.inv_model * vec4(world, 1.0)).xyz;
    if any(local < collider.bounds_min) || any(local > collider.bounds_max) {
        return;
    }

    let distance = sample_sdf(local) * collider.scale;
    if distance >= env.collision_radius {
        return;
    }

    let e = collider.cell_size;
    let gradient = vec3(
        sample_sdf(local + vec3(e, 0.0, 0.0)) - sample_sdf(local - vec3(e, 0.0, 0.0)),
        sample_sdf(local + vec3(0.0, e, 0.0)) - sample_sdf(local - vec3(0.0, e, 0.0)),
        sample_sdf(local + vec3(0.0, 0.0, e)) - sample_sdf(local - vec3(0.0, 0.0, e)),
    );
    let inv = mat3x3(collider.inv_model[0].xyz, collider.inv_model[1].xyz, collider.inv_model[2].xyz);
    let world_gradient = transpose(inv) * gradient;
    let gradient_length = length(world_gradient);
    if gradient_length < 1e-6 {
        return;
    }
    let normal = world_gradient / gradient_length;

    var relative = p.velocity - collider.velocity;
    let normal_speed = dot(relative, normal);
    if normal_speed < 0.0 {
        let tangent = relative - normal * normal_speed;
        relative = tangent * (1.0 - params.friction) - normal * normal_speed * params.restitution;
    }

    p.position = world + normal * (env.collision_radius - distance);
    p.velocity = relative + collider.velocity;
    store(i, p);
}
