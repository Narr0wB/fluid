struct Particle {
    position: vec3<f32>,
    density: f32,
    velocity: vec3<f32>,
    pressure: f32,
}

struct Params {
    bounds_min: vec3<f32>,
    count: u32,
    bounds_max: vec3<f32>,
    viscosity: f32,
    density: f32,
    pressure_const: f32,
    mass: f32,
    dt: f32,
    restitution: f32,
    friction: f32,
    collision_radius: f32,
    max_speed: f32,
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

override H: f32 = 0.08;
override H2: f32 = H * H;
const PI: f32 = 3.14159265;
override POLY6: f32 = 315.0 / (64.0 * PI * H2 * H2 * H2 * H2 * H);
override SPIKY: f32 = -45.0 / (PI * H2 * H2 * H2);
override LAPLACIAN: f32 = -SPIKY;
const GRAVITY: vec3<f32> = vec3(0.0, -9.81, 0.0);

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> accelerations: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@group(1) @binding(0) var<uniform> collider: Collider;
@group(1) @binding(1) var sdf_scene: texture_3d<f32>;
@group(1) @binding(2) var sdf_sampler: sampler;

fn density_to_pressure(density: f32) -> f32 {
    let r = density / params.density;
    let r2 = r * r;
    let r7 = r2 * r2 * r2 * r;
    return max(params.pressure_const * (r7 - 1.0), 0.0);
}

@compute @workgroup_size(64)
fn density(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count {
        return;
    }

    let xi = particles[i].position;
    var sum = 0.0;

    for (var j = 0u; j < params.count; j++) {
        let dist = xi - particles[j].position;
        let r2 = dot(dist, dist);

        if r2 > H2 {
            continue;
        }

        let w = H2 - r2;
        sum += w * w * w;
    }

    let rho = sum * params.mass * POLY6;
    particles[i].density = rho;
    particles[i].pressure = density_to_pressure(rho);
}

@compute @workgroup_size(64)
fn pressure(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count {
        return;
    }

    let xi = particles[i].position;
    let vi = particles[i].velocity;
    let rhoi = particles[i].density;
    let pi_term = particles[i].pressure / (rhoi * rhoi);
    var acceleration = vec3(0.0);

    for (var j = 0u; j < params.count; j++) {
        if j == i {
            continue;
        }

        let dist = xi - particles[j].position;
        let r2 = dot(dist, dist);

        if r2 > H2 || r2 < 1e-10 {
            continue;
        }

        let r = sqrt(r2);
        let q = H - r;
        let rhoj = particles[j].density;
        let pj_term = particles[j].pressure / (rhoj * rhoj);

        acceleration += -(dist / r) * params.mass * (pi_term + pj_term) * SPIKY * q * q;
        acceleration += params.viscosity * params.mass * ((particles[j].velocity - vi) / rhoj) * LAPLACIAN * q;
    }

    accelerations[i] = vec4(acceleration + GRAVITY, 0.0);
}

@compute @workgroup_size(64)
fn integration(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count {
        return;
    }

    var vel = particles[i].velocity + accelerations[i].xyz * params.dt;
    let speed = length(vel);
    if speed > params.max_speed {
        vel *= params.max_speed / speed;
    }

    var pos = particles[i].position + vel * params.dt;

    let lo = params.bounds_min + params.collision_radius;
    let hi = params.bounds_max - params.collision_radius;
    for (var axis = 0; axis < 3; axis++) {
        if pos[axis] < lo[axis] {
            pos[axis] = lo[axis];
            vel[axis] *= -params.restitution;
        } else if pos[axis] > hi[axis] {
            pos[axis] = hi[axis];
            vel[axis] *= -params.restitution;
        }
    }

    particles[i].velocity = vel;
    particles[i].position = pos;
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

    let world = particles[i].position;
    let local = (collider.inv_model * vec4(world, 1.0)).xyz;
    if any(local < collider.bounds_min) || any(local > collider.bounds_max) {
        return;
    }

    let distance = sample_sdf(local) * collider.scale;
    if distance >= params.collision_radius {
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

    var relative = particles[i].velocity - collider.velocity;
    let normal_speed = dot(relative, normal);
    if normal_speed < 0.0 {
        let tangent = relative - normal * normal_speed;
        relative = tangent * (1.0 - params.friction) - normal * normal_speed * params.restitution;
    }

    particles[i].position = world + normal * (params.collision_radius - distance);
    particles[i].velocity = relative + collider.velocity;
}
