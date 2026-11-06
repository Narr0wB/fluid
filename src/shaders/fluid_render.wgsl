struct Particle {
    position: vec3<f32>,
    density: f32,
    velocity: vec3<f32>,
    pressure: f32,
}

struct FluidParams {
    color: vec3<f32>,
    particle_radius: f32,
}

struct SphereOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) center: vec3<f32>,
    @location(2) speed: f32,
}

struct DepthOut {
    @location(0) linear_depth: f32,
    @builtin(frag_depth) depth: f32,
}

struct ColorDepthOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

const DEPTH_BLUR_TAPS: i32 = 14;
const THICKNESS_BLUR_TAPS: i32 = 8;
const THICKNESS_BLUR_RADIUS: f32 = 10.0;
const MAX_BLUR_PIXELS: f32 = 48.0;

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> fluid: FluidParams;
@group(1) @binding(1) var<storage, read> particles: array<Particle>;

@group(2) @binding(0) var source: texture_2d<f32>;

@group(2) @binding(1) var fluid_depth: texture_2d<f32>;
@group(2) @binding(2) var fluid_thickness: texture_2d<f32>;
@group(2) @binding(3) var scene_color: texture_2d<f32>;
@group(2) @binding(4) var scene_depth: texture_depth_2d;
@group(2) @binding(5) var linear_sampler: sampler;

@vertex
fn vs_sphere(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> SphereOut {
    var corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    let corner = corners[vertex];
    let center = (camera.view * vec4(particles[instance].position, 1.0)).xyz;

    var out: SphereOut;
    out.clip = camera.projection * vec4(center + vec3(corner * fluid.particle_radius, 0.0), 1.0);
    out.corner = corner;
    out.center = center;
    out.speed = length(particles[instance].velocity);
    return out;
}

fn sphere_view_position(in: SphereOut) -> vec3<f32> {
    let r2 = dot(in.corner, in.corner);
    if r2 > 1.0 {
        discard;
    }
    return in.center + vec3(in.corner, sqrt(1.0 - r2)) * fluid.particle_radius;
}

fn clip_depth(view_position: vec3<f32>) -> f32 {
    let clip = camera.projection * vec4(view_position, 1.0);
    return clip.z / clip.w;
}

@fragment
fn fs_sphere_depth(in: SphereOut) -> DepthOut {
    let position = sphere_view_position(in);
    return DepthOut(-position.z, clip_depth(position));
}

@fragment
fn fs_thickness(in: SphereOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.corner, in.corner);
    if r2 > 1.0 {
        discard;
    }
    return vec4(2.0 * fluid.particle_radius * sqrt(1.0 - r2), 0.0, 0.0, 1.0);
}

fn speed_gradient(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    let blue = vec3(0.0, 0.114, 1.0);
    let teal = vec3(0.0, 1.0, 0.612);
    let yellow = vec3(0.953, 1.0, 0.0);
    let red = vec3(1.0, 0.0, 0.0);
    if t < 0.54 {
        return mix(blue, teal, t / 0.54);
    }
    if t < 0.76 {
        return mix(teal, yellow, (t - 0.54) / 0.22);
    }
    return mix(yellow, red, (t - 0.76) / 0.24);
}

@fragment
fn fs_sphere_speed(in: SphereOut) -> ColorDepthOut {
    let position = sphere_view_position(in);
    let normal = vec3(in.corner, sqrt(max(1.0 - dot(in.corner, in.corner), 0.0)));
    let shade = 0.55 + 0.45 * normal.z;
    return ColorDepthOut(vec4(speed_gradient(in.speed / 10.0) * shade, 1.0), clip_depth(position));
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> FullscreenOut {
    return fullscreen_vertex(index);
}

fn load_source(coord: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(source));
    return textureLoad(source, clamp(coord, vec2(0), size - 1), 0).r;
}

fn blur_depth(coord: vec2<i32>, direction: vec2<i32>) -> f32 {
    let center = load_source(coord);
    if center <= 0.0 {
        return 0.0;
    }

    let height = f32(textureDimensions(source).y);
    let world_radius = fluid.particle_radius * 2.5;
    let pixels = clamp(world_radius * camera.projection[1][1] * 0.5 * height / center, 1.0, MAX_BLUR_PIXELS);
    let sigma = pixels * 0.5;
    let range = fluid.particle_radius * 4.0;

    var sum = 0.0;
    var weights = 0.0;
    for (var i = -DEPTH_BLUR_TAPS; i <= DEPTH_BLUR_TAPS; i++) {
        let offset = f32(i) / f32(DEPTH_BLUR_TAPS) * pixels;
        let sample_depth = load_source(coord + direction * i32(round(offset)));
        if sample_depth <= 0.0 {
            continue;
        }
        let spatial = offset / sigma;
        let difference = (sample_depth - center) / range;
        let weight = exp(-0.5 * spatial * spatial - difference * difference);
        sum += sample_depth * weight;
        weights += weight;
    }
    return sum / weights;
}

fn blur_thickness(coord: vec2<i32>, direction: vec2<i32>) -> f32 {
    var sum = 0.0;
    var weights = 0.0;
    for (var i = -THICKNESS_BLUR_TAPS; i <= THICKNESS_BLUR_TAPS; i++) {
        let offset = f32(i) / f32(THICKNESS_BLUR_TAPS) * THICKNESS_BLUR_RADIUS;
        let spatial = offset / (THICKNESS_BLUR_RADIUS * 0.5);
        let weight = exp(-0.5 * spatial * spatial);
        sum += load_source(coord + direction * i32(round(offset))) * weight;
        weights += weight;
    }
    return sum / weights;
}

@fragment
fn fs_blur_depth_x(in: FullscreenOut) -> @location(0) vec4<f32> {
    return vec4(blur_depth(vec2<i32>(in.clip.xy), vec2(1, 0)), 0.0, 0.0, 1.0);
}

@fragment
fn fs_blur_depth_y(in: FullscreenOut) -> @location(0) vec4<f32> {
    return vec4(blur_depth(vec2<i32>(in.clip.xy), vec2(0, 1)), 0.0, 0.0, 1.0);
}

@fragment
fn fs_blur_thickness_x(in: FullscreenOut) -> @location(0) vec4<f32> {
    return vec4(blur_thickness(vec2<i32>(in.clip.xy), vec2(1, 0)), 0.0, 0.0, 1.0);
}

@fragment
fn fs_blur_thickness_y(in: FullscreenOut) -> @location(0) vec4<f32> {
    return vec4(blur_thickness(vec2<i32>(in.clip.xy), vec2(0, 1)), 0.0, 0.0, 1.0);
}

fn view_position_at(coord: vec2<i32>, size: vec2<f32>) -> vec4<f32> {
    let clamped = clamp(coord, vec2(0), vec2<i32>(size) - 1);
    let depth = textureLoad(fluid_depth, clamped, 0).r;
    let uv = (vec2<f32>(clamped) + 0.5) / size;
    return vec4(view_ray(uv, camera.projection) * depth, depth);
}

fn closest_difference(center: vec4<f32>, forward: vec4<f32>, backward: vec4<f32>) -> vec3<f32> {
    let forward_valid = forward.w > 0.0;
    let backward_valid = backward.w > 0.0;
    if forward_valid && (!backward_valid || abs(forward.w - center.w) <= abs(center.w - backward.w)) {
        return forward.xyz - center.xyz;
    }
    if backward_valid {
        return center.xyz - backward.xyz;
    }
    return vec3(0.0);
}

fn scene_linear_depth(coord: vec2<i32>) -> f32 {
    let depth = textureLoad(scene_depth, coord, 0);
    return camera.projection[3][2] / (depth + camera.projection[2][2]);
}

@fragment
fn fs_composite(in: FullscreenOut) -> @location(0) vec4<f32> {
    let background = textureSampleLevel(scene_color, linear_sampler, in.uv, 0.0).rgb;
    let coord = vec2<i32>(in.clip.xy);
    let size = vec2<f32>(textureDimensions(fluid_depth));
    let center = view_position_at(coord, size);
    if center.w <= 0.0 {
        return vec4(background, 1.0);
    }

    let dx = closest_difference(center, view_position_at(coord + vec2(1, 0), size), view_position_at(coord - vec2(1, 0), size));
    let dy = closest_difference(center, view_position_at(coord - vec2(0, 1), size), view_position_at(coord + vec2(0, 1), size));
    let crossed = cross(dx, dy);
    let view_normal = select(vec3(0.0, 0.0, 1.0), normalize(crossed), dot(crossed, crossed) > 1e-14);

    let thickness = textureSampleLevel(fluid_thickness, linear_sampler, in.uv, 0.0).r;
    let normal = view_to_world_direction(view_normal, camera.view);
    let to_camera = view_to_world_direction(normalize(-center.xyz), camera.view);

    let n_dot_v = clamp(dot(normal, to_camera), 0.0, 1.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - n_dot_v, 5.0);

    let refraction_strength = min(thickness * 0.35, 0.06);
    let refracted_uv = clamp(in.uv + view_normal.xy * vec2(-1.0, 1.0) * refraction_strength, vec2(0.0), vec2(1.0));
    let refracted_coord = clamp(vec2<i32>(refracted_uv * size), vec2(0), vec2<i32>(size) - 1);
    var transmitted = background;
    if scene_linear_depth(refracted_coord) >= center.w {
        transmitted = textureSampleLevel(scene_color, linear_sampler, refracted_uv, 0.0).rgb;
    }

    let absorption = (vec3(1.0) - fluid.color) * 5.0;
    let attenuation = exp(-absorption * thickness);
    let scatter = fluid.color * (1.0 - exp(-thickness * 1.5)) * 0.25;
    let water = transmitted * attenuation + scatter;

    let reflected = environment(reflect(-to_camera, normal));
    let half_vector = normalize(to_camera + SUN_DIRECTION);
    let specular = pow(max(dot(normal, half_vector), 0.0), 600.0) * SUN_COLOR * 2.0;

    let lit = mix(water, reflected, fresnel) + specular;
    let coverage = smoothstep(0.0, 0.008, thickness);
    return vec4(mix(background, lit, coverage), 1.0);
}
