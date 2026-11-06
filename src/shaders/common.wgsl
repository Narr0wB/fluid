struct Camera {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    position: vec4<f32>,
}

struct FullscreenOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

const SUN_DIRECTION: vec3<f32> = vec3(0.4558, 0.7977, 0.3419);
const SUN_COLOR: vec3<f32> = vec3(3.0, 2.8, 2.5);

fn fullscreen_vertex(index: u32) -> FullscreenOut {
    let uv = vec2(f32((index << 1u) & 2u), f32(index & 2u));
    var out: FullscreenOut;
    out.clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

fn view_ray(uv: vec2<f32>, projection: mat4x4<f32>) -> vec3<f32> {
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return vec3(ndc.x / projection[0][0], ndc.y / projection[1][1], -1.0);
}

fn view_to_world_direction(direction: vec3<f32>, view: mat4x4<f32>) -> vec3<f32> {
    let rotation = mat3x3(view[0].xyz, view[1].xyz, view[2].xyz);
    return transpose(rotation) * direction;
}

fn environment(direction: vec3<f32>) -> vec3<f32> {
    let up = max(direction.y, 0.0);
    let sky = mix(vec3(0.85, 0.9, 0.98), vec3(0.22, 0.42, 0.78), pow(up, 0.5));
    let ground = mix(vec3(0.62, 0.66, 0.72), vec3(0.2, 0.2, 0.22), clamp(-direction.y * 3.0, 0.0, 1.0));
    let base = mix(ground, sky, smoothstep(-0.015, 0.015, direction.y));
    let sun = pow(max(dot(direction, SUN_DIRECTION), 0.0), 1500.0) * 20.0;
    return base + SUN_COLOR * sun;
}
