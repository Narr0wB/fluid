struct Model {
    model: mat4x4<f32>,
    color: vec3<f32>,
    checker: f32,
}

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> mesh: Model;

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> VertexOut {
    let world = mesh.model * vec4(position, 1.0);
    var out: VertexOut;
    out.clip = camera.view_projection * world;
    out.world = world.xyz;
    out.normal = (mesh.model * vec4(normal, 0.0)).xyz;
    return out;
}

fn checker(world: vec2<f32>) -> f32 {
    let cell = world / 0.25;
    let width = max(fwidth(cell), vec2(1e-4));
    let tri = abs(fract(cell * 0.5) - 0.5) * 2.0 - 0.5;
    let smooth_edge = clamp(tri / width, vec2(-1.0), vec2(1.0));
    return 0.5 + 0.5 * smooth_edge.x * smooth_edge.y;
}

@fragment
fn fs_lit(in: VertexOut) -> @location(0) vec4<f32> {
    let to_camera = normalize(camera.position.xyz - in.world);
    var normal = normalize(in.normal);
    if dot(normal, to_camera) < 0.0 {
        normal = -normal;
    }

    var albedo = mesh.color;
    if mesh.checker > 0.5 {
        albedo *= mix(0.55, 1.0, checker(in.world.xz));
    }

    let half_vector = normalize(to_camera + SUN_DIRECTION);
    let diffuse = max(dot(normal, SUN_DIRECTION), 0.0);
    let ambient = mix(vec3(0.25, 0.24, 0.22), vec3(0.45, 0.55, 0.7), normal.y * 0.5 + 0.5);
    let specular = pow(max(dot(normal, half_vector), 0.0), 64.0) * 0.12;
    return vec4(albedo * (ambient + SUN_COLOR * diffuse * 0.18) + SUN_COLOR * specular, 1.0);
}

@fragment
fn fs_line(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4(mesh.color, 1.0);
}
