@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> FullscreenOut {
    return fullscreen_vertex(index);
}

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4<f32> {
    let direction = normalize(view_to_world_direction(view_ray(in.uv, camera.projection), camera.view));
    return vec4(environment(direction), 1.0);
}
