@group(0) @binding(0) var color: texture_2d<f32>;
@group(0) @binding(1) var color_sampler: sampler;

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3(0.0), vec3(1.0));
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> FullscreenOut {
    return fullscreen_vertex(index);
}

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4<f32> {
    return vec4(aces(textureSampleLevel(color, color_sampler, in.uv, 0.0).rgb), 1.0);
}
