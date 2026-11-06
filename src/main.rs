mod camera;
mod controller;
mod fluid;
mod gpu;
mod renderer;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::camera::Camera;
use crate::controller::CameraController;
use crate::fluid::{Fluid, FluidComputePipeline, FluidSettings, SMOOTHING_RADIUS, particle_cube};
use crate::gpu::Gpu;
use crate::renderer::Renderer;
use crate::renderer::fluid::{FluidInstance, FluidMaterial};
use crate::renderer::mesh::{Mesh, MeshMaterial, Topology, floor_quad, wireframe_box};

const BOUNDS_MIN: Vec3 = Vec3::ZERO;
const BOUNDS_MAX: Vec3 = Vec3::new(2.0, 6.0, 2.0);

const REST_DENSITY: f32 = 1000.0;
const PRESSURE_CONSTANT: f32 = 3000.0;
const VISCOSITY: f32 = 0.03;
const RESTITUTION: f32 = 0.3;
const FRICTION: f32 = 0.1;

const SIM_STEP: f32 = 0.004;
const MAX_STEPS_PER_FRAME: u32 = 8;
const SLOW_FRAME_SECONDS: f32 = 1.0 / 30.0;
const FAST_FRAME_SECONDS: f32 = 1.0 / 55.0;

const OBJECT_CENTER: Vec3 = Vec3::new(1.0, 0.85, 1.0);
const OBJECT_SIZE: f32 = 1.1;

fn particle_spacing() -> f32 {
    (4.0 * SMOOTHING_RADIUS.powi(3) * std::f32::consts::PI / (3.0 * 50.0)).cbrt()
}

fn draw_radius() -> f32 {
    particle_spacing() * 0.8
}

fn collision_radius() -> f32 {
    particle_spacing() * 0.5
}

struct State {
    gpu: Gpu,
    renderer: Renderer,
    compute: FluidComputePipeline,
    fluid: Fluid,
    fluid_instance: FluidInstance,
    bounding_box: Mesh,
    floor: Mesh,
    object: Mesh,
}

impl State {
    fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let (gpu, surface) = pollster::block_on(Gpu::new(window.clone()))?;

        let size = window.inner_size();
        let camera = Camera::new(
            Vec3::new(3.6, 1.7, 3.8),
            Vec3::new(1.0, 0.8, 1.0),
            45f32.to_radians(),
            size.width.max(1) as f32 / size.height.max(1) as f32,
        );
        let renderer = Renderer::new(&gpu, surface, window, camera)?;

        let compute = FluidComputePipeline::new(&gpu);

        let spacing = particle_spacing();
        let mut particles = particle_cube(spacing, Vec3::new(0.5, 2.5, 0.5), Vec3::ZERO, 30);
        particles.extend(particle_cube(spacing, Vec3::new(0.8, 5.0, 0.8), Vec3::ZERO, 10));

        let fluid = Fluid::new(
            &compute,
            particles,
            FluidSettings {
                density: REST_DENSITY,
                pressure_const: PRESSURE_CONSTANT,
                viscosity: VISCOSITY,
                mass: REST_DENSITY * spacing.powi(3),
                restitution: RESTITUTION,
                friction: FRICTION,
                collision_radius: collision_radius(),
                bounds_min: BOUNDS_MIN,
                bounds_max: BOUNDS_MAX,
            },
        );
        let fluid_instance = renderer.create_fluid(
            fluid.particles(),
            fluid.count(),
            FluidMaterial {
                color: Vec3::new(0.15, 0.45, 0.75),
                particle_radius: draw_radius(),
            },
        );

        let (vertices, indices) = wireframe_box(BOUNDS_MIN, BOUNDS_MAX);
        let bounding_box = renderer.create_mesh(
            &vertices,
            &indices,
            Topology::Lines,
            Mat4::IDENTITY,
            MeshMaterial { color: Vec3::ONE, checker: false },
        );
        let (vertices, indices) = floor_quad(BOUNDS_MIN, BOUNDS_MAX);
        let floor = renderer.create_mesh(
            &vertices,
            &indices,
            Topology::Triangles,
            Mat4::IDENTITY,
            MeshMaterial { color: Vec3::new(0.75, 0.73, 0.68), checker: true },
        );

        let object_path = std::env::args_os()
            .nth(1)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/torus.obj"));
        let object = Mesh::load(
            &gpu.device,
            &gpu.queue,
            renderer.model_layout(),
            compute.collider_layout(),
            &object_path,
            MeshMaterial { color: Vec3::new(0.8, 0.3, 0.2), checker: false },
        )?;
        let (local_min, local_max) = object.local_bounds();
        let scale = OBJECT_SIZE / (local_max - local_min).max_element();
        let model = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            Quat::from_rotation_x(0.5),
            OBJECT_CENTER,
        ) * Mat4::from_translation(-(local_min + local_max) * 0.5);
        object.set_model(&gpu.queue, model, Vec3::ZERO);

        Ok(State { gpu, renderer, compute, fluid, fluid_instance, bounding_box, floor, object })
    }
}

struct App {
    state: Option<State>,
    error: Option<anyhow::Error>,
    controller: CameraController,
    last_frame: Instant,
    accumulator: f32,
    steps_per_frame: u32,
    paused: bool,
    shaded: bool,
    fps_timer: Instant,
    fps_frames: u32,
    cursor_captured: bool,
}

impl App {
    fn new() -> Self {
        App {
            state: None,
            error: None,
            controller: CameraController::new(2.0, 0.005, 5f32.to_radians()),
            last_frame: Instant::now(),
            accumulator: 0.0,
            steps_per_frame: 1,
            paused: false,
            shaded: true,
            fps_timer: Instant::now(),
            fps_frames: 0,
            cursor_captured: false,
        }
    }

    fn frame(&mut self) {
        let Some(state) = &mut self.state else {
            return;
        };

        let now = Instant::now();
        let elapsed = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.controller.update(state.renderer.camera_mut(), elapsed);

        if !self.paused {
            if elapsed > SLOW_FRAME_SECONDS {
                self.steps_per_frame = (self.steps_per_frame - 1).max(1);
            } else if elapsed < FAST_FRAME_SECONDS {
                self.steps_per_frame = (self.steps_per_frame + 1).min(MAX_STEPS_PER_FRAME);
            }

            self.accumulator += elapsed;
            let steps = ((self.accumulator / SIM_STEP) as u32).min(self.steps_per_frame);
            for _ in 0..steps {
                state.compute.step(&state.fluid, &[&state.object], SIM_STEP);
            }
            self.accumulator = (self.accumulator - steps as f32 * SIM_STEP).clamp(0.0, SIM_STEP);
        }

        state.renderer.render(
            &[&state.floor, &state.object, &state.bounding_box],
            &state.fluid_instance,
            self.shaded,
        );

        self.fps_frames += 1;
        let since_title = self.fps_timer.elapsed().as_secs_f32();
        if since_title >= 1.0 {
            let fps = self.fps_frames as f32 / since_title;
            let status = if self.paused { " | paused" } else { "" };
            state.renderer.window().set_title(&format!("Fluid | {} particles | {fps:.0} fps{status}", state.fluid.count()));
            self.fps_timer = Instant::now();
            self.fps_frames = 0;
        }
    }

    fn sync_cursor(&mut self) {
        let looking = self.controller.is_looking();
        if looking == self.cursor_captured {
            return;
        }
        let Some(state) = &self.state else {
            return;
        };

        let window = state.renderer.window();
        window.set_cursor_visible(!looking);
        let grabbed = if looking {
            window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
        } else {
            window.set_cursor_grab(CursorGrabMode::None)
        };
        if let Err(error) = grabbed {
            eprintln!("could not change cursor grab: {error}");
        }
        self.cursor_captured = looking;
    }

    fn handle_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::KeyC => self.paused = !self.paused,
            KeyCode::KeyT => self.shaded = !self.shaded,
            KeyCode::KeyR => {
                if let Some(state) = &self.state {
                    state.fluid.reset(&state.gpu.queue);
                }
                self.accumulator = 0.0;
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let attributes = Window::default_attributes().with_title("Fluid");
        let result = event_loop
            .create_window(attributes)
            .map_err(anyhow::Error::from)
            .and_then(|window| State::new(Arc::new(window)));
        match result {
            Ok(state) => {
                self.state = Some(state);
                self.last_frame = Instant::now();
            }
            Err(error) => {
                self.error = Some(error);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        self.controller.window_event(&event);
        self.sync_cursor();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    state.renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed
                    && !event.repeat
                    && let PhysicalKey::Code(code) = event.physical_key
                {
                    self.handle_key(code);
                }
            }
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        self.controller.device_event(&event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.renderer.window().request_redraw();
        }
    }
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    match app.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
