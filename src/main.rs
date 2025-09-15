use nannou::prelude::*;
use nannou::noise::*;

const OBSTACLE_RADIUS: f32 = 100.0;

fn main() {
    nannou::app(model)
        .update(update)
        .run();
}

struct Particle {
    pos: Vec2,
    pos_prev: Vec2,
    life: f32,
}

struct Model {
    _window: window::Id,
    vectors: Vec<Vec<Vec2>>,
    particles: Vec<Particle>,
}

fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .size(800, 800)
        .view(view)
        .build()
        .unwrap();

    let mut noise = Billow::new();
    noise = noise.set_seed(1);

    let r = app.window_rect();
    let scale = 0.005;

    let potential = (0 .. r.w() as usize)
        .map(|x| {
            (0 .. r.h() as usize)
                .map(|y| {
                    // Step 1: Get raw noise value
                    let psi = noise.get([x as f64 * scale, y as f64 * scale]);

                    // Step 2: Apply boundary constraint to the potential
                    let distance = distance(x, y) as f64 - OBSTACLE_RADIUS as f64;
                    ramp(distance / 100.0) * psi
                }).collect::<Vec<f64>>()
        }).collect::<Vec<Vec<f64>>>();

    // Create vector field
    let vectors = (0 .. r.w() as usize)
        .map(|x| {
            (0 .. r.h() as usize)
                .map(|y| {
                    let h = 1.1;
                    // ψ(x, y+h) and ψ(x, y-h) for ∂ψ/∂y
                    let ψ_y_plus  = sample_potential(&potential, x as usize, (y as f64 + h) as usize, r.w() as usize, r.h() as usize);
                    let ψ_y_minus = sample_potential(&potential, x as usize, (y as f64 - h) as usize, r.w() as usize, r.h() as usize);

                    // ψ(x+h, y) and ψ(x-h, y) for ∂ψ/∂x
                    let ψ_x_plus = sample_potential(&potential, (x as f64 + h) as usize, y as usize, r.w() as usize, r.h() as usize);
                    let ψ_x_minus = sample_potential(&potential, (x as f64 - h) as usize, y as usize, r.w() as usize, r.h() as usize);

                    // Finite difference approximations
                    let δψ_δy = (ψ_y_plus - ψ_y_minus) as f32;
                    let δψ_δx = (ψ_x_plus - ψ_x_minus) as f32;

                    // Stream function: v = (∂ψ/∂y, -∂ψ/∂x)
                    let v = vec2(δψ_δy, -δψ_δx);

                    if in_obstacle(x, y) {
                        vec2(0.0, 0.0)
                    // } else if in_obstacle_buffer_zone(x, y) {
                    //     let n = normal_vector_of_obstacle(x, y);
                    //     v - v.dot(n)*n
                    } else {
                        v
                    }
                }).collect::<Vec<Vec2>>()
        }).collect::<Vec<Vec<Vec2>>>();

    // Initialize particle positions
    let particles = (0 .. 500)
        .map(|_| {
            reset_particle(r)
        }).collect::<Vec<Particle>>();

    Model { _window, vectors, particles }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let r = app.window_rect();
    let side = r.w().min(r.h());

    // Update particle positions
    for p in &mut model.particles {
        p.pos_prev = p.pos;
        let x = ((p.pos.x - r.left()) as usize).clamp(0, (r.w() - 1.0) as usize);
        let y = ((p.pos.y - r.bottom()) as usize).clamp(0, (r.h() - 1.0) as usize);

        let velocity = model.vectors[x][y] * side * 0.02;
        p.pos.x += velocity.x;
        p.pos.y += velocity.y;

        // Reset particle if life exceeded
        if p.life <= -1.0 || is_out_of_bounds(p, r) {
            *p = reset_particle(r);
        }

        p.life -= 0.01;
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    let r = app.window_rect();

    // draw.background().color(BLACK);
    draw.rect()
        .w_h(app.window_rect().w(), app.window_rect().h())
        .color(srgba(0.0, 0.0, 0.0, 0.05)); // Very transparent black

    // Create quiver field
    if app.elapsed_frames() == 0 {
        let step = 15;
        for x in (0 .. r.w() as usize).step_by(step) {
            for y in (0 .. r.h() as usize).step_by(step) {
                let side = r.w().min(r.h());
                let start = vec2(x as f32, y as f32) + r.bottom_left();

                let uv = model.vectors[x][y] * side * 0.02;

                let end = start + uv;

                draw.line()
                    .weight(1.0)
                    .points(start, end)
                    .rgb(0.1, 0.1, 0.1);
            }
        }
    }

    // Draw particles
    for p in &model.particles {
        draw.line()
            .start(p.pos_prev)
            .end(p.pos)
            .weight(1.0)
            // .color(WHITE);
            .color(srgba(1.0, 1.0, 1.0, 1.0 - abs(p.life))); // Very transparent white

        draw.ellipse()
            .x_y(p.pos.x, p.pos.y)
            .radius(0.5)
            // .color(WHITE);
            .color(srgba(1.0, 1.0, 1.0, 1.0 - abs(p.life))); // Very transparent white
    }

    draw.to_frame(app, &frame).unwrap();
}

fn reset_particle(r: Rect) -> Particle {
    let pos = vec2(
            random_range(r.left(), r.right()),
            random_range(r.bottom(), r.top()),
        );

    // Convert world coordinates to grid coordinates
    let grid_x = ((pos.x - r.left()) as usize).clamp(0, (r.w() - 1.0) as usize);
    let grid_y = ((pos.y - r.bottom()) as usize).clamp(0, (r.h() - 1.0) as usize);

    if !in_obstacle(grid_x as usize, grid_y as usize) {
        Particle {
            pos: pos,
            pos_prev: pos,
            life: 1.0,
        }
    } else {
        reset_particle(r)
    }
}

fn is_out_of_bounds(p: &Particle, r: Rect) -> bool {
    p.pos.x < r.left() || p.pos.x > r.right() || p.pos.y < r.bottom() || p.pos.y > r.top()
}

fn in_obstacle(x: usize, y: usize) -> bool {
    let center: Vec2 = vec2(400.0, 400.0);

    let dist = (vec2(x as f32, y as f32) - center).length();
    dist <= OBSTACLE_RADIUS
}

fn distance(x: usize, y: usize) -> f32 {
    let center: Vec2 = vec2(400.0, 400.0);
    let pos = vec2(x as f32, y as f32);
    (pos - center).length()
}

fn ramp(r: f64) -> f64 {
    if r >= 1.0 { 1.0 }
    else if r <= -1.0 { -1.0 }
    else { 15.0/8.0 * r - 10.0/8.0 * r.powi(3) + 3.0/8.0 * r.powi(5) }
}

fn sample_potential(potential: &Vec<Vec<f64>>, x: usize, y: usize, width: usize, height: usize) -> f64 {
    potential[x.clamp(0, width - 1)][y.clamp(0, height - 1)]
}
