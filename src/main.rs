use nannou::noise::*;
use nannou::prelude::*;

const PARTICLES: usize = 2000;

fn main() {
    nannou::app(model).update(update).run();
}

struct Particle {
    pos: Vec2,
    pos_prev: Vec2,
    life: f32,
}

struct Model {
    _window: window::Id,
    land: Vec<Vec<f64>>,
    vectors: Vec<Vec<Vec2>>,
    particles: Vec<Particle>,
}

fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .size(1000, 1000)
        .view(view)
        .resized(window_resize)
        .build()
        .unwrap();

    let land = generate_land(app);
    let vectors = generate_vectors(app, &land);
    let particles = generate_particles(app, &land);

    Model {
        _window,
        land,
        vectors,
        particles,
    }
}

fn window_resize(app: &App, model: &mut Model, _dim: Vec2) {
    model.land = generate_land(app);
    model.vectors = generate_vectors(app, &model.land);
}

fn generate_land(app: &App) -> Vec<Vec<f64>> {
    let simplex = OpenSimplex::new().set_seed(1);
    let noise = Clamp::new(&simplex as &dyn NoiseFn<[f64; 2]>).set_bounds(0.0, 1.0);

    let r = app.window_rect();
    let scale = 0.0015;

    (0..r.w() as usize)
        .map(|x| {
            (0..r.h() as usize)
                .map(|y| noise.get([x as f64 * scale, y as f64 * scale]))
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>()
}

fn generate_vectors(app: &App, land: &Vec<Vec<f64>>) -> Vec<Vec<Vec2>> {
    let mut noise = Perlin::new();
    noise = noise.set_seed(1);

    let r = app.window_rect();
    let scale = 0.01;

    let potential = (0..r.w() as usize)
        .map(|x| {
            (0..r.h() as usize)
                .map(|y| {
                    // Step 1: Get raw noise value
                    let psi = noise.get([x as f64 * scale, y as f64 * scale]);

                    // Step 2: Apply boundary constraint to the potential
                    let h = land[x][y];
                    ramp(h) * psi
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    // Create vector field
    (0..r.w() as usize)
        .map(|x| {
            (0..r.h() as usize)
                .map(|y| {
                    if in_land(land, x, y) {
                        vec2(0.0, 0.0)
                    } else {
                        let h = 1.1;
                        // ψ(x, y+h) and ψ(x, y-h) for ∂ψ/∂y
                        let ψ_y_plus = sample_potential(
                            &potential,
                            x as usize,
                            (y as f64 + h) as usize,
                            r.w() as usize,
                            r.h() as usize,
                        );
                        let ψ_y_minus = sample_potential(
                            &potential,
                            x as usize,
                            (y as f64 - h) as usize,
                            r.w() as usize,
                            r.h() as usize,
                        );

                        // ψ(x+h, y) and ψ(x-h, y) for ∂ψ/∂x
                        let ψ_x_plus = sample_potential(
                            &potential,
                            (x as f64 + h) as usize,
                            y as usize,
                            r.w() as usize,
                            r.h() as usize,
                        );
                        let ψ_x_minus = sample_potential(
                            &potential,
                            (x as f64 - h) as usize,
                            y as usize,
                            r.w() as usize,
                            r.h() as usize,
                        );

                        // Finite difference approximations
                        let δψ_δy = (ψ_y_plus - ψ_y_minus) as f32;
                        let δψ_δx = (ψ_x_plus - ψ_x_minus) as f32;

                        // Stream function: v = (∂ψ/∂y, -∂ψ/∂x)
                        vec2(δψ_δy, -δψ_δx)
                    }
                })
                .collect::<Vec<Vec2>>()
        })
        .collect::<Vec<Vec<Vec2>>>()
}

fn generate_particles(app: &App, land: &Vec<Vec<f64>>) -> Vec<Particle> {
    // Initialize particle positions
    let r = app.window_rect();
    (0..PARTICLES)
        .map(|_| reset_particle(r, land))
        .collect::<Vec<Particle>>()
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let r = app.window_rect();
    let side = r.w().min(r.h());

    // Update particle positions
    for p in &mut model.particles {
        p.pos_prev = p.pos;
        let [x, y] = grid_space(&p.pos, &r);

        let velocity = model.vectors[x][y] * side * 0.02;
        p.pos.x += velocity.x;
        p.pos.y += velocity.y;

        // Reset particle if life exceeded
        if p.life <= -1.0 || is_out_of_bounds(p, r) {
            *p = reset_particle(r, &model.land);
        }

        p.life -= 0.01;
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    let r = app.window_rect();

    draw.rect()
        .w_h(app.window_rect().w(), app.window_rect().h())
        .rgba(0.0, 0.0, 0.0, 0.03); // Very transparent black

    // Create quiver field
    if app.elapsed_frames() == 0 {
        let step = 15;
        for x in (0..r.w() as usize).step_by(step) {
            for y in (0..r.h() as usize).step_by(step) {
                let side = r.w().min(r.h());
                let start = vec2(x as f32, y as f32) + r.bottom_left();

                let uv = model.vectors[x][y] * side * 0.05;

                let end = start + uv;

                draw.line()
                    .weight(1.0)
                    .points(start, end)
                    .rgba(1.0, 0.1, 0.1, 0.5);
            }
        }
    }

    // Draw land
    if app.elapsed_frames() == 0 {
        let step = 8;
        let dim = vec2(step as f32, step as f32);
        for x in (0..r.w() as usize).step_by(step) {
            for y in (0..r.h() as usize).step_by(step) {
                let pos = vec2(x as f32, y as f32) + r.bottom_left() + dim / 2.0;
                let h = model.land[x][y] as f32;
                let above_sea = if h == 0.0 { 0.5 } else { 0.0 };
                let sea = if h == 0.0 { 0.0 } else { 1.0 - h };
                let deepest_level = if h == 1.0 { 1.0 } else { 0.0 };

                draw.rect()
                    .xy(pos)
                    .wh(dim)
                    .rgb(deepest_level, above_sea, sea);
                // draw.text(&format!("{:.2}", h)).xy(pos).color(BLACK);
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

fn reset_particle(r: Rect, land: &Vec<Vec<f64>>) -> Particle {
    let pos = vec2(
        random_range(r.left(), r.right()),
        random_range(r.bottom(), r.top()),
    );

    // Convert world coordinates to grid coordinates
    let [x, y] = grid_space(&pos, &r);

    if !in_land(land, x as usize, y as usize) {
        Particle {
            pos: pos,
            pos_prev: pos,
            life: 1.0,
        }
    } else {
        reset_particle(r, land)
    }
}

fn grid_space(p: &Vec2, r: &Rect) -> [usize; 2] {
    [
        ((p.x - r.left()) as usize).clamp(0, (r.w() - 1.0) as usize),
        ((p.y - r.bottom()) as usize).clamp(0, (r.h() - 1.0) as usize),
    ]
}

fn is_out_of_bounds(p: &Particle, r: Rect) -> bool {
    p.pos.x < r.left() || p.pos.x > r.right() || p.pos.y < r.bottom() || p.pos.y > r.top()
}

fn in_land(land: &Vec<Vec<f64>>, x: usize, y: usize) -> bool {
    (land[x][y] as f32) <= 0.0
}

fn ramp(r: f64) -> f64 {
    if r >= 1.0 {
        1.0
    } else if r <= -1.0 {
        -1.0
    } else {
        15.0 / 8.0 * r - 10.0 / 8.0 * r.powi(3) + 3.0 / 8.0 * r.powi(5)
    }
}

fn sample_potential(
    potential: &Vec<Vec<f64>>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> f64 {
    potential[x.clamp(0, width - 1)][y.clamp(0, height - 1)]
}
