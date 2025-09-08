use nannou::prelude::*;
use nannou::noise::*;

fn main() {
    nannou::app(model)
        .update(update)
        .run();
}

struct Particle {
    pos: Vec2,
    pos_prev: Vec2,
    life: usize,
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

    let mut noise = Perlin::new();
    noise = noise.set_seed(1);

    let r = app.window_rect();
    let scale = 0.01;

    // Create vector field
    let vectors = (0 .. r.w() as usize)
        .map(|x| {
            (0 .. r.h() as usize)
                .map(|y| {
                    let offset = 1.1;
                    let n1 = noise.get([x as f64 * scale, (y as f64 + offset) * scale]);
                    let n2 = noise.get([x as f64 * scale, (y as f64 - offset) * scale]);
                    let n3 = noise.get([(x as f64 + offset) * scale, y as f64 * scale]);
                    let n4 = noise.get([(x as f64 - offset) * scale, y as f64 * scale]);

                    let u = (n1 - n2) as f32;
                    let v = (n4 - n3) as f32;
                    vec2(u, v)
                }).collect::<Vec<Vec2>>()
        }).collect::<Vec<Vec<Vec2>>>();

    // Initialize particle positions
    let particles = (0 .. 1000)
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
        let uv = model.vectors[x][y] * side * 0.02;
        p.pos.x += uv.x;
        p.pos.y += uv.y;

        // Reset particle if life exceeded
        if p.life == 0 || is_out_of_bounds(p, r) {
            *p = reset_particle(r);
        }

        p.life -= 1;
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

    // Update and draw particles
    for p in &model.particles {
        draw.line()
            .start(p.pos_prev)
            .end(p.pos)
            .weight(1.0)
            .color(WHITE);

        draw.ellipse()
            .x_y(p.pos.x, p.pos.y)
            .radius(0.5)
            .color(WHITE);
    }

    draw.to_frame(app, &frame).unwrap();
}

fn _dynamic_noise(app: &App, _model: &Model, _frame: Frame) {
    let draw = app.draw();
    let r = app.window_rect();

    draw.background().color(BLACK);

    let mut noise = Perlin::new();
    noise = noise.set_seed(1);
    let scale = 0.01;

    let time = app.time * 0.1;

    let step = 15;
    for x in (0 .. r.w() as usize).step_by(step) {
        for y in (0 .. r.h() as usize).step_by(step) {
            let side = r.w().min(r.h());
            let start = vec2(x as f32, y as f32) + r.bottom_left();

            let offset = 100.1;
            let n1 = noise.get([x as f64 * scale, (y as f64 + offset) * scale, time as f64]);
            let n2 = noise.get([x as f64 * scale, (y as f64 - offset) * scale, time as f64]);
            let n3 = noise.get([(x as f64 + offset) * scale, y as f64 * scale, time as f64]);
            let n4 = noise.get([(x as f64 - offset) * scale, y as f64 * scale, time as f64]);

            let u = (n1 - n2) as f32;
            let v = (n4 - n3) as f32;
            let uv = vec2(u as f32, v as f32) * side * 0.02;

            let mut end = start + uv;

            let start_to_mouse = app.mouse.position() - start;
            if start_to_mouse.length() < 100.0 {
                let target_mag = start_to_mouse.length().min(side * 0.5);
                end += start_to_mouse.normalize_or_zero() * (-target_mag * 0.05);
            }

            draw.line()
                .weight(1.0)
                .points(start, end)
                .rgb(1.0, 1.0, 1.0);
        }
    }
}

fn reset_particle(r: Rect) -> Particle {
    let pos = vec2(
            random_range(r.left(), r.right()),
            random_range(r.bottom(), r.top()),
        );
    Particle {
        pos: pos,
        pos_prev: pos,
        life: random_range(10, 1000),
    }
}

fn is_out_of_bounds(p: &Particle, r: Rect) -> bool {
    p.pos.x < r.left() || p.pos.x > r.right() || p.pos.y < r.bottom() || p.pos.y > r.top()
}
