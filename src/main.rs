use nannou::color::Gradient;
use nannou::noise::*;
use nannou::prelude::*;

const PARTICLES: usize = 2000;
const TIME_DILUTION: f32 = 0.05;
const LAND_GRID: usize = 4;
const CONTOUR_GRID: usize = 8;
const NUMBER_OF_CONTOURS: usize = 24;
const RENDER_PARTICLES: bool = false;

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

    let land = generate_land(app, 0.0);
    let vectors = generate_vectors(app, &land, 0.0);
    let particles = generate_particles(app, &land);

    Model {
        _window,
        land,
        vectors,
        particles,
    }
}

fn window_resize(app: &App, model: &mut Model, _dim: Vec2) {
    model.land = generate_land(app, app.time * TIME_DILUTION);
    model.vectors = generate_vectors(app, &model.land, app.time * TIME_DILUTION);
}

fn generate_land(app: &App, t: f32) -> Vec<Vec<f64>> {
    let r = app.window_rect();
    let simplex = OpenSimplex::new().set_seed(1);
    let noise = Turbulence::new(&simplex)
        .set_frequency(1.0)
        .set_power(3.0 * (app.mouse.y as f64) / (r.h() as f64))
        .set_roughness(2);
    // let noise = simplex;

    let scale = 0.007; // * (app.mouse.y as f64) / (r.h() as f64);

    (0..r.w() as usize)
        .step_by(LAND_GRID)
        .map(|x| {
            (0..r.h() as usize)
                .step_by(LAND_GRID)
                .map(|y| {
                    noise.get([
                        x as f64 * scale,
                        y as f64 * scale,
                        app.mouse.x as f64 * 0.001 + t as f64,
                    ])
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>()
}

fn generate_vectors(app: &App, land: &Vec<Vec<f64>>, t: f32) -> Vec<Vec<Vec2>> {
    let mut noise = Perlin::new();
    noise = noise.set_seed(1);

    let r = app.window_rect();
    let scale = 0.01;

    let potential = (0..r.w() as usize)
        .map(|x| {
            (0..r.h() as usize)
                .map(|y| {
                    // Step 1: Get raw noise value
                    let psi = noise.get([x as f64 * scale, y as f64 * scale, t as f64]);

                    // Step 2: Apply boundary constraint to the potential
                    let land_x = (x as usize / LAND_GRID).min(land.len() - 1);
                    let land_y = (y as usize / LAND_GRID).min(land[0].len() - 1);
                    let h = land[land_x][land_y];
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

    // Regenerate land and vectors each frame
    model.land = generate_land(app, app.time * TIME_DILUTION);

    if !RENDER_PARTICLES {
        return;
    };

    model.vectors = generate_vectors(app, &model.land, app.time * TIME_DILUTION);

    // Update particle positions
    for p in &mut model.particles {
        p.pos_prev = p.pos;
        let [x, y] = grid_space(&p.pos, &r);

        let velocity = model.vectors[x][y] * side * 0.04;
        p.pos.x += velocity.x;
        p.pos.y += velocity.y;

        // Reset particle if life exceeded
        if p.life <= -1.0 || is_out_of_bounds(p, r) {
            *p = reset_particle(r, &model.land);
        }

        p.life -= 0.01 * random::<f32>();
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    let r = app.window_rect();

    // Print FPS every 60 frames
    if app.elapsed_frames() % 60 == 0 {
        println!("FPS: {:.1}", app.fps());
    }

    draw.rect()
        .w_h(app.window_rect().w(), app.window_rect().h())
        .rgba(0.0, 0.0, 0.0, 0.1); // Very transparent black
    // draw.background().color(BLACK);

    // // Create quiver field
    // if app.elapsed_frames() == 0 {
    //     let step = 15;
    //     for x in (0..r.w() as usize).step_by(step) {
    //         for y in (0..r.h() as usize).step_by(step) {
    //             let side = r.w().min(r.h());
    //             let start = vec2(x as f32, y as f32) + r.bottom_left();

    //             let uv = model.vectors[x][y] * side * 0.05;

    //             let end = start + uv;

    //             draw.line()
    //                 .weight(1.0)
    //                 .points(start, end)
    //                 .rgba(1.0, 0.1, 0.1, 0.5);
    //         }
    //     }
    // }

    // // Draw land
    // let step = CONTOUR_GRID;
    // let dim = vec2(step as f32, step as f32);
    // for x in (0..r.w() as usize).step_by(step) {
    //     for y in (0..r.h() as usize).step_by(step) {
    //         let pos = vec2(x as f32, y as f32) + r.bottom_left() + dim / 2.0;

    //         let land_x = (x / LAND_GRID).min(model.land.len() - 1);
    //         let land_y = (y / LAND_GRID).min(model.land[0].len() - 1);
    //         let h = model.land[land_x][land_y] as f32;

    //         let above_sea = if h == 0.0 { 0.5 } else { 0.0 };
    //         let sea = if h == 0.0 { 0.0 } else { 1.0 - h };
    //         let deepest_level = if h == 1.0 { 1.0 } else { 0.0 };

    //         draw.rect()
    //             .xy(pos)
    //             .wh(dim)
    //             .rgb(deepest_level, above_sea, sea);
    //         // draw.text(&format!("{:.2}", h)).xy(pos).color(BLACK);
    //     }
    // }

    // Draw contours on land
    let step = CONTOUR_GRID;
    let (min, max) = model
        .land
        .iter()
        .flat_map(|inner| inner.iter())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
            (min.min(val), max.max(val))
        });
    let gradient = Gradient::new(vec![
        hsla(
            (app.time as f32 * TIME_DILUTION).cos(),
            (app.time as f32 * TIME_DILUTION).cos() * 0.438,
            0.5,
            0.1,
        ),
        hsla(
            (app.time as f32 * TIME_DILUTION + 3.14 / 2.0).cos(),
            0.0,
            0.0,
            0.0,
        ),
    ]);
    let contours: Vec<(f64, Hsla)> = (0..NUMBER_OF_CONTOURS)
        .map(|i| {
            (
                min + (max - min) * i as f64 / (NUMBER_OF_CONTOURS as f64 - 1.0),
                gradient.get(i as f32 / (NUMBER_OF_CONTOURS as f32 - 1.0)),
            )
        })
        .collect();
    for (contour, color) in contours {
        for x in (0..r.w() as usize).step_by(step) {
            for y in (0..r.h() as usize).step_by(step) {
                let pos = vec2(x as f32, y as f32);

                let step = step as f32;

                let p1 = pos;
                let p2 = pos + vec2(step, 0.0);
                let p3 = pos + vec2(step, step);
                let p4 = pos + vec2(0.0, step);
                let square = [p1, p2, p3, p4];

                // draw.polyline().points(square.map(|p| p + r.bottom_left()));

                let heights = square.map(|p| {
                    let land_x = (p.x as usize / LAND_GRID).min(model.land.len() - 1);
                    let land_y = (p.y as usize / LAND_GRID).min(model.land[0].len() - 1);
                    (p, model.land[land_x][land_y])
                });

                // for (p, h) in heights {
                //     draw.ellipse().xy(p + r.bottom_left()).wh(dim).rgb(
                //         if h <= contours[0] {
                //             1.0
                //         } else if h <= contours[1] {
                //             0.75
                //         } else if h <= contours[2] {
                //             0.5
                //         } else {
                //             0.25
                //         },
                //         0.0,
                //         0.0,
                //     );
                // }

                draw_contour(&draw, contour, heights, color, &r);
            }
        }
    }

    if !RENDER_PARTICLES {
        draw.to_frame(app, &frame).unwrap();
        return;
    };

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

fn draw_contour(draw: &Draw, c: f64, s: [(Vec2, f64); 4], color: Hsla, r: &Rect) {
    // Convert to binary index (0-15)
    let bits = s.map(|(_, h)| h >= c);
    let idx =
        (bits[0] as u8) | ((bits[1] as u8) << 1) | ((bits[2] as u8) << 2) | ((bits[3] as u8) << 3);

    // Helper to interpolate edge position based on heights
    let lerp_edge = |p1: Vec2, h1: f64, p2: Vec2, h2: f64| -> Vec2 {
        if (h2 - h1).abs() < 0.0001 {
            (p1 + p2) / 2.0
        } else {
            let t = ((c - h1) / (h2 - h1)).clamp(0.0, 1.0);
            p1 + (p2 - p1) * t as f32
        }
    };
    let midpoint_edge = |p1: Vec2, _h1: f64, p2: Vec2, _h2: f64| -> Vec2 { (p1 + p2) / 2.0 };
    let edge = midpoint_edge;

    // Interpolated edge positions (in grid space)
    let left = edge(s[3].0, s[3].1, s[0].0, s[0].1);
    let bottom = edge(s[0].0, s[0].1, s[1].0, s[1].1);
    let right = edge(s[1].0, s[1].1, s[2].0, s[2].1);
    let top = edge(s[2].0, s[2].1, s[3].0, s[3].1);

    // Helper do the actual drawing
    let line = |start: Vec2, end: Vec2| {
        draw.line()
            .start(start + r.bottom_left())
            .end(end + r.bottom_left())
            .weight(2.0)
            .color(color);
    };

    match idx {
        0 | 15 => {}
        1 => line(left, bottom),
        2 => line(bottom, right),
        3 => line(left, right),
        4 => line(right, top),
        5 => {
            line(left, bottom);
            line(right, top);
        }
        6 => line(bottom, top),
        7 => line(left, top),
        8 => line(top, left),
        9 => line(bottom, top),
        10 => {
            line(bottom, right);
            line(top, left);
        }
        11 => line(right, top),
        12 => line(right, left),
        13 => line(bottom, right),
        14 => line(left, bottom),
        _ => {}
    };

    return;

    // Fill the region where height >= c with color
    let offset = r.bottom_left();
    match idx {
        0 => {} // No fill
        1 => {
            // Triangle: p1, left, bottom
            draw.tri()
                .points(s[0].0 + offset, left + offset, bottom + offset)
                .color(color);
        }
        2 => {
            // Triangle: p2, bottom, right
            draw.tri()
                .points(s[1].0 + offset, bottom + offset, right + offset)
                .color(color);
        }
        3 => {
            // Quad: p1, p2, right, left
            draw.quad()
                .points(
                    s[0].0 + offset,
                    s[1].0 + offset,
                    right + offset,
                    left + offset,
                )
                .color(color);
        }
        4 => {
            // Triangle: p3, right, top
            draw.tri()
                .points(s[2].0 + offset, right + offset, top + offset)
                .color(color);
        }
        5 => {
            // Two triangles: (p1, left, bottom) and (p3, right, top)
            draw.tri()
                .points(s[0].0 + offset, left + offset, bottom + offset)
                .color(color);
            draw.tri()
                .points(s[2].0 + offset, right + offset, top + offset)
                .color(color);
        }
        6 => {
            // Quad: p2, p3, top, bottom
            draw.quad()
                .points(
                    s[1].0 + offset,
                    s[2].0 + offset,
                    top + offset,
                    bottom + offset,
                )
                .color(color);
        }
        7 => {
            // Pentagon: p1, p2, p3, top, left
            draw.polygon()
                .points([
                    s[0].0 + offset,
                    s[1].0 + offset,
                    s[2].0 + offset,
                    top + offset,
                    left + offset,
                ])
                .color(color);
        }
        8 => {
            // Triangle: p4, top, left
            draw.tri()
                .points(s[3].0 + offset, top + offset, left + offset)
                .color(color);
        }
        9 => {
            // Quad: p1, bottom, top, p4
            draw.quad()
                .points(
                    s[0].0 + offset,
                    bottom + offset,
                    top + offset,
                    s[3].0 + offset,
                )
                .color(color);
        }
        10 => {
            // Two triangles: (p2, bottom, right) and (p4, top, left)
            draw.tri()
                .points(s[1].0 + offset, bottom + offset, right + offset)
                .color(color);
            draw.tri()
                .points(s[3].0 + offset, top + offset, left + offset)
                .color(color);
        }
        11 => {
            // Pentagon: p1, p2, right, top, p4
            draw.polygon()
                .points([
                    s[0].0 + offset,
                    s[1].0 + offset,
                    right + offset,
                    top + offset,
                    s[3].0 + offset,
                ])
                .color(color);
        }
        12 => {
            // Quad: p3, p4, left, right
            draw.quad()
                .points(
                    s[2].0 + offset,
                    s[3].0 + offset,
                    left + offset,
                    right + offset,
                )
                .color(color);
        }
        13 => {
            // Pentagon: p1, bottom, right, p3, p4
            draw.polygon()
                .points([
                    s[0].0 + offset,
                    bottom + offset,
                    right + offset,
                    s[2].0 + offset,
                    s[3].0 + offset,
                ])
                .color(color);
        }
        14 => {
            // Pentagon: p2, p3, p4, left, bottom
            draw.polygon()
                .points([
                    s[1].0 + offset,
                    s[2].0 + offset,
                    s[3].0 + offset,
                    left + offset,
                    bottom + offset,
                ])
                .color(color);
        }
        15 => {
            // Full square
            draw.quad()
                .points(
                    s[0].0 + offset,
                    s[1].0 + offset,
                    s[2].0 + offset,
                    s[3].0 + offset,
                )
                .color(color);
        }
        _ => {}
    };
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
    let land_x = (x / LAND_GRID).min(land.len() - 1);
    let land_y = (y / LAND_GRID).min(land[0].len() - 1);
    (land[land_x][land_y] as f32) <= 0.0
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
