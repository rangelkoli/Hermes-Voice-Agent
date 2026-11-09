//! Animated status orb, drawn two pixels per terminal cell with half blocks.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

pub const FPS: f32 = 12.0;
// Each tick moves this share of the way to the new look, so states melt together.
const EASE: f32 = 0.2;
// Mic RMS is small: speech at a laptop sits near 0.02 to 0.1. Raise this for a quiet mic.
const MIC_GAIN: f32 = 2.5;
const LIGHT: [f32; 3] = [-0.45, -0.55, 0.7];

type Look = [f32; 8];

/// main rgb, accent rgb, swirl, speed in radians per second
fn look(status: &str) -> Look {
    match status {
        "loading" => [110., 110., 115., 70., 70., 80., 0.5, 0.8],
        "waiting" => [150., 125., 75., 90., 60., 110., 0.5, 0.5],
        "hearing" => [125., 255., 178., 61., 222., 200., 0.8, 2.0],
        "transcribing" => [108., 182., 255., 95., 208., 224., 0.9, 2.5],
        "thinking" => [95., 208., 224., 150., 90., 255., 1.0, 3.5],
        "speaking" => [255., 180., 84., 255., 90., 130., 0.9, 4.0],
        "muted" => [70., 72., 76., 50., 52., 56., 0.0, 0.0],
        "approval" => [255., 191., 0., 255., 120., 0., 0.8, 5.0],
        "offline" => [255., 45., 45., 120., 20., 30., 0.3, 0.6],
        "mic lost" => [255., 79., 216., 120., 40., 160., 0.3, 0.6],
        _ => [61., 222., 122., 40., 150., 190., 0.6, 0.9],
    }
}

/// A square RGB image, two pixels tall per row of cells, row-major.
pub fn paint(rows: usize, background: [f32; 3], look: &Look, phase: f32, level: f32) -> Vec<[u8; 3]> {
    let size = rows * 2;
    let radius = 0.64 + 0.04 * phase.sin() + 0.14 * level;
    let mut image = Vec::with_capacity(size * size);
    for row in 0..size {
        for col in 0..size {
            let x = (col as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let y = (row as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let distance = x.hypot(y);
            // Surface normals of a sphere, then one light from the upper left.
            let (nx, ny) = (x / radius, y / radius);
            let nz = (1.0 - nx * nx - ny * ny).clamp(0.0, 1.0).sqrt();
            let facing = (LIGHT[0] * nx + LIGHT[1] * ny + LIGHT[2] * nz).clamp(0.0, 1.0);
            let bands = (4.0 * nx + 2.5 * (3.0 * ny + phase).sin() + 1.7 * phase).sin()
                * (3.0 * ny - 2.0 * nz + phase).sin();
            let mix = 0.5 + 0.5 * look[6] * bands;
            let shade = 0.4 + 0.6 * facing;
            let rim = (1.0 - nz).powi(3) * 0.35;
            let gloss = 110.0 * facing.powi(24);
            // A soft halo fades out before the image edge so it never shows a hard border.
            let halo = (-7.0 * (distance - radius).max(0.0)).exp()
                * (0.16 + 0.3 * level)
                * ((1.0 - distance) * 6.0).clamp(0.0, 1.0);
            let edge = ((radius - distance) * size as f32 / 2.0).clamp(0.0, 1.0);
            let mut pixel = [0u8; 3];
            for ch in 0..3 {
                let (main, accent) = (look[ch], look[ch + 3]);
                let body = (main * (1.0 - mix) + accent * mix) * shade + main * rim + gloss;
                let outside = background[ch] + (main - background[ch]) * halo;
                pixel[ch] = (outside * (1.0 - edge) + body * edge).clamp(0.0, 255.0) as u8;
            }
            image.push(pixel);
        }
    }
    image
}

/// The app status as a living sphere that swells with the mic level.
pub struct Orb {
    look: Look,
    phase: f32,
    swell: f32,
}

impl Default for Orb {
    fn default() -> Self {
        Self { look: look("listening"), phase: 0.0, swell: 0.0 }
    }
}

impl Orb {
    pub fn tick(&mut self, status: &str, level: f32) {
        let target = look(status);
        for (now, goal) in self.look.iter_mut().zip(target) {
            *now += (goal - *now) * EASE;
        }
        self.phase += self.look[7] / FPS;
        let swell = if status == "muted" { 0.0 } else { (level.sqrt() * MIC_GAIN).min(1.0) };
        self.swell += (swell - self.swell) * 0.4;
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer, background: [u8; 3]) {
        let rows = area.height as usize;
        if rows == 0 {
            return;
        }
        let back = background.map(f32::from);
        let image = paint(rows, back, &self.look, self.phase, self.swell);
        let size = rows * 2;
        let width = size.min(area.width as usize);
        let left = area.x + (area.width as usize - width) as u16 / 2;
        let rgb = |p: [u8; 3]| Color::Rgb(p[0], p[1], p[2]);
        for y in 0..rows {
            for x in 0..width {
                let cell = &mut buf[(left + x as u16, area.y + y as u16)];
                cell.set_symbol("▀");
                cell.set_fg(rgb(image[y * 2 * size + x]));
                cell.set_bg(rgb(image[(y * 2 + 1) * size + x]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orb_centre_is_lit_and_corners_are_background() {
        let image = paint(8, [0.0; 3], &look("listening"), 0.0, 0.0);
        assert_eq!(image.len(), 16 * 16);
        assert_eq!(image[0], [0, 0, 0]);
        assert!(image[8 * 16 + 8].iter().any(|&c| c > 40));
    }
}
