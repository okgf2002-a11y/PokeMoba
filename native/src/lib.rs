//! Optional Rust acceleration for the PokeMoba engine.
//!
//! Every function here reproduces, bit for bit, the pure-Python code it replaces
//! (engine/grid.py, engine/movement.py, engine/spatial.py): same samples, same
//! visiting order, same floating-point operations. The engine falls back to the
//! Python code when this module is missing, so it must never change a result.

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

const TILE_BUSH: u8 = 4;
const TILE_PILLAR: u8 = 7;
const RES: usize = 8; // fine cells per tile (engine/wall_shape.RES)
const WALL_R_MAX: f64 = 0.42; // movement._WALL_R_MAX
const SHAPE_FRACS: [f64; 5] = [0.2, 0.4, 0.6, 0.8, 1.0];

// ---------------------------------------------------------------------------
// math.hypot, exactly as CPython's vector_norm (Modules/mathmodule.c, 3.10+).

#[inline]
fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

#[inline]
fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let z = x - a;
    (x, b - z)
}

fn vector_norm(vec: &mut [f64], max: f64) -> f64 {
    if max.is_infinite() {
        return max;
    }
    if vec.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    if max == 0.0 || vec.len() <= 1 {
        return max;
    }
    let max_e = frexp_exp(max);
    if max_e < -1023 {
        for v in vec.iter_mut() {
            *v /= f64::MIN_POSITIVE;
        }
        return f64::MIN_POSITIVE * vector_norm(vec, max / f64::MIN_POSITIVE);
    }
    let scale = ldexp1(-max_e);
    let (mut csum, mut frac1, mut frac2) = (1.0f64, 0.0f64, 0.0f64);
    for &v in vec.iter() {
        let x = v * scale;
        let pr = dl_mul(x, x);
        let sm = dl_fast_sum(csum, pr.0);
        csum = sm.0;
        frac1 += pr.1;
        frac2 += sm.1;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let pr = dl_mul(-h, h);
    let sm = dl_fast_sum(csum, pr.0);
    csum = sm.0;
    frac1 += pr.1;
    frac2 += sm.1;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

/// Exponent e of frexp(v) (v = m * 2^e, 0.5 <= m < 1), v finite and > 0.
fn frexp_exp(v: f64) -> i32 {
    let bits = v.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    if exp == 0 {
        // subnormal
        let mant = bits & ((1u64 << 52) - 1);
        let lz = mant.leading_zeros() as i32 - 12; // zeros inside the 52-bit field
        -1022 - lz
    } else {
        exp - 1022
    }
}

/// 2^e as f64 (exact, subnormal results included).
fn ldexp1(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

#[inline]
fn hypot(x: f64, y: f64) -> f64 {
    let (ax, ay) = (x.abs(), y.abs());
    let mut v = [ax, ay];
    // CPython: max over the abs values; NaN handled inside.
    let m = if ay > ax { ay } else { ax };
    vector_norm(&mut v, m)
}

#[pyfunction]
fn py_hypot(x: f64, y: f64) -> f64 {
    hypot(x, y)
}

// ---------------------------------------------------------------------------

/// Read-only snapshot of a Grid with the wall shape on (Grid._shape_on True).
#[pyclass(frozen)]
struct GridView {
    w: usize,
    h: usize,
    tiles: Vec<u8>,     // row-major [y * w + x]
    fine: Vec<u8>,      // _shape_b: (h*8) x (w*8)
    any_t: Vec<u8>,     // _shape_any_b: per tile
    dirs: Vec<(f64, f64)>, // movement._SHAPE_DIRS (computed in Python: same libm)
}

impl GridView {
    #[inline]
    fn tile(&self, x: usize, y: usize) -> u8 {
        self.tiles[y * self.w + x]
    }

    #[inline]
    fn in_bounds_f(&self, px: f64, py: f64) -> bool {
        0.0 <= px && px < self.w as f64 && 0.0 <= py && py < self.h as f64
    }

    #[inline]
    fn fine_at(&self, px: f64, py: f64) -> bool {
        let fy = (py * 8.0) as usize;
        let fx = (px * 8.0) as usize;
        self.fine[fy * self.w * RES + fx] != 0
    }

    #[inline]
    fn walkable_point(&self, px: f64, py: f64, bb: bool) -> bool {
        if !self.in_bounds_f(px, py) {
            return false;
        }
        let t = self.tile(px as usize, py as usize);
        if t == TILE_PILLAR || (bb && t == TILE_BUSH) {
            return false;
        }
        !self.fine_at(px, py)
    }

    #[inline]
    fn blocked(&self, px: f64, py: f64) -> bool {
        self.in_bounds_f(px, py) && self.fine_at(px, py)
    }

    fn near(&self, px: f64, py: f64, r: f64) -> bool {
        let (w, h) = (self.w as i64, self.h as i64);
        let x0 = ((px - r).floor() as i64).max(0);
        let x1 = ((px + r).floor() as i64).min(w - 1);
        let y0 = ((py - r).floor() as i64).max(0);
        let y1 = ((py + r).floor() as i64).min(h - 1);
        let mut ty = y0;
        while ty <= y1 {
            let mut tx = x0;
            while tx <= x1 {
                if self.any_t[(ty * w + tx) as usize] != 0 {
                    return true;
                }
                tx += 1;
            }
            ty += 1;
        }
        false
    }

    fn los(&self, ax: f64, ay: f64, bx: f64, by: f64, bb: bool) -> bool {
        let (dx, dy) = (bx - ax, by - ay);
        let dist = hypot(dx, dy);
        if dist < 1e-9 {
            return self.walkable_point(ax, ay, bb);
        }
        let steps = (dist * 4.0) as i64 + 1;
        let sf = steps as f64;
        let fw = self.w * RES;
        for i in 0..=steps {
            let t = i as f64 / sf;
            let x = ax + dx * t;
            let y = ay + dy * t;
            if !self.in_bounds_f(x, y) {
                return false;
            }
            if self.fine[(y * 8.0) as usize * fw + (x * 8.0) as usize] != 0 {
                return false;
            }
            let tt = self.tile(x as usize, y as usize);
            if tt == TILE_PILLAR || (bb && tt == TILE_BUSH) {
                return false;
            }
        }
        true
    }

    fn shape_push(&self, px: f64, py: f64, r: f64) -> Option<(f64, f64)> {
        if self.blocked(px, py) {
            return None;
        }
        let (mut sx, mut sy, mut deep) = (0.0f64, 0.0f64, 0.0f64);
        for &(c, s) in self.dirs.iter() {
            for &f in SHAPE_FRACS.iter() {
                if self.blocked(px + c * r * f, py + s * r * f) {
                    let d = r * (1.0 - f) + 0.2 * r;
                    sx -= c * d;
                    sy -= s * d;
                    if d > deep {
                        deep = d;
                    }
                    break;
                }
            }
        }
        if deep <= 0.0 {
            return None;
        }
        let n = hypot(sx, sy);
        if n < 1e-9 {
            return None;
        }
        Some((px + sx / n * deep, py + sy / n * deep))
    }

    /// movement._keep_off_walls with the shape on; r already = min(radius, 0.42).
    fn keep_off(&self, px0: f64, py0: f64, r: f64, bb: bool) -> (f64, f64) {
        let (mut px, mut py) = (px0, py0);
        let cx = px.floor() as i64;
        let cy = py.floor() as i64;
        let (w, h) = (self.w as i64, self.h as i64);
        for _ in 0..2 {
            let mut moved = false;
            for ty in (cy - 1)..(cy + 2) {
                for tx in (cx - 1)..(cx + 2) {
                    if 0 <= tx && tx < w && 0 <= ty && ty < h {
                        let t = self.tile(tx as usize, ty as usize);
                        if !(t == TILE_PILLAR || (bb && t == TILE_BUSH)) {
                            continue;
                        }
                    }
                    let (txf, tyf) = (tx as f64, ty as f64);
                    // min(max(px, tx), tx + 1): equal values whichever side wins.
                    let qx = px.max(txf).min(txf + 1.0);
                    let qy = py.max(tyf).min(tyf + 1.0);
                    let (dx, dy) = (px - qx, py - qy);
                    let d2 = dx * dx + dy * dy;
                    if d2 >= r * r || d2 < 1e-12 {
                        continue;
                    }
                    let d = d2.sqrt();
                    let k = (r - d) / d;
                    let (nx, ny) = (px + dx * k, py + dy * k);
                    if self.walkable_point(nx, ny, bb) {
                        px = nx;
                        py = ny;
                        moved = true;
                    }
                }
            }
            if self.near(px, py, r) {
                if let Some((nx, ny)) = self.shape_push(px, py, r) {
                    if self.walkable_point(nx, ny, bb) {
                        px = nx;
                        py = ny;
                        moved = true;
                    }
                }
            }
            if !moved {
                break;
            }
        }
        (px, py)
    }

    /// movement._move_out (+ _keep_off_walls) on index i.
    #[inline]
    fn move_out(&self, xs: &mut [f64], ys: &mut [f64], i: usize, ox: f64, oy: f64,
                kr: f64, bb: bool) {
        let nx = xs[i] + ox;
        let ny = ys[i] + oy;
        if self.walkable_point(nx, ny, bb) {
            let (px, py) = self.keep_off(nx, ny, kr, bb);
            xs[i] = px;
            ys[i] = py;
        }
    }
}

#[inline]
fn cell_key(x: f64, y: f64) -> (i64, i64) {
    // SpatialHash._key with cell_size 2.0 (inv = 0.5, exact).
    ((x * 0.5).floor() as i64, (y * 0.5).floor() as i64)
}

#[pymethods]
impl GridView {
    #[new]
    fn new(w: usize, h: usize, tiles: Vec<u8>, fine: Vec<u8>, any_t: Vec<u8>,
           dirs: Vec<(f64, f64)>) -> PyResult<Self> {
        if tiles.len() != w * h || any_t.len() != w * h || fine.len() != w * h * RES * RES {
            return Err(pyo3::exceptions::PyValueError::new_err("GridView: bad buffer sizes"));
        }
        Ok(GridView { w, h, tiles, fine, any_t, dirs })
    }

    #[pyo3(signature = (px, py, bb=false))]
    fn is_walkable_point(&self, px: f64, py: f64, bb: bool) -> bool {
        self.walkable_point(px, py, bb)
    }

    fn shape_blocked(&self, px: f64, py: f64) -> bool {
        self.blocked(px, py)
    }

    fn shape_near(&self, px: f64, py: f64, r: f64) -> bool {
        self.near(px, py, r)
    }

    #[pyo3(signature = (ax, ay, bx, by, bb=false))]
    fn line_of_sight(&self, ax: f64, ay: f64, bx: f64, by: f64, bb: bool) -> bool {
        self.los(ax, ay, bx, by, bb)
    }

    fn keep_off_walls(&self, px: f64, py: f64, r: f64, bb: bool) -> (f64, f64) {
        self.keep_off(px, py, r, bb)
    }

    /// movement.resolve_collisions (units already filtered: not in the air).
    /// Returns the new positions; the caller writes back the changed ones.
    fn resolve_collisions(&self, mut xs: Vec<f64>, mut ys: Vec<f64>, rs: Vec<f64>,
                          ids: Vec<i64>, bbs: Vec<bool>, iterations: usize)
                          -> (Vec<f64>, Vec<f64>) {
        let n = xs.len();
        let kr: Vec<f64> = rs.iter().map(|&r| if WALL_R_MAX < r { WALL_R_MAX } else { r }).collect();
        let mut cells: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        let mut nb: Vec<usize> = Vec::with_capacity(64);
        let mut seen: HashSet<(i64, i64)> = HashSet::new();
        for _ in 0..iterations {
            for v in cells.values_mut() {
                v.clear();
            }
            for i in 0..n {
                cells.entry(cell_key(xs[i], ys[i])).or_default().push(i);
            }
            seen.clear();
            for a in 0..n {
                let (kx, ky) = cell_key(xs[a], ys[a]);
                nb.clear();
                for dy in -1..=1i64 {
                    for dx in -1..=1i64 {
                        if let Some(b) = cells.get(&(kx + dx, ky + dy)) {
                            nb.extend_from_slice(b);
                        }
                    }
                }
                for j in 0..nb.len() {
                    let b = nb[j];
                    if b == a {
                        continue;
                    }
                    let pair = if ids[a] < ids[b] { (ids[a], ids[b]) } else { (ids[b], ids[a]) };
                    if !seen.insert(pair) {
                        continue;
                    }
                    let ddx = xs[b] - xs[a];
                    let ddy = ys[b] - ys[a];
                    let dist = hypot(ddx, ddy);
                    let min_dist = rs[a] + rs[b];
                    if dist >= min_dist {
                        continue;
                    }
                    if dist < 1e-6 {
                        let px = min_dist * 0.5;
                        self.move_out(&mut xs, &mut ys, a, -px, -0.0, kr[a], bbs[a]);
                        self.move_out(&mut xs, &mut ys, b, px, 0.0, kr[b], bbs[b]);
                        continue;
                    }
                    let overlap = (min_dist - dist) * 0.5;
                    // Vec2.normalized (dist >= 1e-6 > 1e-9 here).
                    let (nx, ny) = (ddx / dist, ddy / dist);
                    let neg = -overlap;
                    self.move_out(&mut xs, &mut ys, a, nx * neg, ny * neg, kr[a], bbs[a]);
                    self.move_out(&mut xs, &mut ys, b, nx * overlap, ny * overlap, kr[b], bbs[b]);
                }
            }
        }
        (xs, ys)
    }
}

#[pymodule]
fn pokemoba_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_hypot, m)?)?;
    m.add_class::<GridView>()?;
    Ok(())
}
