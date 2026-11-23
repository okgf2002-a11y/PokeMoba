"""Avanzamento lungo il percorso e collisioni a cerchio fra unità.

Funzioni pure che operano su Unit + Grid. Nessuno stato globale.
"""
from __future__ import annotations

import math

from .entities import Unit
from .grid import TILE_BUSH, TILE_PILLAR, Grid
from .spatial import SpatialHash
from .vec import Vec2, direction_index

# Distanza entro cui un waypoint si considera raggiunto.
_ARRIVE_EPS = 0.02
# Aggiramento dei pilastri temporanei: angoli provati (gradi) attorno alla
# direzione di marcia quando il passo entrerebbe in un pilastro.
_STEER_ANGLES = (35, -35, 70, -70, 100, -100)


def _in_pillar(grid: Grid | None, p: Vec2) -> bool:
    return grid is not None and grid.tile_at(math.floor(p.x), math.floor(p.y)) == TILE_PILLAR


def _steer(grid: Grid, pos: Vec2, step: Vec2) -> Vec2 | None:
    """Passo deviato per aggirare un pilastro (che compare dopo il calcolo del
    percorso, o sulla traiettoria diretta di minion/bot): prova direzioni
    ruotate e prende la prima libera. None = bloccato."""
    for deg in _STEER_ANGLES:
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        alt = pos + Vec2(step.x * ca - step.y * sa, step.x * sa + step.y * ca)
        if not _in_pillar(grid, alt) and grid.is_walkable_point(alt.x, alt.y):
            return alt
    return None


def _nearest_walkable(grid: Grid, pos: Vec2, block_bush: bool) -> Vec2 | None:
    """Centro del tile calpestabile piu' vicino a pos (ricerca ad anelli)."""
    cx, cy = math.floor(pos.x), math.floor(pos.y)
    best, best_d = None, 1e18
    for r in range(1, 8):
        for ty in range(cy - r, cy + r + 1):
            for tx in range(cx - r, cx + r + 1):
                if max(abs(tx - cx), abs(ty - cy)) != r or not grid.is_walkable(tx, ty, block_bush):
                    continue
                d = (tx + 0.5 - pos.x) ** 2 + (ty + 0.5 - pos.y) ** 2
                if d < best_d:
                    best, best_d = Vec2(tx + 0.5, ty + 0.5), d
        if best is not None:
            return best
    return None


def _slide(grid: Grid, pos: Vec2, nxt: Vec2, block_bush: bool) -> Vec2 | None:
    """Passo pos->nxt senza entrare nei tile bloccanti: se il passo diretto e'
    bloccato prova a scivolare lungo un solo asse (spigoli e pareti). Anche il
    passo libero e' verificato a meta' per non tagliare gli angoli."""
    ok = grid.is_walkable_point
    if ok(nxt.x, nxt.y, block_bush):
        # Diagonale fra due tile bloccanti: passa solo se almeno un lato e' libero.
        if (int(math.floor(pos.x)) == int(math.floor(nxt.x))
                or int(math.floor(pos.y)) == int(math.floor(nxt.y))
                or ok(nxt.x, pos.y, block_bush) or ok(pos.x, nxt.y, block_bush)):
            return nxt
    ax, ay = Vec2(nxt.x, pos.y), Vec2(pos.x, nxt.y)
    # prima l'asse col movimento maggiore
    cand = (ax, ay) if abs(nxt.x - pos.x) >= abs(nxt.y - pos.y) else (ay, ax)
    for c in cand:
        if (c.x != pos.x or c.y != pos.y) and ok(c.x, c.y, block_bush):
            return c
    return None


def advance_unit(unit: Unit, dt: float, grid: Grid | None = None) -> None:
    """Muove l'unità lungo i suoi waypoint per un intervallo dt.

    Consuma più waypoint se dt è grande. Aggiorna `facing` e `moving`. Se
    l'unità ha `bush_blocks` (minion) e il prossimo passo entrerebbe in un
    cespuglio, il movimento si ferma al bordo invece di attraversarlo.
    """
    block_bush = grid is not None and getattr(unit, "bush_blocks", False)
    # Anti-incastro: chi si ritrova col centro dentro un muro (scatto, spinta,
    # cambio di meteo) viene riportato sul tile calpestabile piu' vicino.
    # In volo (air_z > 0) i muri si scavalcano.
    if (grid is not None and getattr(unit, "air_z", 0.0) <= 0.0
            and not grid.is_walkable_point(unit.pos.x, unit.pos.y, block_bush)):
        fix = _nearest_walkable(grid, unit.pos, block_bush)
        if fix is not None:
            unit.pos = fix

    if not unit.path:
        unit.moving = False
        if grid is not None and getattr(unit, "air_z", 0.0) <= 0.0:
            _keep_off_walls(unit, grid, block_bush)
        return

    airborne = getattr(unit, "air_z", 0.0) > 0.0

    def _blocked(p) -> bool:
        return (grid is not None and not airborne
                and not grid.is_walkable_point(p.x, p.y, block_bush=block_bush))

    budget = unit.speed * dt
    pos = unit.pos
    while budget > 1e-9 and unit.path:
        target = unit.path[0]
        to = target - pos
        d = to.length()
        # Pilastro di roccia sulla traiettoria: lo aggira deviando il passo (o si
        # ferma se non c'è spazio). Un waypoint dentro il pilastro non si raggiunge.
        if d > _ARRIVE_EPS and grid is not None:
            step = to.normalized() * min(budget, d)
            if _in_pillar(grid, pos + step) or _in_pillar(grid, target):
                alt = _steer(grid, pos, step) if _in_pillar(grid, pos + step) else pos + step
                if alt is None or _blocked(alt):
                    unit.facing = direction_index(to.x, to.y)
                    break
                pos = alt
                unit.facing = direction_index(to.x, to.y)
                budget = 0.0
                break
        if d <= _ARRIVE_EPS:
            if _blocked(target):
                break
            pos = target
            unit.path.pop(0)
            continue
        if d <= budget:
            if _blocked(target):
                break
            if grid is not None and not airborne:
                moved = _slide(grid, pos, target, block_bush)
                if moved is None:
                    break
                if moved is not target:      # scivolato: waypoint non raggiunto
                    pos = moved
                    unit.facing = direction_index(to.x, to.y)
                    budget = 0.0
                    break
            pos = target
            budget -= d
            unit.facing = direction_index(to.x, to.y)
            unit.path.pop(0)
        else:
            step = to.normalized() * budget
            nxt = pos + step
            if grid is not None and not airborne:
                nxt = _slide(grid, pos, nxt, block_bush)
            if nxt is None:
                unit.facing = direction_index(to.x, to.y)
                budget = 0.0
                break
            pos = nxt
            unit.facing = direction_index(to.x, to.y)
            budget = 0.0

    unit.pos = pos
    if grid is not None and not airborne:
        _keep_off_walls(unit, grid, block_bush)
    unit.moving = bool(unit.path)


# Raggio massimo con cui il corpo di un'unita' tiene le distanze dai muri: sotto
# 0.5 per poter passare nei varchi larghi un tile.
_WALL_R_MAX = 0.42


# Direzioni e frazioni del raggio con cui si campiona la sagoma disegnata dei
# muri attorno al corpo (dal bordo verso il centro).
_SHAPE_DIRS = tuple((math.cos(i * math.pi / 8), math.sin(i * math.pi / 8))
                    for i in range(16))
_SHAPE_FRACS = (0.2, 0.4, 0.6, 0.8, 1.0)


def _keep_off_walls(unit: Unit, grid: Grid, block_bush: bool) -> None:
    """Il corpo (cerchio) non entra nei muri: la collisione col solo centro
    faceva sporgere lo sprite dentro i muri e incastrare negli spigoli. Spinge il
    centro fuori, accettando solo posizioni calpestabili. Con la sagoma dei muri
    attiva (mappe di partita) i muri contano dove sono DISEGNATI; i tile
    (cerchio contro rettangolo) restano per pilastri, cespugli e bordo mappa."""
    r = min(getattr(unit, "radius", 0.35), _WALL_R_MAX)
    px, py = unit.pos.x, unit.pos.y
    shape = grid._shape_on
    if shape:
        nv = grid.native_view()
        if nv is not None:                   # stesso calcolo, in Rust (pokemoba_native)
            qx, qy = nv.keep_off_walls(px, py, r, block_bush)
            if qx != px or qy != py:
                unit.pos = Vec2(qx, qy)
            return
    cx, cy = math.floor(px), math.floor(py)
    for _ in range(2):
        moved = False
        for ty in range(cy - 1, cy + 2):
            for tx in range(cx - 1, cx + 2):
                if shape:
                    if grid.in_bounds(tx, ty):
                        t = grid.tiles[ty][tx]
                        if not (t == TILE_PILLAR or (block_bush and t == TILE_BUSH)):
                            continue
                elif grid.is_walkable(tx, ty, block_bush):
                    continue
                qx = min(max(px, tx), tx + 1)
                qy = min(max(py, ty), ty + 1)
                dx, dy = px - qx, py - qy
                d2 = dx * dx + dy * dy
                if d2 >= r * r or d2 < 1e-12:
                    continue
                d = math.sqrt(d2)
                k = (r - d) / d
                nx, ny = px + dx * k, py + dy * k
                if grid.is_walkable_point(nx, ny, block_bush):
                    px, py = nx, ny
                    moved = True
        if shape and grid.shape_near(px, py, r):
            p = _shape_push(grid, px, py, r)
            if p is not None and grid.is_walkable_point(p[0], p[1], block_bush):
                px, py = p
                moved = True
        if not moved:
            break
    if px != unit.pos.x or py != unit.pos.y:
        unit.pos = Vec2(px, py)


def _shape_push(grid: Grid, px: float, py: float, r: float):
    """Spinta che porta il cerchio (px, py, r) fuori dalla sagoma disegnata dei
    muri: per ogni direzione la frazione di raggio piu' interna dentro il muro
    da' la profondita'; si spinge all'opposto della somma pesata, di quanto
    entra il punto piu' profondo. None se il cerchio non tocca la sagoma o se il
    centro e' gia' dentro (ci pensa l'anti-incastro di advance_unit)."""
    if grid.shape_blocked(px, py):
        return None
    sx = sy = 0.0
    deep = 0.0
    for c, s in _SHAPE_DIRS:
        for f in _SHAPE_FRACS:
            if grid.shape_blocked(px + c * r * f, py + s * r * f):
                d = r * (1.0 - f) + 0.2 * r
                sx -= c * d
                sy -= s * d
                if d > deep:
                    deep = d
                break
    if deep <= 0.0:
        return None
    n = math.hypot(sx, sy)
    if n < 1e-9:
        return None
    return px + sx / n * deep, py + sy / n * deep


def resolve_collisions(units: list[Unit], grid: Grid, iterations: int = 2,
                       spatial: SpatialHash | None = None) -> None:
    """Separa le unità che si sovrappongono (collisione a cerchio).

    Usa una griglia spaziale per confrontare solo le unità vicine invece di
    tutte le coppie. Per ogni coppia troppo vicina le spinge via a metà strada,
    senza farle finire dentro i muri.
    """
    if spatial is None:
        spatial = SpatialHash(cell_size=2.0)
    # Chi e' in volo (air_z > 0: lanciato in aria, salti, Taxi Volante) passa
    # sopra le altre unita': non spinge e non viene spinto.
    units = [u for u in units if getattr(u, "air_z", 0.0) <= 0.0]
    nv = grid.native_view() if grid is not None else None
    if nv is not None:
        # Stesso algoritmo (griglia spaziale da 2 tile, stesso ordine di visita e
        # di spinta), in Rust: si riscrivono solo le posizioni cambiate.
        xs = [u.pos.x for u in units]
        ys = [u.pos.y for u in units]
        nxs, nys = nv.resolve_collisions(
            xs, ys, [u.radius for u in units], [u.id for u in units],
            [bool(getattr(u, "bush_blocks", False)) for u in units], iterations)
        for u, x0, y0, x1, y1 in zip(units, xs, ys, nxs, nys):
            if x1 != x0 or y1 != y0:
                u.pos = Vec2(x1, y1)
        return
    for _ in range(iterations):
        spatial.rebuild(units)
        seen: set[tuple[int, int]] = set()
        for a in units:
            for b in spatial.neighbors(a):
                if b is a:
                    continue
                pair = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                if pair in seen:
                    continue
                seen.add(pair)
                delta = b.pos - a.pos
                dist = delta.length()
                min_dist = a.radius + b.radius
                if dist >= min_dist:
                    continue
                if dist < 1e-6:
                    # Sovrapposizione perfetta: spingi su un asse arbitrario.
                    push = Vec2(min_dist * 0.5, 0.0)
                    _move_out(a, Vec2(-push.x, -push.y), grid)
                    _move_out(b, push, grid)
                    continue
                overlap = (min_dist - dist) * 0.5
                dir_ = delta.normalized()
                _move_out(a, dir_ * (-overlap), grid)
                _move_out(b, dir_ * overlap, grid)


def resolve_obstacles(units: list[Unit], obstacles: list, grid: Grid) -> None:
    """Spinge fuori le unità dai cerchi statici (x, y, raggio), es. le torri.

    La spinta è radiale, quindi chi cammina contro l'ostacolo scivola lungo il
    bordo con la componente tangenziale del passo. Se punta dritto al centro
    (componente tangenziale ~0) riceve una piccola spinta laterale verso il lato
    del waypoint, così non resta incastrato. Un waypoint DENTRO l'ostacolo si
    considera raggiunto appena lo si tocca.
    """
    for u in units:
        if getattr(u, "air_z", 0.0) > 0.0:
            continue                    # in volo passa sopra la torre; atterrando viene spinto fuori
        bb = getattr(u, "bush_blocks", False)
        for ox, oy, r in obstacles:
            dx, dy = u.pos.x - ox, u.pos.y - oy
            min_d = r + u.radius
            d2 = dx * dx + dy * dy
            if d2 >= min_d * min_d:
                continue
            d = math.sqrt(d2)
            nx, ny = (dx / d, dy / d) if d > 1e-6 else (0.0, 1.0)
            new = Vec2(ox + nx * min_d, oy + ny * min_d)
            if u.path:
                w = u.path[0]
                if (w.x - ox) ** 2 + (w.y - oy) ** 2 < min_d * min_d:
                    u.path.pop(0)
                if u.path:
                    tx, ty = u.path[0].x - u.pos.x, u.path[0].y - u.pos.y
                    tl = math.hypot(tx, ty)
                    px, py = -ny, nx
                    side = tx * px + ty * py
                    if tl > 1e-6 and abs(side) < 0.3 * tl and tx * nx + ty * ny < 0:
                        sgn = 1.0 if side > 0 or (abs(side) < 1e-6 and u.id % 2) else -1.0
                        new = Vec2(new.x + px * 0.04 * sgn, new.y + py * 0.04 * sgn)
            if not grid.is_walkable_point(new.x, new.y, bb):
                # Torre accanto a un muro: la spinta radiale finirebbe nel muro.
                # Il punto libero piu' vicino FUORI dall'ostacolo (prima sul bordo,
                # poi un po' piu' lontano), cosi' nessuno resta dentro la torre.
                new = _free_outside(grid, ox, oy, min_d, math.atan2(ny, nx), bb,
                                    [o for o in obstacles if o[0] != ox or o[1] != oy])
            if new is not None:
                u.pos = new


def _free_outside(grid: Grid, ox: float, oy: float, min_d: float, ang: float,
                  bb: bool, others: list) -> Vec2 | None:
    """Punto calpestabile a distanza >= min_d da (ox, oy), il piu' vicino
    all'angolo `ang` (direzione della spinta radiale), fuori dagli altri ostacoli."""
    for extra in (0.0, 0.2, 0.45, 0.75, 1.1, 1.5):
        rr = min_d + extra
        for k in range(13):                         # 0, +-15, +-30 ... +-180 gradi
            for sgn in ((1,) if k in (0, 12) else (1, -1)):
                a = ang + sgn * k * (math.pi / 12)
                x, y = ox + math.cos(a) * rr, oy + math.sin(a) * rr
                if not grid.is_walkable_point(x, y, bb):
                    continue
                if any((x - qx) ** 2 + (y - qy) ** 2 < qr * qr for qx, qy, qr in others):
                    continue
                return Vec2(x, y)
    return None


def _move_out(unit: Unit, offset: Vec2, grid: Grid) -> None:
    """Applica uno spostamento di separazione se la destinazione è calpestabile."""
    new_pos = unit.pos + offset
    bb = getattr(unit, "bush_blocks", False)
    if grid.is_walkable_point(new_pos.x, new_pos.y, block_bush=bb):
        unit.pos = new_pos
        if getattr(unit, "air_z", 0.0) <= 0.0:
            _keep_off_walls(unit, grid, bb)
