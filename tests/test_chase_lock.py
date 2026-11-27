"""Aggancio del bersaglio: un ordine di movimento lo molla davvero (da fermo non si
reinsegue), e un bersaglio irraggiungibile non tiene il Pokemon incollato al muro."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_basic_attack import DT, TED, TestBasicAttack, spawn   # noqa: E402
from engine.vec import Vec2   # noqa: E402
from sim.commands import AttackCommand, AttackMoveCommand, MoveCommand   # noqa: E402
from sim.simulation import Simulation   # noqa: E402


class TestChaseLock(unittest.TestCase):
    # stesse impalcature dei test dell'attacco base, senza ereditarne i test
    foe = TestBasicAttack.foe
    vis = TestBasicAttack.vis
    step = TestBasicAttack.step

    def setUp(self):
        TestBasicAttack.setUp(self)
        self.sim = Simulation(self.world, DT, self.match)
        self.p = spawn(self.world, self.match, TED, Vec2(27.5, 47.5), 0)
        self.t = self.foe(Vec2(30.5, 47.5))      # un po' oltre la portata di Teddiursa

    def run_ticks(self, n):
        for _ in range(n):
            self.sim.step()
            self.vis()

    def test_move_order_drops_lock_and_idle_does_not_rechase(self):
        self.sim.queue_command(AttackCommand(seq=1, entity_id=self.p.id, target_id=self.t.id))
        self.run_ticks(10)
        self.assertEqual(self.p.order, "attack")
        self.sim.queue_command(MoveCommand(seq=2, entity_id=self.p.id, x=24.5, y=47.5))
        self.run_ticks(120)
        self.assertEqual(self.p.order, "idle")
        x = self.p.pos.x
        self.run_ticks(60)
        self.assertAlmostEqual(self.p.pos.x, x, places=3)   # fermo: niente re-lock

    def test_unreachable_target_is_given_up(self):
        for y in range(len(self.world.grid.tiles)):
            for x in range(29, 40):
                self.world.grid.tiles[y][x] = 1        # muro da bordo a bordo: irraggiungibile
        self.t.pos = Vec2(34.5, 47.5)
        self.sim.queue_command(AttackCommand(seq=1, entity_id=self.p.id, target_id=self.t.id))
        self.run_ticks(int(4.0 / DT))
        self.assertEqual(self.p.order, "idle")             # ha rinunciato
        x = self.p.pos.x
        self.run_ticks(60)
        self.assertAlmostEqual(self.p.pos.x, x, places=2)  # e non lo reinsegue

    # --- inseguimenti legittimi: l'ordine di attacco NON va annullato --------------
    def attack(self):
        self.sim.queue_command(AttackCommand(seq=1, entity_id=self.p.id, target_id=self.t.id))

    def flee(self, factor, sec):
        """Il bersaglio scappa verso destra a `factor` volte la velocita' di chi insegue."""
        self.t.pos = Vec2(31.5, 47.5)
        self.attack()
        x = self.t.pos.x
        for _ in range(int(sec / DT)):
            x += self.p.speed * factor * DT
            self.t.pos = Vec2(x, 47.5)
            self.sim.step()
            self.vis()
            self.assertEqual(self.p.order, "attack")
        self.assertGreater(self.p.pos.x, 27.5 + 0.8 * self.p.speed * factor * sec)

    def test_chase_target_fleeing_at_same_speed(self):
        self.flee(1.0, 5.0)

    def test_chase_slightly_slower_target(self):
        self.flee(0.9, 7.0)

    def test_chase_slower_target_between_hits(self):
        self.flee(0.7, 7.0)

    def test_chase_goes_around_a_wall(self):
        self.t.pos = Vec2(33.5, 47.5)
        for y in range(41, 54):
            self.world.grid.tiles[y][30] = 1               # per passare si gira attorno
        self.attack()
        for _ in range(int(8.0 / DT)):
            self.sim.step()
            self.vis()
            self.assertEqual(self.p.order, "attack")
            if self.match._within_attack(self.p, self.t):
                return
        self.fail("non e' arrivato a portata")

    def test_root_does_not_cancel_the_attack_order(self):
        self.t.pos = Vec2(34.5, 47.5)
        self.attack()
        self.run_ticks(2)
        self.match._apply_root(self.p, 2.5)                # paralisi piu' lunga dello stallo
        self.run_ticks(int(2.4 / DT))
        self.assertEqual(self.p.order, "attack")
        self.run_ticks(int(2.0 / DT))
        self.assertEqual(self.p.order, "attack")
        self.assertTrue(self.match._within_attack(self.p, self.t))

    def test_heavy_slow_does_not_cancel_the_attack_order(self):
        self.t.pos = Vec2(40.5, 47.5)
        self.attack()
        self.p.slow_mult, self.p.slow_until = 0.1, 1e9
        self.run_ticks(int(4.0 / DT))
        self.assertEqual(self.p.order, "attack")

    def test_attackmove_away_ignores_enemy_behind(self):
        self.t.pos = Vec2(25.5, 47.5)                      # alle spalle, a portata
        self.sim.queue_command(AttackMoveCommand(seq=1, entity_id=self.p.id, x=45.5, y=47.5))
        x0 = self.p.pos.x
        self.run_ticks(30)
        self.assertGreater(self.p.pos.x, x0 + 2.0)         # se ne va, niente muro invisibile
        self.assertIsNone(self.p.target_id)

    def test_attackmove_chase_has_a_leash(self):
        self.sim.queue_command(AttackMoveCommand(seq=1, entity_id=self.p.id, x=90.5, y=47.5))
        dropped = False
        for _ in range(int(8.0 / DT)):                     # il bersaglio scappa sempre avanti
            self.t.pos = Vec2(self.p.pos.x + 3.0, 47.5)
            self.sim.step()
            self.vis()
            dropped = dropped or self.match._chase_skipped(self.p) == self.t.id
        self.assertTrue(dropped)                           # a guinzaglio finito lo molla

    def test_target_blinking_in_fog_does_not_make_it_oscillate(self):
        """Il bersaglio sparisce/riappare (cespuglio, giungla): niente avanti-indietro
        fra lui e il punto cliccato, e nella nebbia non viene colpito."""
        self.t.pos = Vec2(31.5, 45.0)
        real = self.match.vision.is_enemy_visible
        self.tick = 0
        self.match.vision.is_enemy_visible = (
            lambda team, eid: real(team, eid) and (self.tick // 4) % 2 == 0)
        self.sim.queue_command(AttackMoveCommand(seq=1, entity_id=self.p.id, x=40.5, y=49.5))
        xs, hits = [], 0
        for _ in range(90):
            self.tick += 1
            was = self.p.aa_active
            self.sim.step()
            self.vis()
            xs.append((self.p.pos.x, self.p.pos.y))
            if self.p.aa_active and not was and (self.tick // 4) % 2 == 1:
                hits += 1                                  # attacco iniziato nella nebbia
        self.assertEqual(hits, 0)
        back = sum(1 for a, b in zip(xs, xs[1:]) if b[0] < a[0] - 1e-6)
        self.assertEqual(back, 0)                          # mai indietro

    def test_astar_chase_never_walks_back_to_passed_waypoints(self):
        for y in range(40, 56):
            for x in range(33, 36):
                self.world.grid.tiles[y][x] = 1            # muro: bisogna girarci attorno
        self.t.pos = Vec2(40.5, 47.5)
        self.sim.queue_command(AttackCommand(seq=1, entity_id=self.p.id, target_id=self.t.id))
        pts = []
        for _ in range(int(6.0 / DT)):
            self.sim.step()
            self.vis()
            pts.append((self.p.pos.x, self.p.pos.y))
        import math
        d0 = math.hypot(pts[0][0] - 33.0, pts[0][1] - 40.0)
        back = 0
        for a, b, c in zip(pts, pts[3:], pts[6:]):         # inversioni nette di direzione
            u = (b[0] - a[0], b[1] - a[1]); v = (c[0] - b[0], c[1] - b[1])
            nu, nv = math.hypot(*u), math.hypot(*v)
            if nu > 0.05 and nv > 0.05 and (u[0] * v[0] + u[1] * v[1]) / (nu * nv) < -0.5:
                back += 1
        self.assertEqual(back, 0)


if __name__ == "__main__":
    unittest.main()
