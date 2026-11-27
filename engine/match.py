"""Regole di gioco della partita (Fase 2).

Opera su un World già popolato dei Pokémon (creati dallo scene/client). Qui
dentro vivono: costruzione delle strutture dalla mappa, ondate di minion, IA di
minion e bot, combattimento, respawn e condizione di vittoria.

Puro: nessun pygame. In Fase 5 girerà identico sul server.

Prestazioni:
  - I minion seguono percorsi fissi (waypoint della lane) e usano A* SOLO quando
    inseguono un bersaglio dietro un ostacolo.
  - Target e vicinato via griglia spaziale, non O(n^2).
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import config
from . import evolution, skills
from .entities import Core, Minion, Pillar, Pokemon, Structure, Unit, Wild
from .grid import TILE_BUSH, TILE_PILLAR, TILE_RIVER, _native
from .moves import Move
from .items import ITEM_SLOTS, ItemCatalog
from .jungle import JungleSystem
from .pathfinding import find_path, plan_move, smooth_path
from .spatial import SpatialHash
from .noivern import ECHO_ID
from .mienshao import PROJ_SHAPES
from .venomoth import MOTH_PASSIVES
from .delibird import DELI_PASSIVES
from .item_effects import (ACRO_SLOW, EVERSTONE_XP, EVERSTONE_XP_LEVEL, EVIOLITE_MULT,
                           SNOWBALL_TEN, ItemEffectsMixin, cdr_to_haste)
from .bot_plan import BotPlanMixin
from .bot_utility import BotUtilityMixin
from .surrender import SurrenderMixin
from .lines import LINE_MIXINS, bot_cast_names, hook_names
from .bot_params import DEFAULT_BOT_PARAMS, BotParams
from .smite import SmiteMixin, smite_damage
from .basic_attack import (ACQUIRE_EXTRA, AMOVE_DROP_EXTRA, AMOVE_EXTRA, AS_CAP,
                           IDLE_CHASE_LEASH, BasicAttackMixin)
from .spiritomb import PASSIVE_ID as SPIRITOMB_PASSIVE
from .stats import (CRIT_MULT, HIGH_CRIT_CHANCE, STAB_MULT, Stats,
                    ability_haste_factor, combine_tenacity as stat_combine_ten,
                    derive, evolution_mult as stat_evo_mult,
                    level_from_xp, mitigate as stat_mitigate,
                    penetrate as stat_penetrate, respawn_time)
from .types import TypeChart
from .vec import DIR_VECTORS, Vec2, direction_index
from .vision import VisionSystem

# Priorità di bersaglio (numero più basso = preferito).
_PRIO = {Minion: 0, Pokemon: 1, Structure: 2}

# Statistiche dei minion per tipo (stile LoL: melee tanky, caster dalla distanza,
# cannone grosso e lento). Valori greybox da bilanciare.
MINION_STATS = {
    "melee":  dict(hp=170, attack_damage=12, attack_range=1.1, attack_cooldown=1.0,
                   speed=3.1, radius=0.28, gold=20, xp=60),
    "caster": dict(hp=90,  attack_damage=18, attack_range=3.4, attack_cooldown=1.3,
                   speed=3.1, radius=0.24, gold=15, xp=30),
    "cannon": dict(hp=460, attack_damage=34, attack_range=3.0, attack_cooldown=1.6,
                   speed=2.9, radius=0.34, gold=50, xp=95),
}

_WAYPOINT_REACHED = 0.6
_TOWER_AGGRO_WINDOW = 4.0   # s: finestra in cui una torre può colpire un Pokémon
_UNIT_OR_STRUCT = (Unit, Structure)          # chiave della cache di World._of_type
_CLEANUP_TYPES = (Minion, Wild, Structure)   # tipi tolti dal mondo alla morte
_MINION_LEASH = 5.5         # tile: distanza max dalla lane prima di tornare indietro
_MINION_AGGRO_WINDOW = 4.0  # s: i minion focalizzano il campione che attacca un alleato
_MINION_RETALIATE = 2.5     # s: finestra per ritorsione contro chi li ha colpiti
XP_RADIUS = 8.0             # tile: raggio entro cui i nemici ricevono esperienza
# Livello a cui si sblocca la SCELTA del potenziamento di ogni slot (scaglionata).
SLOT_UPGRADE_LEVEL = {"Q": 5, "W": 6, "E": 7, "R": 8}

# --- Forma Shedinja (passiva Muta / R Rancore, linea Nincada) ----------------
# Statistiche base REALI dello Shedinja (HP a 1 FISSO: non derivato). Coleottero/
# Spettro. La forma dura MUTA_DURATION; la passiva/ult condividono la ricarica.
SHEDINJA_BASE = {"hp": 1, "attack": 90, "defense": 45,
                 "sp_attack": 30, "sp_defense": 30, "speed": 40}
SHEDINJA_TYPES = ["Bug", "Ghost"]
SHEDINJA_DEX = 292
# Tipi superefficaci contro Coleottero/Spettro: bucano la Magidifesa (letali).
_SHEDINJA_WEAK = {"Fire", "Flying", "Rock", "Ghost", "Dark"}
FORZABRUTA_WILD_MULT = 1.30   # ForzaBruta (Totodile): danno ai neutrali
def _src_name(u) -> str:
    """Nome leggibile di chi infligge danno (riepilogo della morte)."""
    if isinstance(u, Pokemon):
        return u.name
    if isinstance(u, Core):
        return "Nucleo"
    if isinstance(u, Structure):
        return "Torre"
    if isinstance(u, Minion):
        return "Minion"
    if isinstance(u, Wild):
        return "Leggendario" if getattr(u, "kind", "") in ("meteo", "arceus") else "Mostro"
    return "?"


RECAP_WINDOW = 10.0     # secondi di danni nel riepilogo della morte
MUTA_WILD_MULT = 1.30    # danno ai Pokémon neutrali (come Miele/Scalaroccia)
MUTA_WILD_MULT_LARVA = 1.70   # da Nincada (stage 0, fino al liv.8): Attacco base 45
MUTA_DURATION = 5.0       # s in forma Shedinja
MUTA_COOLDOWN = 150.0      # ricarica condivisa Muta/Rancore
MUTA_REVERT_HP = 0.30      # Muta: torna Ninjask col 30% degli HP max
RANCORE_REVIVE_HP = 0.10   # Rancore: rivive Ninjask col 10% degli HP max

# --- linea Poliwag: pioggia, aura Piovischio, Roccia di Re --------------------
PIOVISCHIO_RADIUS = 4.0       # raggio dell'aura di pioggia permanente di Politoad
WATER_BOOST = 1.2             # mosse Acqua alleate in pioggia (Piovischio/Pioggiadanza): +20%
FIRE_REDUCE = 0.8             # mosse Fuoco nemiche dentro l'aura Piovischio: -20%
NUOTOVELOX_SPEED = 1.2        # +20% velocità in acqua/pioggia (Poliwag/Poliwhirl)
KINGROCK_SHIELD_FRAC = 0.30   # Roccia di Re: scudo pari al 30% della cura data a un alleato
KINGROCK_SHIELD_DUR = 3.0

# --- linea Roggenrola: Sabbiafiume, pilastri di roccia, lancio in aria -------
SABBIAFIUME_RADIUS = 3.0      # raggio della tempesta di sabbia permanente attorno a Gigalith
SABBIAFIUME_TICK = 0.5        # s tra due tick di danno della tempesta
SABBIAFIUME_DPS_BASE = 8.0    # danno al secondo (senza tipo) = base + per_livello × livello
SABBIAFIUME_DPS_LVL = 2.5     # (liv.8 ~28/s: poco sopra la rigenerazione passiva, resta piccolo)
PILLAR_DURATION = 4.0         # durata dei pilastri di roccia (Rocciotomba)
SHOCKWAVE_RADIUS = 2.2        # Pesobomba: raggio dell'onda d'urto contro muro/pilastro
SHOCKWAVE_AIR = 1.0           # Pesobomba: durata del lancio in aria dell'onda d'urto


def _prio(entity) -> int:
    for cls, p in _PRIO.items():
        if isinstance(entity, cls):
            return p
    return 9


# Stati che Sincronismo (Espeon) rimanda a chi glieli applica.
_SYNC_STATUSES = ("slow", "stun", "freeze", "confuse", "silence", "root", "sleep",
                  "poison", "fear", "dashlock", "atk_debuff", "def_debuff",
                  "spdef_debuff", "blind")



# Animazione dello sprite al lancio, per forma della mossa (modificabile per
# Pokémon e mossa dall'editor delle animazioni: game/anim_tuning.py).
CAST_ANIMS = {"projectile": "Shoot", "bubble": "Shoot",
            "cone": "Attack", "dash": "Attack", "bolt": "Shoot",
            "curse": "Shoot", "fan": "Shoot", "dashback": "Attack",
            "rain": "Shoot", "storm": "Shoot", "leap": "Attack",
            "arc": "Shoot", "throwback": "Attack", "raindance": "Charge",
            "healzone": "Charge", "flamethrower": "Shoot", "smoke": "Charge",
            "fly": "Charge", "rockwave": "Attack", "smackdown": "Attack",
            "sandboost": "Charge", "spikes": "Shoot", "rockcharge": "Attack",
            "avalanche": "Shoot", "explosion": "Charge",
            "spiritbeam": "Shoot", "painsplit": "Charge", "legion": "Charge",
            "memento": "Charge", "wish": "Charge", "endure": "Charge",
            "thunder": "Shoot", "blizzard": "Shoot", "futuresight": "Charge",
            "moonlight": "Charge", "gleam": "Charge",
            "acidarmor": "Charge", "defog": "Shoot", "tailwind": "Shoot",
            "roost": "Charge", "taxi": "Charge",
            "nightshade": "Shoot", "clearsmog": "Shoot", "armorcannon": "Charge",
            "floatarmor": "Charge", "psychic": "Shoot", "grudgeblade": "Attack",
            "shadowclaw": "Charge", "dragonpulse": "Shoot", "whirlwind": "Charge",
            "hurricane": "Shoot", "dracometeor": "Charge",
            "honeyscent": "Shoot", "rest": "Sleep", "earthquake": "Rumble",
            "nightslash": "Attack", "nightfall": "Charge", "beatup": "Attack",
            "sludgelob": "Shoot", "detect": "Charge", "fakeout": "Strike",
            "sleevedance": "Twirl", "reversal": "Attack",
            "flashcannon": "Shoot", "thunderwave": "Shoot", "dragontail": "Swing",
            "metalsound": "Charge", "irondefense": "Charge", "electroshot": "Charge",
            "thunderstorm": "Shoot",
            "psyburst": "Shoot", "psybeam": "Shoot", "hattvoice": "Swing",
            "moonorb": "Shoot", "trickroom": "Charge", "witchwave": "Charge",
            "magbolt": "Shoot", "magfield": "SpAttack", "voltdash": "Charge",
            "magpillar": "SpAttack", "magterrain": "Charge",
            "icetrail": "Charge", "takedown": "Attack", "icefang": "Attack",
            "ancientpower": "Shoot", "iceage": "Shoot",
            "mothbeam": "Shoot", "powdercloud": "Shoot", "mothgust": "Swing",
            "quiverdance": "Rotate", "powderstorm": "Charge",
            "octishot": "Shoot", "ammoswap": "Rotate", "octazooka": "Shoot",
            "octibeam": "Charge", "octigrab": "Strike",
            "florseed": "Rotate", "florheal": "Charge", "florstorm": "Rotate",
            "florgarden": "Charge", "florsolar": "Charge",
            "scyfury": "Attack", "scyfocus": "Charge", "scywing": "Attack",
            "deligift": "Shoot", "delispikes": "Shoot", "delibeam": "Shoot",
            "delidash": "Charge", "deliparadox": "Charge", "delidelivery": "Shoot",
            "scybullet": "Attack", "scysuperpower": "Attack", "scyclaws": "Charge",
            "kleaaxe": "Attack", "kleachop": "Charge", "kleasmash": "Charge",
            "incinkick": "Double", "incinfang": "Attack", "incincross": "Swing",
            "incinbully": "Charge", "incinropes": "Hop", "incinlariat": "Rotate",
            "incingrab": "Swing"}

class Match(*LINE_MIXINS, BasicAttackMixin, SmiteMixin, ItemEffectsMixin, BotPlanMixin,
            BotUtilityMixin, SurrenderMixin):
    def __init__(self, world, map_data):
        self.world = world
        self.map = map_data
        self.time = 0.0
        self.winner: int | None = None
        self._first_blood_done = False   # prima uccisione della partita (punto 4)
        self._dmg_label = None           # nome della fonte del danno in corso (riepilogo morte)
        self.announcements: list[tuple[float, str]] = []  # (tempo, testo) per l'HUD
        self._wave_timer = config.FIRST_WAVE_TIME
        self._wave_number = 0
        self._spawn_queue: list[tuple[float, int, str]] = []  # (due, team, lane)
        self._combat_hash = SpatialHash(cell_size=3.0)
        self._ai: dict[int, dict] = {}  # stato IA per unità (indice waypoint, cooldown A*)
        # IA dei minion in Rust (stessa logica di _ai_minion), se c'e' l'estensione.
        self._minion_ai = None
        if _native is not None and hasattr(_native, "MinionAI"):
            self._minion_ai = _native.MinionAI(
                Pokemon, Minion, Structure, Vec2, _WAYPOINT_REACHED, _MINION_LEASH,
                _MINION_AGGRO_WINDOW, _MINION_RETALIATE, config.TICK_DT)
        # Aiutanti dell'IA dei bot in Rust (stesse scansioni), solo con l'hash nativo.
        self._bot_nat = None
        if _native is not None and hasattr(_native, "BotHelpers")                 and isinstance(self._combat_hash, _native.SpatialHash):
            self._bot_nat = _native.BotHelpers(Pokemon, Minion)
        # Parametri dell'IA dei bot, uno per squadra (engine/bot_params.py).
        self.bot_params: list[BotParams] = [BotParams(), BotParams()]
        self._move_resume: dict[int, tuple] = {}  # ultimo ricalcolo di un ordine di movimento (dest, pos)
        # Bersaglio che il giocatore ha mollato di proposito (ordine di movimento) o
        # che non si riesce a raggiungere: da fermo non lo si reinsegue per un po'.
        self._chase_skip: dict[int, tuple] = {}   # id Pokemon -> (id bersaglio, fino a)
        self._lock_seen: dict[int, tuple] = {}    # id Pokemon -> (id bersaglio, ultima volta visto)
        self._chase_prog: dict[int, list] = {}    # id Pokemon -> stato di _chase_stalled
        data_dir = Path(__file__).resolve().parent.parent / "data"
        self.types = TypeChart.load(data_dir / "types.json")
        self.item_catalog = ItemCatalog.load(data_dir / "items.json")
        self.fx_events: list[dict] = []   # effetti visivi per il client (cosmetici)
        self.projectiles: list[dict] = []  # proiettili che viaggiano (es. bolla Idropulsar)
        self.area_effects: list[dict] = []  # aree persistenti (es. Pioggia di Frecce)
        self.rain_zones: list[dict] = []    # zone di pioggia a tempo (Pioggiadanza): boost Acqua + Nuotovelox
        self.air_landings: list[dict] = []  # Rimbalzo: atterraggi programmati (salto in aria)
        self.tosses: list[dict] = []        # Ribaltiro: nemici scaraventati in arco (animazione)
        self.channels: list[dict] = []      # canalizzazioni (Lanciafiamme continuo, Volo, Esplosione)
        # Linea Roggenrola: onde di Rocciotomba, cariche di Pesobomba, valanghe di
        # Frana (si muovono tick per tick) e zone Abbattimento (bloccano gli scatti).
        self.rock_waves: list[dict] = []
        self.rock_charges: list[dict] = []
        self.avalanches: list[dict] = []
        self.smackdown_zones: list[dict] = []
        self.item_clouds: list[dict] = []   # nubi della Palla Fumo (oggetto attivabile)
        self.sun_zones: list[dict] = []     # aree di sole locale (Rocciacalda)
        for name in hook_names("init"):   # linee con logica propria (engine/lines.py)
            getattr(self, name)()
        self._init_basic_attack()   # attacco base: proiettili a ricerca
        self.vision = VisionSystem(self.world.grid)
        self._build_structures()
        self._tower_lanes = {}
        for e in self.world.structures():
            if e.kind == "tower" and e.lane:
                self._tower_lanes.setdefault(e.team, set()).add(e.lane)
        self.jungle = JungleSystem(self)

    # --- costruzione ---------------------------------------------------------
    def _build_structures(self):
        for s in self.map.structures:
            x, y = s["pos"]
            pos = Vec2(x + 0.5, y + 0.5)
            team = s["team"]
            if s["type"] == "core":
                core = self.world.spawn_core(pos, team, hp=config.CORE_HP)
                setattr(core, "color", config.TEAM_COLORS[team])
            else:
                tier = s.get("tier", 1)
                hp = config.TOWER_HP_INNER if tier >= 2 else config.TOWER_HP_OUTER
                tw = self.world.spawn_tower(pos, team, s.get("lane", "mid"), tier,
                                            hp=hp, attack_damage=config.TOWER_DAMAGE,
                                            attack_range=config.TOWER_RANGE)
                setattr(tw, "color", config.TEAM_COLORS[team])
                setattr(tw, "plates_left", config.PLATE_COUNT)   # placche (punto 6)

    def oriented_lane(self, lane_name: str, team: int) -> list[Vec2]:
        """Waypoint della lane dalla propria base a quella nemica."""
        pts = [Vec2(x, y) for (x, y) in self.map.lanes.get(lane_name, [])]
        # Le lane nel JSON vanno da base0 a base1; la squadra 1 le percorre al contrario.
        return pts if team == 0 else list(reversed(pts))

    # --- ciclo ---------------------------------------------------------------
    def update(self, dt: float, tick: int) -> None:
        if self.winner is not None:
            return
        self.time += dt
        self._tick = tick
        self._surrender_tick()
        if self.winner is not None:
            return
        self._combat_hash.rebuild(self.world.combatants())
        # Visione per entrambe le squadre (ricalcolo solo al cambio tile): la usano
        # i bot per ingaggiare solo nemici realmente visibili, e il client per la
        # nebbia. In Fase 5 il server la userà per filtrare gli snapshot.
        self.vision.update(self.world, 0)
        self.vision.update(self.world, 1)

        self.jungle.update(dt)
        self._update_pillars(dt)     # pilastri di roccia (ri-timbrati dopo un cambio meteo)
        self._update_recall(dt)
        self._update_waves(dt)
        self._update_ai(dt)
        self._update_submerge(dt)    # SUB: riemersione (prima dei cast)
        self._update_air_landings(dt)  # Rimbalzo: atterraggio dopo il salto (prima dei cast)
        self._update_casts(dt)       # esecuzione mosse Q/W/E/R
        self._update_channels(dt)    # canalizzazioni (Lanciafiamme continuo, Volo, Esplosione)
        for name in hook_names("update"):   # linee con logica propria (engine/lines.py)
            getattr(self, name)(dt)
        self._update_rock_moves(dt)  # Rocciotomba / Pesobomba / Frana / Abbattimento
        self._update_projectiles(dt) # proiettili che viaggiano (bolla)
        self._update_area_effects(dt)  # aree persistenti (Pioggia di Frecce, Acquanello, Ultimocanto)
        self._update_rain_zones(dt)    # Pioggiadanza: scadenza delle zone di pioggia a tempo
        self._update_blink_return(dt)  # Spettrotuffo: ritorno automatico
        self._update_curses(dt)        # Maledizione: danno ritardato
        self._update_forms(dt)         # forma Shedinja: ritorno a Ninjask (Muta/Rancore)
        self._update_decoys(dt)        # Doppioteam: scadenza dei cloni esca
        self._update_combat(dt)      # attacchi base
        self._update_status(dt)      # scadenza stati + velocità effettiva + ricariche
        self._update_items(dt)       # oggetti: rigen, Bicicletta, Palla Fumo (dopo la velocità)
        self._update_sandstorms(dt)  # Sabbiafiume: tempesta di sabbia attorno a Gigalith
        self.world.tick(dt)          # movimento + collisioni
        self._update_tosses(dt)      # Ribaltiro: arco del nemico lanciato (dopo le collisioni)
        self._update_mag_drags(dt)   # Magnetismo: trascinamenti e spinte (dopo le collisioni)
        self._update_swine_shoves(dt)  # Riduttore: respinta (dopo le collisioni)
        self._update_regen(dt)
        self._update_smite(dt)       # Smite: ricarica delle cariche
        self._update_plates(dt)      # placche delle torri (oro ai vicini)
        self._update_gold(dt)
        self._update_bot_shop(dt)
        self._update_respawns(dt)
        self._update_evolution(dt)   # evoluzioni + offerte di scelta
        self._check_win()

    # --- ondate --------------------------------------------------------------
    def _update_waves(self, dt: float) -> None:
        self._wave_timer -= dt
        if self._wave_timer <= 0.0:
            self._wave_timer += config.WAVE_INTERVAL
            self._wave_number += 1
            # Composizione dell'ondata: 3 melee + 3 caster, + cannone ogni N ondate.
            comp = (["melee"] * config.MELEE_PER_WAVE
                    + ["caster"] * config.CASTER_PER_WAVE)
            if self._wave_number % config.CANNON_EVERY == 0:
                comp.append("cannon")
            for team in (0, 1):
                for lane in self.map.lanes.keys():
                    for k, mtype in enumerate(comp):
                        due = self.time + k * config.MINION_SPAWN_GAP
                        self._spawn_queue.append((due, team, lane, mtype))

        # Fai spawnare i minion la cui attesa è scaduta.
        ready = [s for s in self._spawn_queue if s[0] <= self.time]
        if ready:
            self._spawn_queue = [s for s in self._spawn_queue if s[0] > self.time]
            for (_due, team, lane, mtype) in ready:
                self._spawn_minion(team, lane, mtype)

    def _spawn_minion(self, team: int, lane: str, mtype: str) -> None:
        pts = self.oriented_lane(lane, team)
        if len(pts) < 2:
            return
        base = pts[0]
        d = (pts[1] - base).normalized()
        perp = Vec2(-d.y, d.x)
        # Scostamento perpendicolare per non impilarli, pseudo-casuale ma deterministico.
        # La corsia entra come INDICE: l'hash di una stringa cambia a ogni processo
        # (PYTHONHASHSEED), quello di una tupla di interi no.
        lane_i = list(self.map.lanes).index(lane)
        jig = (hash((int(self.time * 10), team, lane_i, len(self.world.minions()))) % 7 - 3) * 0.25
        pos = base + d * 3.0 + perp * jig
        stats = dict(MINION_STATS[mtype])
        # I minion crescono col tempo (hp e danno) per rompere gli stalli.
        scale = 1.0 + config.MINION_SCALE_PER_MIN * (self.time / 60.0)
        stats["hp"] = stats["hp"] * scale
        stats["attack_damage"] = stats["attack_damage"] * scale
        m = self.world.spawn_minion(pos, team, lane, pts,
                                    self._minion_color(team, mtype),
                                    minion_type=mtype, **stats)
        m.lane_idx = 1
        self._ai[m.id] = {"astar_cd": 0.0}

    @staticmethod
    def _minion_color(team: int, mtype: str) -> tuple:
        base = config.TEAM_COLORS[team]
        if mtype == "caster":   # più chiaro
            return tuple(min(255, int(c + (255 - c) * 0.45)) for c in base)
        if mtype == "cannon":   # più scuro
            return tuple(int(c * 0.6) for c in base)
        return base

    # --- IA ------------------------------------------------------------------
    # Modalita' allenamento (tools/tier_sim.py): l'IA dei minion decide ogni N tick
    # (a turno per id), tra una decisione e l'altra prosegue l'ultima. 1 = ogni tick.
    minion_ai_stride = 1

    def _update_ai(self, dt: float) -> None:
        k = self.minion_ai_stride
        nv = self.world.grid.native_view() if self._minion_ai is not None else None
        if nv is not None and isinstance(self._combat_hash, _native.SpatialHash):
            self._minion_ai.run(self.world.minions(), self._ai, self.world.entities,
                                self._astar_move, self._combat_hash, nv,
                                self.time, dt, k, self._tick if k > 1 else 0)
        elif k > 1:
            t = self._tick
            mdt = dt * k
            for m in self.world.minions():
                if m.alive and (m.id + t) % k == 0:
                    self._ai_minion(m, mdt)
        else:
            for m in self.world.minions():
                if m.alive:
                    self._ai_minion(m, dt)
        for p in self.world.pokemons():
            if not p.alive:
                continue
            if p.is_decoy:
                continue              # il clone esca segue solo il suo percorso
            if p.fear_until > self.time and p.fear_src is not None:
                self._flee(p, dt)     # spavento: fuga automatica dalla fonte
                continue
            if p.charm_until > self.time:
                self._charmed(p, dt)  # affascinato (Incantevole): va verso Sylveon
                continue
            if p.taunt_until > self.time:
                self._taunted(p, dt)  # provocato (Provocazione): attacca chi l'ha provocato
                continue
            if p.octi_grab_until > self.time and self._octi_grabbing(p, dt):
                continue              # Presa a Ventosa: attacca solo l'afferrato
            if p.is_bot and self._flor_bot_seek(p, dt):
                continue              # Giardino: un bot ferito raccoglie un fiore vicino
            if p.is_bot and self._bp(p).utility:
                self._bot_utility_step(p, dt)
                continue              # IA a punteggio (engine/bot_utility.py; opzionale)
            if p.is_bot and self.bot_plan and self._bot_plan_step(p, dt):
                continue              # IA a piani (shopping, assedio, torri, roam; opzionale)
            if p.is_bot and p.is_jungler:
                self._ai_jungler(p, dt)
            elif p.is_bot:
                self._ai_bot(p, dt)
            else:
                self._ai_player(p, dt)
        # Neutrali (giungla): stazionari, colpiscono chi si avvicina.
        for w in self.world.wilds():
            if not w.alive:
                continue
            tgt = self._nearest_wild_target(w)
            w.target_id = tgt.id if tgt is not None else None
            if tgt is not None:
                self._face(w, tgt)

    def _nearest_wild_target(self, w):
        best, bd = None, 1e9
        for e in self._combat_hash.query_radius(w.pos.x, w.pos.y, w.attack_range + 2.0):
            if e.team == 2 or not getattr(e, "alive", False):
                continue
            if not isinstance(e, (Pokemon, Minion)):
                continue
            d = w.pos.dist(e.pos)
            if d <= w.attack_range + getattr(e, "radius", 0.0) + 0.5 and d < bd:
                best, bd = e, d
        return best

    def _ai_minion(self, m: Minion, dt: float) -> None:
        state = self._ai.setdefault(m.id, {})
        target = self._valid_minion_target(m, state)
        if target is None:
            target = self._acquire_minion_target(m)
            m.target_id = target.id if target else None
            if target is not None:
                # Ancora del guinzaglio: dove il minion si trovava quando ha ingaggiato.
                state["leash"] = Vec2(m.pos.x, m.pos.y)
            else:
                state.pop("leash", None)
        if target is not None:
            if self._within_attack(m, target):
                m.stop()
                self._face(m, target)
                m.chasing = False
            else:
                origin = state.get("leash", m.pos)
                if m.pos.dist(origin) <= _MINION_LEASH:
                    m.chasing = True
                    self._chase(m, target, dt)
                else:
                    # Troppo lontano dal punto di ingaggio: molla e torna in lane.
                    m.target_id = None
                    state.pop("leash", None)
                    m.chasing = False
                    self._follow_lane(m)
            return
        m.chasing = False
        self._follow_lane(m)

    def _valid_minion_target(self, m: Minion, state: dict):
        if m.target_id is None:
            return None
        t = self.world.get(m.target_id)
        if t is None or not getattr(t, "alive", False) or t.team == m.team:
            return None
        origin = state.get("leash", m.pos)
        # Perde il bersaglio se è troppo lontano o se si è allontanato dall'ingaggio.
        if m.pos.dist(t.pos) > m.aggro * 1.6 or m.pos.dist(origin) > _MINION_LEASH:
            return None
        return t

    def _acquire_minion_target(self, m: Minion):
        """Scelta del bersaglio in stile LoL.

        Priorità:
          1. Campione nemico che sta attaccando un alleato (difesa).
          2. Chi ha colpito questo minion di recente (ritorsione).
          3. Minion nemico più vicino.
          4. Campione nemico più vicino.
          5. Struttura nemica più vicina.
        """
        cands = []
        for e in self._combat_hash.query_radius(m.pos.x, m.pos.y, m.aggro):
            if e is m or e.team == m.team or not getattr(e, "alive", False):
                continue
            if m.pos.dist(e.pos) <= m.aggro + getattr(e, "radius", 0.0):
                cands.append(e)
        if not cands:
            return None

        def nearest(items):
            return min(items, key=lambda e: m.pos.dist(e.pos)) if items else None

        # 1) Campione che ha appena attaccato un alleato.
        defenders = [e for e in cands if isinstance(e, Pokemon)
                     and self.time - getattr(e, "_minion_aggro_at", -1e9) <= _MINION_AGGRO_WINDOW]
        if defenders:
            return nearest(defenders)
        # 2) Ritorsione contro chi ci ha colpito.
        if (m.last_attacker_id is not None
                and self.time - m.last_damaged_at <= _MINION_RETALIATE):
            attacker = self.world.get(m.last_attacker_id)
            if attacker in cands:
                return attacker
        # 3-5) Priorità per tipo.
        minions = [e for e in cands if isinstance(e, Minion)]
        if minions:
            return nearest(minions)
        pokes = [e for e in cands if isinstance(e, Pokemon)]
        if pokes:
            return nearest(pokes)
        structs = [e for e in cands if isinstance(e, Structure)]
        return nearest(structs)

    def _ai_player(self, p: Pokemon, dt: float) -> None:
        """Il Pokémon del giocatore è guidato dagli ORDINI (non da un'IA autonoma)."""
        if self._immobile(p):
            p.stop()                    # l'ordine resta: move_dest lo fa riprendere
            self._move_resume.pop(p.id, None)
            return
        if p.recalling:
            if p.order in ("move", "attack", "attackmove"):
                p.recalling = False     # un nuovo ordine interrompe il richiamo e si esegue
            else:                       # fermo durante il richiamo
                p.target_id = None
                p.stop()
                return
        if p.order == "move":
            if p.path:
                p.target_id = None   # in movimento puro non attacca
                return
            if self._resume_move(p):
                p.target_id = None
                return
            p.order = "idle"         # arrivato: passa a idle (auto-attacco di prossimità)
            p.move_dest = None

        if p.order == "attack":
            # Clic su un nemico: lo insegue finché non è a portata, poi attacca.
            t = self.world.get(p.order_target_id) if p.order_target_id is not None else None
            lk = self._lock_state(p, t)
            if lk and not self._chase_stalled(p, t):
                if lk == 1:
                    p.target_id = t.id
                    self._chase_or_hold(p, t, dt)
                else:                          # sparito un attimo: si continua senza colpirlo
                    p.target_id = None
                    self._chase_ghost(p, t, dt)
                return
            if t is not None and getattr(t, "alive", False):
                self.skip_chase(p, t.id)   # irraggiungibile: da fermo non lo si reinsegue
            p.order = "idle"          # morto, sparito nella nebbia o irraggiungibile: smetti
            p.order_target_id = None
            p.idle_anchor = None
            p.target_id = None
            p.stop()

        reach = self._auto_reach(p) + p.radius
        if p.order == "attackmove":
            # Attacco-movimento: verso il punto, attaccando il primo nemico valido
            # che entra nel raggio d'acquisizione (il più vicino al CURSORE).
            st = self._ai.setdefault(p.id, {})
            am = st.get("am")                 # [punto dell'ordine, dove ha agganciato]
            if am is None or am[0] is not p.order_point:
                am = st["am"] = [p.order_point, None]
            pt = p.order_point or p.pos
            tid = p.target_id if p.target_id is not None else st.get("lock_id")
            t = self.world.get(tid) if tid is not None else None
            lk = self._lock_state(p, t, p.champs_only)
            keep = (lk != 0
                    and p.pos.dist(t.pos) <= reach + t.radius + AMOVE_EXTRA + AMOVE_DROP_EXTRA)
            if keep and not self._within_attack(p, t):
                # Inseguimento a guinzaglio: oltre AMOVE_CHASE_LEASH tile dal punto
                # d'aggancio (o senza progressi) si molla, anche se il bersaglio scappa.
                if ((am[1] is not None and p.pos.dist(am[1]) > self.AMOVE_CHASE_LEASH)
                        or self._chase_stalled(p, t)):
                    self.skip_chase(p, t.id)
                    keep = False
            if not keep:
                t = self._acquire_near_point(p, reach + AMOVE_EXTRA, pt, p.champs_only,
                                             self._chase_skipped(p), ahead=pt)
                lk = 1 if t is not None else 0
            if t is not None:
                if am[1] is None or t.id != tid:
                    am[1] = Vec2(p.pos.x, p.pos.y)
                st["lock_id"] = t.id
                if lk == 1:
                    p.target_id = t.id
                    self._chase_or_hold(p, t, dt)
                else:
                    p.target_id = None
                    self._chase_ghost(p, t, dt)
                return
            am[1] = None
            st.pop("lock_id", None)
            p.target_id = None
            # move_dest: il punto raggiungibile verso il clic (vedi Simulation).
            dest = p.move_dest if p.move_dest is not None else p.order_point
            if dest is not None and p.pos.dist(dest) > 0.5:
                self._move_towards(p, dest, dt, use_astar=True)
            else:
                p.order = "idle"
                p.move_dest = None
                p.idle_anchor = None
                p.stop()
            return

        # idle / stop: da fermo attacca chi entra nel raggio d'acquisizione (un po'
        # più ampio della portata) e lo insegue per poco; "stop" tiene la posizione.
        hold = p.order == "stop"
        st = self._ai.setdefault(p.id, {})
        if p.idle_anchor is None:
            p.idle_anchor = Vec2(p.pos.x, p.pos.y)
        can_chase = not hold and p.pos.dist(p.idle_anchor) <= IDLE_CHASE_LEASH
        acq = reach + (0.0 if hold else ACQUIRE_EXTRA)
        tid = p.target_id if p.target_id is not None else st.get("lock_id")
        t = self.world.get(tid) if tid is not None else None
        skip = self._chase_skipped(p)
        if t is not None and t.id == skip and not self._within_attack(p, t):
            t = None
        lk = self._lock_state(p, t)
        if not (lk and (self._within_attack(p, t)
                        or (can_chase and p.pos.dist(t.pos) <= acq + t.radius + 1.0))):
            t = self._acquire_near_point(p, acq, p.pos, False, skip)
            lk = 1 if t is not None else 0
        ghost = lk == 2
        if (t is not None and st.get("idle_chase") and not self._within_attack(p, t)
                and self._chase_stalled(p, t)):
            self.skip_chase(p, t.id)        # non ci arriva (muro, angolo): lascia stare
            st.pop("idle_chase", None)
            p.stop()
            t = None
        if t is not None:
            st["lock_id"] = t.id
        if t is not None and not ghost and (self._within_attack(p, t)
                                            or (p.aa_active and p.aa_target_id == t.id)):
            p.target_id = t.id
            if st.pop("idle_chase", False):
                p.stop()
            self._face(p, t)
        elif t is not None and can_chase:
            p.target_id = None if ghost else t.id
            st["idle_chase"] = True
            if ghost:
                self._chase_ghost(p, t, dt)
            else:
                self._chase(p, t, dt)
        else:
            # Limite d'inseguimento: l'attacco automatico non trascina per la mappa.
            st.pop("lock_id", None)
            p.target_id = None
            if st.pop("idle_chase", False):
                p.stop()

    def _aa_target_ok(self, p, t, champs_only: bool = False) -> bool:
        """Bersaglio valido per l'attacco base di `p`: vivo, nemico, bersagliabile
        e VISTO dalla sua squadra (niente attacchi a chi sta nella nebbia)."""
        if t is None or not getattr(t, "alive", False) or t.team == p.team:
            return False
        if champs_only and not isinstance(t, Pokemon):
            return False
        if isinstance(t, Pokemon) and (t.untargetable_until > self.time or t.submerged):
            return False
        if isinstance(t, Unit) and not self.vision.is_enemy_visible(p.team, t.id):
            return False
        return True

    LOCK_GRACE = 0.8          # s: un bersaglio che sparisce un attimo (cespuglio, muro di
                              # giungla) resta agganciato: niente oscillazione col clic

    def _lock_state(self, p, t, champs_only: bool = False) -> int:
        """0 = nessun aggancio, 1 = bersaglio valido e visto, 2 = visto fino a un attimo
        fa (si continua a muoversi verso di lui ma NON lo si colpisce nella nebbia)."""
        if self._aa_target_ok(p, t, champs_only):
            self._lock_seen[p.id] = (t.id, self.time)
            return 1
        if (t is None or not getattr(t, "alive", False) or t.team == p.team
                or (champs_only and not isinstance(t, Pokemon))
                or (isinstance(t, Pokemon) and (t.untargetable_until > self.time or t.submerged))):
            return 0
        e = self._lock_seen.get(p.id)
        return 2 if e is not None and e[0] == t.id and self.time - e[1] <= self.LOCK_GRACE else 0

    def _chase_ghost(self, p, t, dt: float) -> None:
        if self._within_attack(p, t):
            p.stop()
            self._face(p, t)
        else:
            self._chase(p, t, dt)

    # --- bersagli mollati / irraggiungibili ----------------------------------------
    CHASE_SKIP_TIME = 6.0     # s: da fermo non si reinsegue chi si e' mollato
    CHASE_STALL_TIME = 1.2    # s fermi sul posto: il bersaglio e' irraggiungibile
    CHASE_STALL_MOVE = 0.5    # tile: spostamento che dimostra che ci si sta muovendo

    def skip_chase(self, p, target_id) -> None:
        """Il giocatore ha mollato `target_id` (o non si raggiunge): lo si attacca
        ancora se entra a portata, ma da fermo non lo si insegue piu' per un po'."""
        if target_id is not None:
            self._chase_skip[p.id] = (target_id, self.time + self.CHASE_SKIP_TIME)

    def _chase_skipped(self, p):
        e = self._chase_skip.get(p.id)
        if e is None:
            return None
        if self.time >= e[1]:
            del self._chase_skip[p.id]
            return None
        return e[0]

    def _chase_stalled(self, p, t) -> bool:
        """True se `p` insegue `t` ma da CHASE_STALL_TIME secondi resta sul posto
        (muro, angolo, nessuna strada). Conta solo lo spostamento di `p`, non la
        distanza da `t`: chi scappa si insegue ancora, e aggirare un muro allontana
        in linea d'aria. Il tempo non scorre se `p` e' fermo per altri motivi."""
        if (self._within_attack(p, t)
                or p.speed <= 0.0                                   # paralisi, stordimento
                or (p.aa_active and p.aa_target_id == t.id)):       # fermo a caricare il colpo
            self._chase_prog.pop(p.id, None)
            return False
        # Da rallentati si fa meno strada: la soglia scende con la velocita'.
        need = min(self.CHASE_STALL_MOVE, 0.25 * p.speed * self.CHASE_STALL_TIME)
        e = self._chase_prog.get(p.id)          # [id bersaglio, da dove, da quando, ultimo controllo]
        now = self.time
        if (e is None or e[0] != t.id or now - e[3] > 0.25     # inseguimento nuovo/interrotto
                or p.pos.dist(e[1]) > need):
            self._chase_prog[p.id] = [t.id, Vec2(p.pos.x, p.pos.y), now, now]
            return False
        e[3] = now
        return now - e[2] >= self.CHASE_STALL_TIME

    AMOVE_CHASE_LEASH = 5.0   # tile: l'attacco-movimento non insegue oltre dal punto d'aggancio

    def _acquire_near_point(self, p, radius: float, point, champs_only: bool,
                            skip_id=None, ahead=None):
        """Nemico valido entro `radius` (bordo a bordo) da `p`, il più vicino a `point`.
        `skip_id`: un bersaglio mollato, scartato se non e' gia' a portata d'attacco."""
        best, bd = None, 1e9
        for e in self._combat_hash.query_radius(p.pos.x, p.pos.y, radius + 1.5):
            if e is p or not self._aa_target_ok(p, e, champs_only):
                continue
            if e.id == skip_id and not self._within_attack(p, e):
                continue
            if ahead is not None and e.pos.dist(ahead) > p.pos.dist(ahead) + 0.5:
                continue      # alle spalle rispetto alla direzione in cui vai: si ignora
            if p.pos.dist(e.pos) > radius + getattr(e, "radius", 0.0):
                continue
            d = point.dist(e.pos)
            if d < bd:
                best, bd = e, d
        return best

    def _chase_or_hold(self, p, t, dt: float) -> None:
        if self._within_attack(p, t):
            p.stop()
            self._face(p, t)
        else:
            self._chase(p, t, dt)

    def _bp(self, u) -> BotParams:
        """Parametri dell'IA di `u`: i suoi se ne ha (taratura per ruolo), altrimenti
        quelli della squadra (neutrali: i default)."""
        bp = getattr(u, "bot_params", None)
        if bp is not None:
            return bp
        t = getattr(u, "team", -1)
        return self.bot_params[t] if t in (0, 1) else DEFAULT_BOT_PARAMS

    def _ai_bot(self, p: Pokemon, dt: float) -> None:
        if self._immobile(p):
            p.stop()
            return
        bp = self._bp(p)
        # Ritirata se la vita è bassa (con l'IA a punteggio la decide bot_utility).
        if not bp.utility and p.hp < p.max_hp * bp.retreat_hp:
            p.target_id = None
            self._move_towards(p, p.spawn_pos, dt, use_astar=True)
            return
        target = self._valid_target(p)
        if target is None:
            target = self._acquire_target(p, bp.aggro)
        # Non fare dive su una struttura nemica senza minion alleati a proteggerti.
        if (isinstance(target, Structure)
                and not self._has_allied_minion_near(p, target.pos, bp.struct_minion_r)):
            target = None
        # Prudenza: niente inseguimenti sotto torre senza buone possibilita' di kill
        # e niente allontanamenti dalla lane (engine/smite.py).
        target = self._bot_filter_target(p, target)
        p.target_id = target.id if target is not None else None
        if target is not None:
            self._bot_try_cast(p, target)
            if self._within_attack(p, target):
                p.stop()
                self._face(p, target)
            else:
                self._chase(p, target, dt)
            return
        # Difendi una struttura alleata sotto attacco, se vicina.
        defend = self._threatened_structure(p)
        if defend is not None:
            self._move_towards(p, defend.pos, dt, use_astar=True)
            return
        # Spiritomb: gira la mappa a raccogliere gli spiriti.
        if p.passive_id == SPIRITOMB_PASSIVE and self._bot_spirit_roam(p, dt):
            return
        # Altrimenti spingi la lane, ma fermati fuori dalla gittata delle torri
        # nemiche se non hai minion davanti.
        self._follow_bot_lane_safe(p, dt)

    def _has_allied_minion_near(self, p, pos, r: float) -> bool:
        if self._bot_nat is not None:
            return self._bot_nat.minion_near(self._combat_hash, p.team, pos.x, pos.y, r)
        for e in self._combat_hash.query_radius(pos.x, pos.y, r):
            if isinstance(e, Minion) and e.team == p.team and e.alive and pos.dist(e.pos) <= r:
                return True
        return False

    def _unsafe_dive(self, p, point) -> bool:
        bp = self._bp(p)
        if self._bot_nat is not None:
            return self._bot_nat.unsafe_dive(self.world.structures(), self._combat_hash, p.team,
                                             point.x, point.y, bp.tower_margin, bp.struct_minion_r)
        for e in self.world.structures():
            if e.kind != "tower" or not e.alive or e.team == p.team:
                continue
            if point.dist(e.pos) <= e.attack_range + bp.tower_margin and \
                    not self._has_allied_minion_near(p, e.pos, bp.struct_minion_r):
                return True
        return False

    def _threatened_structure(self, p):
        bp = self._bp(p)
        if self._bot_nat is not None:
            return self._bot_nat.threatened(self.world.structures(), self._combat_hash, p.team,
                                            p.pos.x, p.pos.y, bp.defend_enemy_r, bp.defend_max_dist)
        best, bestd = None, 1e9
        for e in self.world.structures():
            if e.team != p.team or not e.alive:
                continue
            enemies = [x for x in self._combat_hash.query_radius(e.pos.x, e.pos.y, bp.defend_enemy_r)
                       if x.team != p.team and getattr(x, "alive", False)
                       and isinstance(x, (Pokemon, Minion))]
            if not enemies:
                continue
            d = p.pos.dist(e.pos)
            if d < bestd:
                bestd, best = d, e
        # Difende solo se la struttura è abbastanza vicina (altrimenti continua
        # a spingere: difendere sempre porterebbe allo stallo).
        return best if best is not None and bestd < bp.defend_max_dist else None

    def _lane_cached(self, lane_name: str, team: int):
        """(waypoint, coordinate piatte) di oriented_lane, in cache: Vec2 e' frozen
        e la mappa delle lane non cambia in partita."""
        c = self.__dict__.setdefault("_lane_cache", {})
        key = (lane_name, team)
        r = c.get(key)
        if r is None:
            pts = self.oriented_lane(lane_name, team)
            r = (pts, tuple(v for q in pts for v in (q.x, q.y)))
            c[key] = r
        return r

    def _follow_bot_lane_safe(self, p: Pokemon, dt: float) -> None:
        st = self._ai.setdefault(p.id, {})
        nat = self._bot_nat
        if nat is not None:
            pts, flat = self._lane_cached(p.assigned_lane, p.team)
            if len(pts) < 2:
                self._move_towards(p, p.spawn_pos, dt)
                return
            bp = self._bp(p)
            grid = self.world.grid
            idx, code = nat.lane_safe(self.world.structures(), self._combat_hash,
                                      grid.native_view(), p.team, p.pos.x, p.pos.y, flat,
                                      st.get("lane_idx", 1), bp.lane_wp_reach,
                                      bp.dive_lookahead, bp.tower_margin, bp.struct_minion_r,
                                      getattr(p, "bush_blocks", False))
            st["lane_idx"] = idx
            if code == 0:
                p.stop()
            elif code == 1:
                p.set_path([pts[idx]])
            elif code == 2:
                self._astar_move(p, pts[idx], dt, getattr(p, "bush_blocks", False))
            else:
                self._set_step(p, pts[idx], dt)
            return
        pts = self.oriented_lane(p.assigned_lane, p.team)
        if len(pts) < 2:
            self._move_towards(p, p.spawn_pos, dt)
            return
        bp = self._bp(p)
        idx = min(st.get("lane_idx", 1), len(pts) - 1)
        wp = pts[idx]
        if p.pos.dist(wp) <= bp.lane_wp_reach and idx < len(pts) - 1:
            idx += 1
            wp = pts[idx]
        st["lane_idx"] = idx
        # Se avanzare porterebbe dentro la gittata di una torre nemica non
        # protetta da minion alleati, aspetta (equilibrio di lane).
        step = (wp - p.pos)
        ahead = p.pos + (step.normalized() * min(bp.dive_lookahead, step.length())) if step.length() > 1e-6 else p.pos
        if self._unsafe_dive(p, ahead) or self._unsafe_dive(p, p.pos):
            p.stop()
            return
        self._set_step(p, wp, dt)

    # --- aiutanti IA ---------------------------------------------------------
    def _valid_target(self, u):
        if u.target_id is None:
            return None
        t = self.world.get(u.target_id)
        if t is None or not getattr(t, "alive", False):
            return None
        # I bot perdono un bersaglio che entra in nebbia/cespuglio.
        if (getattr(u, "is_bot", False) and isinstance(t, Unit)
                and not self.vision.is_enemy_visible(u.team, t.id)):
            return None
        # Mantieni il bersaglio solo entro un guinzaglio ragionevole.
        if isinstance(u, Minion):
            leash = u.aggro * 1.8
        else:
            bp = self._bp(u)
            leash = bp.aggro * bp.leash_mult
        if u.pos.dist(t.pos) > leash:
            return None
        return t

    def _acquire_target(self, u, radius: float):
        bot = getattr(u, "is_bot", False)
        best, best_key = None, None
        for e in self._combat_hash.query_radius(u.pos.x, u.pos.y, radius):
            if e is u or e.team == u.team or not getattr(e, "alive", False):
                continue
            d = u.pos.dist(e.pos)
            if d > radius:
                continue
            # I bot ingaggiano solo nemici che la loro squadra vede davvero
            # (rispettano nebbia e cespugli). Le strutture sono sempre note.
            if bot and isinstance(e, Unit) and not self.vision.is_enemy_visible(u.team, e.id):
                continue
            key = (_prio(e), d)
            if best_key is None or key < best_key:
                best, best_key = e, key
        return best

    def _within_attack(self, u, target) -> bool:
        up = u.pos
        tp = target.pos
        d = math.hypot(up.x - tp.x, up.y - tp.y)
        if isinstance(u, Pokemon):
            # = d <= self._aa_reach(u, target) (bordo a bordo), con _auto_reach,
            # _fx_items e _claw_reach inline: stesse operazioni, stesso ordine.
            now = self.time
            r = u.attack_range
            if "zoom_lens" in (() if getattr(u, "itemlock_until", -1.0) > now
                               else getattr(u, "items", ())):
                r *= 1.15
            if u.passive_id == "distacco" and self._is_hidden(u):
                r += config.DISTACCO_RANGE_BONUS
            if u.shadowclaw_until > now:
                r = r * u.shadowclaw_mult
            return d <= r + getattr(u, "radius", 0.0) + getattr(target, "radius", 0.0)
        reach = getattr(u, "attack_range", 0.0) + getattr(target, "radius", 0.0)
        return d <= reach

    def _chase(self, u, target, dt: float) -> None:
        if isinstance(u, Pokemon) and u.aa_active and u.aa_target_id == target.id:
            # Caricamento in corso: resta fermo, il colpo esce con la tolleranza
            # (inseguire vorrebbe dire muoversi e annullarlo).
            u.stop()
            self._face(u, target)
            return
        if isinstance(u, Minion):
            u.chasing = True
        # Linea di vista libera -> passo diretto, altrimenti A*: lo decide gia'
        # _move_towards (prima la stessa linea di vista si calcolava due volte).
        self._move_towards(u, target.pos, dt, use_astar=True)
        self._face(u, target)

    def _follow_lane(self, m: Minion) -> None:
        if not m.lane_points:
            m.stop()
            return
        if m.lane_idx >= len(m.lane_points):
            m.lane_idx = len(m.lane_points) - 1
        wp = m.lane_points[m.lane_idx]
        if m.pos.dist(wp) <= _WAYPOINT_REACHED and m.lane_idx < len(m.lane_points) - 1:
            m.lane_idx += 1
            wp = m.lane_points[m.lane_idx]
        # Se il tratto dritto verso il waypoint e' ostruito (muro/cespuglio) usa A*.
        self._move_towards(m, wp, config.TICK_DT, use_astar=True)
        m.facing = direction_index(wp.x - m.pos.x, wp.y - m.pos.y)

    def _follow_bot_lane(self, p: Pokemon, dt: float) -> None:
        st = self._ai.setdefault(p.id, {})
        pts = self.oriented_lane(p.assigned_lane, p.team)
        if len(pts) < 2:
            self._move_towards(p, p.spawn_pos, dt)
            return
        idx = st.get("lane_idx", 1)
        idx = min(idx, len(pts) - 1)
        wp = pts[idx]
        if p.pos.dist(wp) <= 1.2 and idx < len(pts) - 1:
            idx += 1
            wp = pts[idx]
        st["lane_idx"] = idx
        self._move_towards(p, wp, dt, use_astar=True)

    def _set_step(self, u, dest: Vec2, dt: float) -> None:
        """Movimento verso dest: diretto se la via e' libera, altrimenti A*."""
        self._move_towards(u, dest, dt, use_astar=True)

    def _move_towards(self, u, dest: Vec2, dt: float, use_astar: bool = False) -> None:
        bb = getattr(u, "bush_blocks", False)   # minion: aggira i cespugli
        if not use_astar or self.world.grid.line_of_sight(u.pos.x, u.pos.y,
                                                           dest.x, dest.y, bb):
            u.set_path([dest])
            if isinstance(u, Pokemon) and not u.is_bot:   # solo il giocatore: i bot restano
                self._ai.setdefault(u.id, {})["astar_cd"] = 0.0   # al prossimo ostacolo A* subito
            return
        self._astar_move(u, dest, dt, bb)

    def _astar_move(self, u, dest: Vec2, dt: float, bb) -> None:
        """Ramo A* di _move_towards (linea di vista chiusa); usato anche
        dall'IA dei minion in Rust."""
        st = self._ai.setdefault(u.id, {})
        st["astar_cd"] = st.get("astar_cd", 0.0) - dt
        recomputed = st["astar_cd"] <= 0.0
        if recomputed:
            st["astar_cd"] = 0.3  # ricalcola A* al massimo ~3 volte/s
            start = (int(u.pos.x), int(u.pos.y))
            goal = (int(dest.x), int(dest.y))
            tile_path = find_path(self.world.grid, start, goal, bb)
            if tile_path:
                pts = smooth_path(self.world.grid, tile_path, bb)[1:]
            else:
                # Arrivo su un tile bloccato (bordo di un muro, cespuglio per i
                # minion): ci si avvicina, senza tenere il percorso di un'altra
                # destinazione.
                pts = plan_move(self.world.grid, (u.pos.x, u.pos.y),
                                (dest.x, dest.y), bb)
            st["astar_path"] = [Vec2(px, py) for (px, py) in pts]
        wps = st.get("astar_path") or [dest]
        if isinstance(u, Pokemon) and not u.is_bot and not recomputed and u.path:
            return        # il percorso in corso si segue: rimetterlo intero riporterebbe
                          # indietro i waypoint gia' raggiunti (avanti-indietro)
        u.set_path(wps)

    def forget_path(self, u) -> None:
        """Nuovo ordine: il prossimo _astar_move ricalcola subito, invece di
        seguire per un altro po' il percorso in cache."""
        st = self._ai.get(u.id)
        if st is not None:
            st.pop("astar_path", None)
            st.pop("astar_cd", None)

    def _face(self, u, target) -> None:
        u.facing = direction_index(target.pos.x - u.pos.x, target.pos.y - u.pos.y)

    # --- combattimento -------------------------------------------------------
    def _update_combat(self, dt: float) -> None:
        # Stesso effetto di "for c in world.combatants()" senza rifare la lista:
        # Unit e Structure hanno sempre `alive` (Combatant).
        for c in self.world._of_type(_UNIT_OR_STRUCT):
            if c.attack_timer > 0.0 and c.alive:
                c.attack_timer = max(0.0, c.attack_timer - dt)
        self._update_aa_projectiles(dt)   # attacchi base a distanza in volo

        # Minion e Pokémon colpiscono il proprio target se a tiro.
        for u in self.world.alive_units():
            if isinstance(u, Pokemon):
                if (self._immobile(u) or u.submerged or u.channel_until > self.time
                        or u.memento_until > self.time or u.puddle_until > self.time
                        or u.charm_until > self.time):
                    self._aa_cancel(u)   # stordito/immerso/canalizza: perde il windup
                    continue
                self._pokemon_auto_attack(u)
                continue
            t = self.world.get(u.target_id) if u.target_id is not None else None
            if t is not None and getattr(t, "alive", False) and self._within_attack(u, t):
                self._try_attack(u, t)

        # Torri: minion nemici a tiro per primi, poi Pokémon nemici "aggressivi".
        for s in self.world.structures():
            if not s.alive or s.attack_damage <= 0:
                continue
            self._tower_fire(s)

        self._cleanup_dead()

    def _tower_fire(self, tower: Structure) -> None:
        enemy_minions = []
        enemy_pokes = []
        tp = tower.pos
        tx, ty, team, rng = tp.x, tp.y, tower.team, tower.attack_range
        now = self.time
        for e in self._combat_hash.query_radius(tx, ty, rng):
            if e.team == team or not getattr(e, "alive", False):
                continue
            if tp.dist(e.pos) > rng + getattr(e, "radius", 0.0):
                continue
            if isinstance(e, Minion):
                enemy_minions.append(e)
            elif isinstance(e, Pokemon) and e.untargetable_until <= now:
                enemy_pokes.append(e)
        forced = getattr(tower, "_forced_id", None)
        if not enemy_minions and not enemy_pokes:
            # Caso comune: nessuno a tiro.
            if forced is not None:
                tower._forced_id = None
            tower.target_id = None
            return

        def cand(i):
            # Come un dict {id: e} di minion poi Pokemon (gli id sono unici).
            for e in enemy_minions:
                if e.id == i:
                    return e
            for e in enemy_pokes:
                if e.id == i:
                    return e
            return None
        target = None
        # Richiesta d'aiuto: chi ha colpito un Pokémon alleato sotto la torre.
        if forced is not None:
            target = cand(forced)
            tower._forced_id = None
        # Il bersaglio resta lo stesso finché è vivo e a tiro (stile LoL).
        if target is None and tower.target_id is not None:
            target = cand(tower.target_id)
        if target is not None:
            pass
        elif enemy_minions:
            target = min(enemy_minions, key=lambda m: tower.pos.dist(m.pos))
        else:
            # Un Pokémon viene colpito solo se ha fatto danno a un alleato di recente.
            aggressive = [p for p in enemy_pokes
                          if self.time - getattr(p, "_tower_aggro_at", -1e9) <= _TOWER_AGGRO_WINDOW]
            if aggressive:
                target = min(aggressive, key=lambda p: tower.pos.dist(p.pos))
        tower.target_id = target.id if target else None
        if target is not None:
            self._try_attack(tower, target)

    def _try_attack(self, attacker, target) -> None:
        if attacker.attack_timer > 0.0:
            return
        cd = attacker.attack_cooldown
        if isinstance(attacker, Pokemon) and attacker.rampage_until > self.time:
            cd *= 0.5   # Oltraggio: attacca più in fretta
        if isinstance(attacker, Pokemon):
            cd *= self._item_aa_cd_mult(attacker)   # Gelomai: -20% vel. d'attacco
        attacker.attack_timer = cd
        # FX dell'auto-attacco dei Pokémon (così si vede quando si attacca):
        # fendente per il corpo a corpo, piccolo colpo per chi tira da lontano.
        if isinstance(attacker, Pokemon):
            mt = attacker.types[0] if attacker.types else "Normal"
            self.fx_events.append({"kind": "attack",
                                   "a": (attacker.pos.x, attacker.pos.y),
                                   "b": (target.pos.x, target.pos.y), "mtype": mt,
                                   "melee": attacker.attack_range <= 2.2,
                                   "fx": attacker.attack_fx})
        elif isinstance(attacker, Structure) and attacker.kind == "tower":
            # Sfera d'energia della torre (cosmetica): vola verso il bersaglio,
            # colorata per squadra (stile LoL), cosi si vede quando la torre spara.
            self.fx_events.append({"kind": "tower_shot",
                                   "a": (attacker.pos.x, attacker.pos.y),
                                   "b": (target.pos.x, target.pos.y),
                                   "team": attacker.team})
        elif isinstance(attacker, Minion) and attacker.minion_type in ("caster", "cannon"):
            # Proiettile dei minion a distanza (caster/cannone): pallino blu/rosso
            # per squadra, cosi si vede quando colpiscono. I melee non sparano.
            self.fx_events.append({"kind": "minion_shot",
                                   "a": (attacker.pos.x, attacker.pos.y),
                                   "b": (target.pos.x, target.pos.y),
                                   "team": attacker.team,
                                   "big": attacker.minion_type == "cannon"})
        if self._item_dodge(attacker, target):   # Luminpolvere
            return
        if self._mien_parry_aa(attacker, target):   # Individua (Mienshao)
            return
        dmg = self._basic_damage(attacker, target)
        self._deal_damage(attacker, target, dmg)
        self._vamp(attacker, dmg, physical=True)
        self._item_after_aa(attacker, target)

    def _nearest_ally(self, u):
        best, bd = None, 1e9
        for e in self._combat_hash.query_radius(u.pos.x, u.pos.y, u.attack_range + 2.0):
            if e is u or e.team != u.team or not getattr(e, "alive", False):
                continue
            if not isinstance(e, (Pokemon, Minion)):
                continue
            d = u.pos.dist(e.pos)
            if d < bd:
                best, bd = e, d
        return best

    def _aa_power(self, attacker) -> tuple[float, bool]:
        """Potenza grezza dell'attacco base di un Pokémon e se è speciale."""
        if attacker.passive_id == ECHO_ID:
            # Ecolocalizzazione (Noibat/Noivern): onda sonora, danno speciale.
            return self._echo_power(attacker), True
        if attacker.passive_id in MOTH_PASSIVES:
            # Scaglie Velenose (Venonat/Venomoth): scaglie a distanza, danno speciale.
            return self._moth_aa_power(attacker), True
        if attacker.passive_id in DELI_PASSIVES:
            # Pacco in Arrivo / Propulsione Quark: palle di neve e getti, danno speciale.
            return self._deli_aa_power(attacker), True
        power = attacker.eff_attack()
        if attacker.rampage_until > self.time:
            power *= 1.4   # Oltraggio: attacchi base più forti
        return power, False

    def _basic_damage(self, attacker, target, power: float | None = None,
                      magic: bool | None = None) -> float:
        """Danno di un attacco base: fisico, mitigato dalla difesa del bersaglio.

        - Pokémon: usano l'Attacco effettivo (con buff/passivi); `power`/`magic`
          permettono di passare la potenza fotografata al rilascio (proiettili).
        - Torri: potenza con "ramp" crescente sullo stesso bersaglio (anti-dive).
        - Minion: potenza fissa, ridotta contro le strutture (grattano lento).
        """
        if isinstance(attacker, Pokemon):
            if power is None:
                power, magic = self._aa_power(attacker)
            magic = bool(magic)
        elif isinstance(attacker, Structure):
            magic = False
            power = attacker.attack_damage * self._tower_ramp(attacker, target)
        else:  # Minion o Wild
            magic = False
            power = attacker.attack_damage
            if isinstance(target, Structure):
                power *= config.STRUCT_MINION_DMG_MULT
                # Giudizio: i minion della squadra buffata fanno doppio alle torri.
                if isinstance(attacker, Minion) and self.jungle.team_has_judgment(attacker.team):
                    power *= 2.0
        if isinstance(target, Pokemon):
            if (target.passive_id == "mantelneve"
                    and not self._octi_ignores_dodge(attacker, target)):   # Prendimira
                power *= 0.9   # Mantelneve (Glaceon): attacchi base nemici -10%
            # Gli auto-attacchi sono fisici e senza tipo (niente STAB/efficacia);
            # possono fare critico e beneficiano di penetrazione armatura.
            return self._mitigate_hit(attacker, target, power, magic=magic,
                                      can_crit=True)
        return max(1.0, power)

    def _mitigate_hit(self, caster, target, raw: float, *, magic: bool,
                      crit_bonus: float = 0.0, can_crit: bool = False,
                      pierce_guard: bool = False, vs_defense: bool = False) -> float:
        """Critico -> penetrazione difesa -> mitigazione -> riduzione fissa.

        La penetrazione (percentuale poi fissa/letalita') non porta mai la difesa
        sotto 0; la riduzione danni fissa del difensore agisce dopo la mitigazione.
        """
        if isinstance(caster, Pokemon):
            chance = (caster.crit if can_crit else 0.0) + crit_bonus
            chance += self._sneasel_crit_bonus(caster)   # Nottesferza, Notte Fonda
            chance += self._octi_crit_bonus(caster, target)   # Prendimira, Presa a Ventosa
            chance += self._scy_crit_bonus(caster)   # Focalenergia: +40%
            if chance > 0.0 and random.random() < chance:
                raw *= self._crit_mult(caster, CRIT_MULT)   # Cecchino: +125%
        if not isinstance(target, Pokemon):
            return max(1.0, raw)
        # Psicoshock: danno speciale mitigato dalla Difesa invece che dalla Dif. Sp.
        defense = (target.eff_sp_defense() if magic and not vs_defense
                   else target.eff_defense())
        if pierce_guard and not magic and target.def_buff_active and target.def_buff_mult > 0:
            defense /= target.def_buff_mult   # Finta: la difesa potenziata non conta
        if isinstance(caster, Pokemon):
            if magic:
                flat = caster.magic_pen_flat + (0.0 if caster.lethality_physical
                                                else caster.lethality)
                defense = stat_penetrate(defense, caster.magic_pen_pct, flat)
            else:
                flat = caster.lethality if caster.lethality_physical else 0.0
                defense = stat_penetrate(defense, caster.armor_pen_pct, flat)
        out = stat_mitigate(raw, defense) - target.flat_reduction
        return max(1.0, out)

    def _vamp(self, caster, dmg: float, physical: bool) -> None:
        """Rubavita (solo fisico/auto) e omnivamp (ogni danno): cura = danno × %."""
        if not isinstance(caster, Pokemon) or dmg <= 0.0 or not caster.alive:
            return
        frac = caster.omnivamp + (caster.lifesteal if physical else 0.0)
        if frac > 0.0:
            caster.hp = min(caster.max_hp, caster.hp + dmg * frac * self._heal_in_mult(caster))

    @staticmethod
    def _tower_ramp(tower, target) -> float:
        if getattr(tower, "_ramp_target", None) == target.id:
            tower._ramp_stacks = getattr(tower, "_ramp_stacks", 0) + 1
        else:
            tower._ramp_target = target.id
            tower._ramp_stacks = 0
        return min(config.TOWER_RAMP_MAX,
                   1.0 + tower._ramp_stacks * config.TOWER_RAMP_PER_HIT)

    def _alive_towers(self, team: int) -> int:
        """Torri vive della squadra; 0 se una lane e' rimasta senza torri (nucleo scoperto)."""
        # Le torri di ogni squadra in cache per versione della lista delle strutture
        # (World._of_type la rifa' a ogni add/remove; kind/team/lane non cambiano):
        # a ogni chiamata resta solo il controllo di `alive`. Stesso conteggio.
        structs = self.world._of_type(Structure)
        c = self.__dict__.get("_alive_towers_c")
        if c is None or c[0] is not structs:
            per = {}
            for e in structs:
                if e.kind == "tower":
                    per.setdefault(e.team, []).append(e)
            c = self._alive_towers_c = (structs, per)
        n = 0
        left = {ln: 0 for ln in getattr(self, "_tower_lanes", {}).get(team, ())}
        for e in c[1].get(team, ()):
            if e.alive:
                n += 1
                if e.lane in left:
                    left[e.lane] += 1
        if config.CORE_OPEN_ON_EMPTY_LANE and any(v == 0 for v in left.values()):
            return 0
        return n

    def _deal_damage(self, attacker, target, amount: float,
                     dmg_type: str | None = None, ignore_shield: bool = False) -> None:
        """Infligge danno. `dmg_type` è il tipo del danno (per le mosse dei
        Pokémon) oppure None = "senza tipo" (auto-attacchi, torri, minion,
        neutrali, stati alterati). Il tipo serve alla Magidifesa dello Shedinja."""
        # Bersaglio non bersagliabile (es. SUB sotto terra): nessun danno.
        if isinstance(target, Pokemon) and target.untargetable_until > self.time:
            return
        amount *= self._honey_wild_mult(attacker, target)   # Miele: +30% ai neutrali
        amount *= self._item_dmg_mult(attacker, target)   # Mielino, Occhialineri, Elettropalla
        amount *= self._sneasel_wild_mult(attacker, target)   # Scalaroccia: +30% ai neutrali
        amount *= self._swine_wild_mult(attacker, target)   # Fiuto: +30% ai neutrali
        amount *= self._swine_type_mult(target, dmg_type)   # Grassospesso: -50% Fuoco/Ghiaccio
        amount *= self._flor_dmg_mult(attacker)   # fiore Rosso (Giardino): +15%
        amount *= self._item_struct_mult(attacker, target)   # Augite Nera: +20% alle torri
        amount *= self._incin_dmg_mult(attacker)   # Cattivo del Ring: +5% per nemico vicino
        if (isinstance(target, Wild) and isinstance(attacker, Pokemon)
                and attacker.passive_id == "muta"):
            # Muta (linea Nincada): +70% ai neutrali da Nincada, +30% dopo
            amount *= MUTA_WILD_MULT_LARVA if attacker.stage == 0 else MUTA_WILD_MULT
        if (isinstance(target, Wild) and isinstance(attacker, Pokemon)
                and attacker.passive_id == "forzabruta"):
            amount *= FORZABRUTA_WILD_MULT   # ForzaBruta (linea Totodile): +30% ai neutrali
        # Clone esca (Doppioteam): qualunque danno lo dissolve, senza ricompense.
        if isinstance(target, Pokemon) and target.is_decoy:
            if amount > 0.0:
                self._pop_decoy(target)
            return
        # Magidifesa (forma Shedinja): ignora i danni TIPATI delle mosse dei
        # Pokémon, tranne quelli superefficaci contro Coleottero/Spettro (Fuoco,
        # Volante, Roccia, Spettro, Buio), che sono letali. Il danno senza tipo
        # (auto-attacchi, torri, minion, neutrali, veleno/stati) passa normale.
        if (isinstance(target, Pokemon) and target.form == "shedinja"
                and dmg_type is not None):
            if dmg_type not in _SHEDINJA_WEAK:
                return   # assorbito dalla Magidifesa
            amount = max(amount, target.hp + target.shield)   # superefficace: letale
        # Assorbacqua (Poliwrath): il danno delle mosse Acqua NEMICHE lo cura
        # invece di ferirlo (il danno senza tipo e quello alleato passano normali).
        if (isinstance(target, Pokemon) and target.passive_id == "assorbacqua"
                and dmg_type == "Water" and attacker is not None
                and getattr(attacker, "team", target.team) != target.team
                and amount > 0.0):
            target.hp = min(target.max_hp, target.hp + amount)
            return
        # Linea Eevee: Assorbivolt (cura) e Fuocardore (si accende).
        if self._eevee_absorb(attacker, target, amount, dmg_type):
            return
        # Il nucleo è invulnerabile finché restano abbastanza torri della squadra.
        if isinstance(target, Core) and self._alive_towers(target.team) > config.CORE_EXPOSE_TOWERS_LEFT:
            return
        self._armature_track(attacker, target, amount)   # Armature (Charcadet)
        self._incin_on_dealt(attacker, target, amount)   # Prepotenza (linea Litten)
        # Scudo (solo Pokémon): assorbe prima degli HP.
        if (isinstance(target, Pokemon) and target.shield > 0.0
                and target.shield_until > self.time and not ignore_shield):
            absorbed = min(target.shield, amount)
            target.shield -= absorbed
            amount -= absorbed
        if isinstance(target, Pokemon) and not ignore_shield:
            amount = self._fat_absorb(target, amount)   # Grassospesso (dopo gli altri scudi)
        target.hp -= amount
        if amount > 0.0:   # statistiche: senza l'eccesso oltre gli HP rimasti
            self._track_damage(attacker, target,
                               min(amount, max(0.0, target.hp + amount)), dmg_type)
        self._scy_on_dealt(attacker, target, amount)   # Chele d'Acciaio: cura del 10%
        if isinstance(target, Pokemon) and target.endure_until > self.time:
            target.hp = max(1.0, target.hp)   # Resistenza: non scende sotto 1 HP
        if amount > 0.0:
            self._item_on_damaged(attacker, target, amount)   # Focalnastro, Cartelrossa...
        target.last_attacker_id = attacker.id
        target.last_damaged_at = self.time
        # Il sonno si rompe al primo danno subito.
        if (isinstance(target, Pokemon) and amount > 0.0 and target.sleep_until > self.time
                and target.rest_until <= self.time):   # Riposo: il danno non sveglia
            target.sleep_until = -1.0
        # Un Pokémon che danneggia un Pokémon/struttura diventa bersaglio valido per le torri.
        if isinstance(attacker, Pokemon) and isinstance(target, (Pokemon, Structure)):
            setattr(attacker, "_tower_aggro_at", self.time)
        # Un Pokémon che attacca un altro Pokémon viene focalizzato dai minion
        # nemici vicini che difendono l'alleato colpito.
        if isinstance(attacker, Pokemon) and isinstance(target, Pokemon):
            setattr(attacker, "_minion_aggro_at", self.time)
            if attacker.team != target.team and amount > 0.0 and not target.is_decoy:
                self._call_for_help(attacker, target)   # minion e torre della vittima
            if attacker.team != target.team:   # registra per gli assist
                dmg = getattr(target, "_damagers", None)
                if dmg is None:
                    dmg = {}
                    setattr(target, "_damagers", dmg)
                dmg[attacker.id] = self.time
        if target.hp <= 0.0 and target.alive:
            # Passiva Muta / Rancore: lo Shedinja può "barare la morte".
            if self._cheat_death(target, attacker):
                return
            self._on_death(target, attacker)

    def _track_damage(self, attacker, target, amount, dmg_type) -> None:
        """Statistiche (danno ai campioni, subito) e registro per il riepilogo
        della morte. Solo bersagli Pokemon: costo = un append per colpo."""
        if not isinstance(target, Pokemon):
            return
        target.stat_taken += amount
        if isinstance(attacker, Pokemon) and attacker.team != target.team:
            attacker.stat_dmg += amount
        log = target.dmg_log
        now = self.time
        if log and log[0][0] < now - RECAP_WINDOW:
            cut = 0
            while cut < len(log) and log[cut][0] < now - RECAP_WINDOW:
                cut += 1
            del log[:cut]
        label = self._dmg_label
        if label is None:
            label = "Altro" if isinstance(attacker, Pokemon) else _src_name(attacker)
        aid = getattr(attacker, "id", None)
        # I tick (veleno, zone) dello stesso attaccante e fonte si fondono: il
        # registro resta corto anche con tanti danni piccoli.
        if log:
            t0, a0, l0, d0, ty0 = log[-1]
            if a0 == aid and l0 == label and now - t0 < 0.5:
                log[-1] = (now, aid, label, d0 + amount, ty0)
                return
        log.append((now, aid, label, amount, dmg_type))
        if len(log) > 40:
            del log[:-40]

    def _make_death_recap(self, victim, killer) -> None:
        """Riassunto dell'ultima morte: chi ha ucciso e il danno subito negli
        ultimi RECAP_WINDOW secondi per attaccante e per fonte."""
        now = self.time
        by = {}
        for t, aid, label, dmg, ty in victim.dmg_log:
            if t < now - RECAP_WINDOW:
                continue
            e = by.setdefault(aid, {"total": 0.0, "src": {}})
            e["total"] += dmg
            s = e["src"].setdefault(label, [0.0, ty])
            s[0] += dmg
        rows = []
        for aid, e in by.items():
            a = self.world.get(aid) if aid is not None else None
            name = _src_name(a)
            rows.append({"id": aid, "name": name,
                         "champ": isinstance(a, Pokemon), "total": round(e["total"]),
                         "src": sorted(([lbl, round(v[0]), v[1]] for lbl, v in e["src"].items()),
                                       key=lambda r: -r[1])[:4]})
        rows.sort(key=lambda r: -r["total"])
        victim.death_recap = {"killer": getattr(killer, "id", None),
                              "killer_name": _src_name(killer),
                              "t": round(now, 1), "rows": rows[:5]}
        victim.dmg_log.clear()

    def _on_death(self, victim, killer) -> None:
        victim.hp = 0.0
        victim.alive = False
        if isinstance(victim, (Minion, Wild)):
            # Cosmetico: morte rapida del minion/mostro (game/death_fx.py).
            self.fx_events.append({"kind": "unit_death", "who": victim.id,
                                   "c": (victim.pos.x, victim.pos.y),
                                   "big": getattr(victim, "kind", "camp") in ("meteo", "arceus")})
        if isinstance(victim, Wild):
            self._honey_on_wild_death(victim, killer)   # Miele: vasetto per campo ripulito
            self.jungle.on_wild_death(victim, killer)
            return
        self._award_xp(victim, killer)
        self._award_gold(victim, killer)
        if isinstance(victim, Pokemon):
            self._make_death_recap(victim, killer)
            victim.stop()
            victim.target_id = None
            victim.airborne_until = -1.0   # niente sprite sospeso in aria da morto
            victim.airborne_start = -1.0
            victim.air_z = 0.0
            victim.judgment_until = -1.0   # perde il Giudizio morendo
            victim.memento_until = -1.0    # Memento interrotto
            victim.puddle_until = -1.0     # Acidarmatura interrotta
            victim.charm_until = -1.0
            victim.taunt_until = -1.0
            victim.invisible = False
            victim.nightfall_until = victim.ambush_until = -1.0   # Notte Fonda interrotta
            # Se muore in forma Shedinja, ripristina l'identità di Ninjask così il
            # respawn avviene normale (max_hp corretto, non 1).
            if victim.form == "shedinja":
                self._exit_shedinja(victim, 0.0)
            victim.respawn_timer = respawn_time(victim.level)
            victim.respawn_max = victim.respawn_timer
        elif isinstance(victim, Core):
            self.winner = killer.team if killer is not None else (1 - victim.team)
            self.announcements.append((self.time,
                                       f"Nucleo squadra {victim.team} distrutto!"))
        elif isinstance(victim, Structure):
            self.announcements.append((self.time,
                                       f"Torre {victim.lane} squadra {victim.team} distrutta!"))

    def _cleanup_dead(self) -> None:
        # Solo i tipi che si tolgono (lista in cache, stesso ordine di
        # world.entities); di solito non e' morto nessuno e non si fa altro.
        dead = [e for e in self.world._of_type(_CLEANUP_TYPES) if not e.alive]
        for e in dead:
            if isinstance(e, Minion):
                self.world.remove(e.id)
                self._ai.pop(e.id, None)
            elif not isinstance(e, Core):
                self.world.remove(e.id)

    # --- passiva Muta / forma Shedinja (linea Nincada) -----------------------
    def _muta_ready(self, p) -> bool:
        """Muta attiva solo da Ninjask (stage >= 1, cioè dal livello 8) a ricarica pronta."""
        return (isinstance(p, Pokemon) and p.passive_id == "muta" and p.stage >= 1
                and p.form != "shedinja" and self.time >= p.muta_cd_until)

    def _cheat_death(self, target, killer) -> bool:
        """Lo Shedinja può "barare la morte". Ritorna True se la morte è annullata."""
        if not isinstance(target, Pokemon) or target.is_decoy:
            return False
        if target.form == "shedinja":
            # Rancore: punisce il killer e rivive come Ninjask al 10% HP.
            if target.rancore_form:
                self._grudge_punish(killer)
                self._exit_shedinja(target, RANCORE_REVIVE_HP)
                self.announcements.append((self.time, f"{target.name}: Rancore!"))
                return True
            return False   # Shedinja da Muta: muore normalmente
        # Ninjask con Muta pronta: diventa Shedinja invece di morire.
        if self._muta_ready(target):
            self._enter_shedinja(target, rancore=False)
            return True
        return False

    def _grudge_punish(self, killer) -> None:
        """Rancore: tutte le mosse del killer vanno in ricarica completa."""
        if not isinstance(killer, Pokemon):
            return
        for ms in killer.moveset.values():
            mv = ms.chosen
            if mv is not None and ms.rank > 0:
                ms.cooldown_remaining = mv.cooldown_at(ms.rank)
                if mv.charges > 0:
                    ms.charges_cur = 0.0
        if killer.passive_id == "muta":
            killer.muta_cd_until = self.time + MUTA_COOLDOWN

    def _enter_shedinja(self, p, rancore: bool) -> None:
        # Salva l'identità di Ninjask per il ritorno.
        p.ninjask_stats = {k: getattr(p.base_stats, k) for k in
                           ("hp", "attack", "defense", "sp_attack", "sp_defense", "speed")}
        p.ninjask_types = list(p.types)
        p.ninjask_dex = p.dex
        p.form = "shedinja"
        p.form_until = self.time + MUTA_DURATION
        p.rancore_form = rancore
        p.muta_cd_until = self.time + MUTA_COOLDOWN
        rms = p.moveset.get("R")   # ricarica condivisa, mostrata sulla R
        if rms is not None:
            rms.cooldown_remaining = max(rms.cooldown_remaining, MUTA_COOLDOWN)
        p.base_stats = Stats.from_dict(SHEDINJA_BASE)
        p.types = list(SHEDINJA_TYPES)
        p.dex = SHEDINJA_DEX
        p.ranged = False
        self._recompute_stats(p)
        p.hp = 1.0
        p.max_hp = 1.0
        p.shield = 0.0
        self.fx_events.append({"kind": "area", "c": (p.pos.x, p.pos.y),
                               "radius": 1.4, "mtype": "Ghost", "src": p.id,
                               "fx": "ningrudge" if rancore else "ninmolt"})   # fx: solo grafica
        self.fx_events.append({"eid": p.id, "form": "shedinja"})
        self.announcements.append(
            (self.time, f"{p.name} -> Shedinja ({'Rancore' if rancore else 'Muta'})!"))

    def _exit_shedinja(self, p, hp_frac: float) -> None:
        from .stats import range_class
        if p.ninjask_stats:
            p.base_stats = Stats.from_dict(p.ninjask_stats)
        if p.ninjask_types:
            p.types = list(p.ninjask_types)
        if p.ninjask_dex:
            p.dex = p.ninjask_dex
        p.form = ""
        p.form_until = -1.0
        p.rancore_form = False
        p.ranged = range_class(p.base_stats) == "ranged"
        self._recompute_stats(p)
        p.hp = max(1.0, p.max_hp * hp_frac)
        p.untargetable_until = self.time   # torna bersagliabile all'istante
        self.fx_events.append({"kind": "area", "c": (p.pos.x, p.pos.y), "radius": 0.8,
                               "mtype": "Bug", "fx": "ninrevert"})   # solo grafica
        self.fx_events.append({"eid": p.id, "form": ""})

    def _update_forms(self, dt: float) -> None:
        for p in self.world.pokemons():
            if p.is_decoy:
                continue
            if p.form == "shedinja" and p.alive and self.time >= p.form_until:
                self._exit_shedinja(p, MUTA_REVERT_HP)   # torna Ninjask al 30%

    def force_shedinja(self, p) -> bool:
        """Debug/test: entra subito in forma Shedinja ignorando la ricarica."""
        if isinstance(p, Pokemon) and p.alive and p.form != "shedinja":
            self._enter_shedinja(p, rancore=False)
            p.muta_cd_until = self.time   # in test: niente ricarica
            return True
        return False

    # --- Doppioteam: clone esca ---------------------------------------------
    def _spawn_decoy(self, caster, dirv, dur: float) -> None:
        pos = Vec2(caster.pos.x, caster.pos.y)
        d = self.world.spawn_pokemon(
            pos=pos, team=caster.team, dex=caster.dex, name=caster.name,
            color=getattr(caster, "color", (200, 200, 200)),
            speed=caster.move_speed_base, is_bot=True,
            assigned_lane=caster.assigned_lane, types=list(caster.types))
        d.radius = caster.radius
        d.is_decoy = True
        d.decoy_until = self.time + dur
        d.stage = caster.stage
        d.level = caster.level
        d.form = ""            # il clone non ha forma (nessuna logica Muta)
        d.attack_range = 0.0   # non attacca
        d.vision = 2.0
        dest = self._nearest_walkable(pos + dirv * 4.0, pos)
        d.set_path([dest])
        self.fx_events.append({"eid": d.id, "anim": "Walk"})

    def _pop_decoy(self, d) -> None:
        self.fx_events.append({"kind": "area", "c": (d.pos.x, d.pos.y),
                               "radius": 0.8, "mtype": "Normal", "fx": "nindouble"})   # fx: solo grafica
        self._remove_decoy(d)

    def _remove_decoy(self, d) -> None:
        d.alive = False
        self.world.remove(d.id)
        self._ai.pop(d.id, None)

    def _update_decoys(self, dt: float) -> None:
        for d in list(self.world.pokemons()):
            if d.is_decoy and (not d.alive or self.time >= d.decoy_until):
                self._remove_decoy(d)

    def _fury_mult(self, caster, target) -> float:
        """Sfuriate: ogni colpo consecutivo sullo stesso bersaglio aumenta il
        danno, fino a 5×. Si azzera cambiando bersaglio o dopo 3s di pausa."""
        if not isinstance(caster, Pokemon):
            return 1.0
        if caster.fury_target == target.id and self.time <= caster.fury_until:
            caster.fury_stacks = min(5, caster.fury_stacks + 1)
        else:
            caster.fury_target = target.id
            caster.fury_stacks = 1
        caster.fury_until = self.time + 3.0
        return float(caster.fury_stacks)

    # --- esperienza / livelli ------------------------------------------------
    # XP condivisa tra i presenti: 100% da solo, 65% a testa in due, 47% in tre+.
    _XP_SHARE = {1: 1.0, 2: 0.65}

    def _team_avg_level(self, team: int) -> float:
        lv = [p.level for p in self.world.pokemons() if p.team == team]
        return sum(lv) / len(lv) if lv else 1.0

    def _avg_player_level(self) -> float:
        lv = [p.level for p in self.world.pokemons()]
        return sum(lv) / len(lv) if lv else 1.0

    def _recovery_mult(self, p: Pokemon) -> float:
        """Recupero: +15% XP da minion/giungla per chi e' 2+ livelli sotto la media."""
        return 1.15 if p.level <= self._team_avg_level(p.team) - 2.0 else 1.0

    def _award_xp(self, victim, killer=None) -> None:
        recipients = [p for p in self.world.pokemons()
                      if p.alive and p.team != victim.team
                      and p.pos.dist(victim.pos) <= XP_RADIUS]
        if not recipients:
            return
        if isinstance(victim, Minion):
            # XP dei minion condivisa tra i presenti, con bonus recupero.
            frac = self._XP_SHARE.get(len(recipients), 0.47)
            for p in recipients:
                self._gain_xp(p, victim.xp * frac * self._recovery_mult(p))
        elif isinstance(victim, Pokemon):
            # Uccisione: 140 + 40×liv. vittima, +20% se la vittima e' di livello
            # piu' alto del killer. Non condivisa, niente recupero.
            base = 140.0 + 40.0 * victim.level
            if isinstance(killer, Pokemon) and victim.level > killer.level:
                base *= 1.2
            for p in recipients:
                self._gain_xp(p, base)
        # strutture: niente esperienza (per ora)

    def _gain_xp(self, p: Pokemon, amount: float) -> None:
        if "everstone" in p.items and p.level < EVERSTONE_XP_LEVEL:
            amount *= EVERSTONE_XP   # Pietrastante: +50% esperienza fino al liv.15
        p.xp += amount
        new_level = level_from_xp(p.xp)
        while p.level < new_level:
            p.level += 1
            self._recompute_stats(p, heal_delta=True)

    @staticmethod
    def _n_stages(p: Pokemon) -> int:
        """Numero di stadi della linea evolutiva (per M, punto 8)."""
        stages = {int(e.get("stage", 0)) for e in p.evolutions} if p.evolutions else set()
        return max(1, len(stages))

    def _recompute_stats(self, p: Pokemon, heal_delta: bool = False,
                         preserve_ratio: bool = False) -> None:
        """Ricalcola le statistiche da livello, stadio e oggetti/passivi.

        Gli HP massimi includono i bonus HP di oggetti e mosse di potenziamento;
        p.stats resta "pulito" (il bonus HP non entra nelle statistiche base).
        """
        old_max = p.max_hp
        old_hp = p.hp
        # Bonus degli oggetti ricalcolati qui: l'Evolcondensa si spegne alla forma finale.
        p.item_bonus = self._item_bonus_for(p)
        for k, v in self._flor_sym_bonus(p).items():   # Simbiosi (Florges): 30% dei suoi bonus
            p.item_bonus[k] = p.item_bonus.get(k, 0.0) + v
        self._echo_convert(p)   # Ecolocalizzazione: Attacco degli oggetti -> Att. Sp.
        self._moth_convert(p)   # Scaglie Velenose: idem
        self._deli_convert(p)   # Pacco in Arrivo (Delibird): idem
        p.evio_mult = (EVIOLITE_MULT if "eviolite" in p.items and not self._is_final_form(p)
                       else 1.0)   # Evolcondensa
        self._octi_convert(p)   # linea Remoraid: Att. Sp. degli oggetti -> Attacco
        d = derive(p.base_stats, p.level, ranged=p.ranged)
        # M (moltiplicatore d'evoluzione, punto 8) si applica a HP, AD, armatura,
        # RM, mana e rigenerazione; NON a MS/AS, gittata, AP e moltiplicatore
        # abilita', cosi' il "feeling" del Pokémon resta stabile fra gli stadi.
        m = stat_evo_mult(p.stage, self._n_stages(p))
        p.stat_mult = m
        p.stats = Stats(hp=d.stats.hp * m, attack=d.stats.attack * m,
                        defense=d.stats.defense * m, sp_attack=d.stats.sp_attack,
                        sp_defense=d.stats.sp_defense * m, speed=d.stats.speed)

        def _b(k):   # somma dei bonus da oggetti, mosse di potenziamento e risvegli
            return (p.item_bonus.get(k, 0.0) + p.passive_bonus.get(k, 0.0)
                    + p.spirit_bonus.get(k, 0.0))

        # Velocita' (con eventuali extra MS/AS dagli oggetti; non toccati da M).
        p.move_speed_base = (d.move_speed_tiles * max(0.1, 1.0 + _b("move_speed_pct"))
                             + _b("move_speed"))
        # Attacchi/s = base del livello × (1 + somma dei bonus), con un tetto.
        p.as_base = d.attack_speed
        p.as_bonus = _b("attack_speed")
        as_total = min(AS_CAP, d.attack_speed * (1.0 + p.as_bonus))
        p.attack_speed = as_total
        p.attack_cooldown = 1.0 / as_total if as_total > 0 else 1.0
        p.ability_mult = d.ability_mult
        p.max_mana = d.max_mana * m
        p.mana = min(p.mana, p.max_mana) if p.mana > 0 else p.max_mana
        p.mana_regen5 = d.mana_regen5 * m
        # Stat di combattimento dagli oggetti (AP = SpA degli oggetti + AP puro).
        p.ap = _b("sp_attack") + _b("ap")
        p.crit = max(0.0, min(1.0, _b("crit")))
        p.armor_pen_pct = max(0.0, _b("armor_pen_pct"))
        p.magic_pen_pct = max(0.0, _b("magic_pen_pct"))
        p.magic_pen_flat = max(0.0, _b("magic_pen_flat"))
        p.lifesteal = max(0.0, _b("lifesteal"))
        p.omnivamp = max(0.0, _b("omnivamp"))
        p.ability_haste = max(0.0, _b("ability_haste")) + cdr_to_haste(_b("cdr"))
        # Identita' innate (non dagli oggetti), con extra sommati dagli oggetti.
        p.tenacity = stat_combine_ten(d.tenacity, _b("tenacity"))
        p.hp_regen5 = d.hp_regen5
        p.flat_reduction = d.flat_reduction
        p.lethality = d.lethality + _b("lethality")
        p.lethality_physical = d.lethality_physical
        p.heal_power = d.heal_power + _b("heal_power")
        p.range_cls = d.range_cls
        p.range_val = d.range_val
        bonus_hp = (p.item_bonus.get("hp", 0.0) + p.passive_bonus.get("hp", 0.0)
                    + p.spirit_bonus.get("hp", 0.0))
        p.max_hp = (p.stats.hp + bonus_hp) * self._ursa_hp_mult(p)   # Luna Piena: +25%
        if p.form == "shedinja":
            p.max_hp = 1.0   # lo Shedinja ha SEMPRE 1 HP (non derivato)
        p.attack_damage = p.stats.attack
        if preserve_ratio and old_max > 0:
            p.hp = min(p.max_hp, old_hp / old_max * p.max_hp)
        elif heal_delta:
            p.hp = min(p.max_hp, old_hp + max(0.0, p.max_hp - old_max))
        else:
            p.hp = min(p.max_hp, old_hp)

    # --- richiamo ------------------------------------------------------------
    def _resume_move(self, p: Pokemon) -> bool:
        """Percorso vuoto ma destinazione dell'ordine non raggiunta (un CC o uno
        spostamento l'ha svuotato): ricalcola il percorso fin li'. Non riprova dal
        punto in cui aveva gia' riprovato senza progressi (es. destinazione dentro
        una torre), cosi' non resta incastrato in un ciclo."""
        d = p.move_dest
        if d is None or p.pos.dist(d) <= 0.25:
            return False
        last = self._move_resume.get(p.id)
        if last is not None and last[0] is d and last[1].dist(p.pos) < 0.3:
            return False
        self._move_resume[p.id] = (d, Vec2(p.pos.x, p.pos.y))
        pts = plan_move(self.world.grid, (p.pos.x, p.pos.y), (d.x, d.y))
        if not pts:
            return False
        p.set_path([Vec2(x, y) for (x, y) in pts])
        return True

    def _update_recall(self, dt: float) -> None:
        for p in self.world.pokemons():
            if not p.recalling:
                continue
            interrupted = (not p.alive or self._immobile(p)
                           or p.last_damaged_at > p.recall_start
                           or p.order in ("move", "attack", "attackmove"))
            if interrupted:
                p.recalling = False
            elif self.time >= p.recall_until:
                p.recalling = False
                p.pos = Vec2(p.spawn_pos.x, p.spawn_pos.y)
                p.hp = min(p.max_hp, p.hp + p.max_hp * config.RECALL_HEAL)
                p.stop()
                p.move_dest = None

    # --- oro / negozio -------------------------------------------------------
    def _update_gold(self, dt: float) -> None:
        # L'oro passivo matura solo a partire da 1:30 (vale per tutti).
        if self.time < config.GOLD_PASSIVE_START:
            return
        for p in self.world.pokemons():
            if p.alive:
                p.gold += config.GOLD_PASSIVE_PER_SEC * dt

    def _minion_gold(self, m: Minion) -> int:
        """Oro del minion. Il cannone cresce di +1/min fino al tetto; gli altri
        valgono il loro valore fisso."""
        if m.minion_type == "cannon":
            extra = int(self.time / 60.0) * config.CANNON_GOLD_PER_MIN
            return min(config.CANNON_GOLD_MAX, config.CANNON_GOLD_BASE + extra)
        return int(m.gold)

    def _award_gold(self, victim, killer) -> None:
        if isinstance(victim, Minion):
            # Oro solo a chi dà l'ultimo colpo (last-hit); kill da torre/altri
            # minion danno 0 (killer non è un Pokémon).
            if isinstance(killer, Pokemon):
                killer.stat_cs += 1
                killer.gold += self._minion_gold(victim)
                self.fx_events.append({"kind": "gold", "who": killer.id,   # cosmetico: "+oro"
                                       "c": (victim.pos.x, victim.pos.y),
                                       "g": self._minion_gold(victim)})
                self._track_gold(killer, self._minion_gold(victim), "minion")   # Sole e Luna
                self._exp_share(victim, killer)   # Condividi Esp. degli alleati vicini
        elif isinstance(victim, Pokemon):
            self._award_kill(victim, killer)
        elif isinstance(victim, Structure):
            self._award_structure(victim, killer)

    def _award_structure(self, victim, killer) -> None:
        """Torre: 150 oro a tutta la squadra che l'ha distrutta, +100 a chi era
        vicino. Il nucleo non dà oro. Le placche residue (entro i 14 min) cadono
        alla distruzione, divise tra i vicini."""
        if isinstance(victim, Core):
            return
        team = 1 - victim.team   # la torre muore solo per mano della squadra avversaria
        for p in self.world.pokemons():
            if p.team != team:
                continue
            p.gold += config.TOWER_GOLD_TEAM
            if p.alive and p.pos.dist(victim.pos) <= config.OBJECTIVE_NEAR_RADIUS:
                p.gold += config.TOWER_GOLD_LOCAL
        # Placche ancora in piedi (burst che salta le soglie): cadono tutte ora.
        left = getattr(victim, "plates_left", 0)
        if left > 0 and self.time < config.PLATE_WINDOW:
            self._award_plates(victim, left)
            victim.plates_left = 0

    def _award_plates(self, tower, n: int) -> None:
        """Assegna `n` placche (120 oro l'una) divise tra i nemici vicini."""
        team = 1 - tower.team
        near = [p for p in self.world.pokemons()
                if p.team == team and p.alive
                and p.pos.dist(tower.pos) <= config.OBJECTIVE_NEAR_RADIUS]
        if not near:
            return                      # nessuno vicino: oro perso
        share = config.PLATE_GOLD * n / len(near)
        for p in near:
            p.gold += share

    def _update_plates(self, dt: float) -> None:
        """Le placche cadono man mano che la torre perde vita, nei primi 14 min.
        Una placca cade quando l'HP scende sotto max_hp × placche/(PLATE_COUNT+1)."""
        if self.time >= config.PLATE_WINDOW:
            return
        step = config.PLATE_COUNT + 1
        for s in self.world.structures():
            if isinstance(s, Core) or not s.alive:
                continue
            left = getattr(s, "plates_left", 0)
            if left <= 0 or s.max_hp <= 0:
                continue
            # Fa cadere tutte le placche le cui soglie sono state superate.
            while left > 0 and s.hp <= s.max_hp * left / step:
                self._award_plates(s, 1)
                left -= 1
            s.plates_left = left

    def _avg_player_gold(self) -> float:
        gs = [p.gold for p in self.world.pokemons()]
        return sum(gs) / len(gs) if gs else 0.0

    def _team_total_gold(self, team: int) -> float:
        return sum(p.gold for p in self.world.pokemons() if p.team == team)

    def _comeback_mult(self, team: int) -> float:
        """Taglie di recupero: gli obiettivi neutrali danno il doppio alla
        squadra indietro di oltre 3000 oro."""
        behind = self._team_total_gold(1 - team) - self._team_total_gold(team)
        return 2.0 if behind > config.COMEBACK_THRESHOLD else 1.0

    def _victim_bounty(self, victim) -> float:
        """Taglia per aver ucciso `victim`: base 300, su con la serie di
        uccisioni (max 600), giù con la serie di morti (min 150)."""
        bounty = float(config.KILL_BASE_BOUNTY)
        if victim.kill_streak >= 2:          # era in serie di uccisioni
            bounty *= 1.0 + config.KILL_STREAK_STEP * (victim.kill_streak - 1)
            bounty = min(bounty, config.BOUNTY_MAX)
        deaths = victim.death_streak + 1     # questa morte compresa
        if deaths >= 2:                      # in serie di morti
            bounty *= 1.0 - config.DEATH_STREAK_STEP * (deaths - 1)
            bounty = max(bounty, config.BOUNTY_MIN)
        return bounty

    def _award_kill(self, victim, killer) -> None:
        """Assegna oro per l'uccisione di un campione: taglia al killer (con
        prima uccisione e taglia sul vantaggio) e 50% agli assistenti. Se a
        uccidere è una torre/minion, la taglia va divisa tra gli assistenti;
        se non ce ne sono, si perde. Poi aggiorna le serie."""
        bounty = self._victim_bounty(victim)
        killer_is_champ = isinstance(killer, Pokemon) and killer.team != victim.team
        first_blood = not self._first_blood_done
        if first_blood:
            self._first_blood_done = True
            self.announcements.append((self.time, "Prima uccisione!"))

        # Assistenti: chi ha colpito la vittima negli ultimi ASSIST_WINDOW
        # secondi (escluso il killer), vivo e della squadra avversaria.
        dmg = getattr(victim, "_damagers", {})
        assisters = []
        for aid, t in list(dmg.items()):
            if self.time - t > config.ASSIST_WINDOW:
                continue
            if killer is not None and aid == getattr(killer, "id", None):
                continue
            a = self.world.get(aid)
            if isinstance(a, Pokemon) and a.alive and a.team != victim.team:
                assisters.append(a)
        setattr(victim, "_damagers", {})
        victim.stat_deaths += 1
        if killer_is_champ:
            killer.stat_kills += 1
        for a in assisters:
            a.stat_assists += 1
        self._item_on_takedown(killer if killer_is_champ else None, assisters)   # Occhialineri, Monetamuleto

        fb_assist = config.FIRST_BLOOD_ASSIST if first_blood else 0
        if killer_is_champ:
            gain = bounty
            if first_blood:
                gain += config.FIRST_BLOOD_BONUS
            avg = self._avg_player_gold()
            if victim.gold > avg:            # taglia sul vantaggio (shutdown)
                gain += min(config.SHUTDOWN_MAX,
                            config.SHUTDOWN_FACTOR * (victim.gold - avg))
            killer.gold += gain
            self._add_soul(killer, config.SOUL_PER_KILL)
            if assisters:                    # gli assistenti si dividono il 50%
                share = bounty * config.ASSIST_SHARE / len(assisters)
                for a in assisters:
                    a.gold += share + fb_assist
                    self._add_soul(a, config.SOUL_PER_ASSIST)
        elif assisters:
            # Uccisione da torre/minion: la taglia intera si divide tra gli assist.
            share = bounty / len(assisters)
            for a in assisters:
                a.gold += share + fb_assist
                self._add_soul(a, config.SOUL_PER_ASSIST)

        # Aggiorna le serie (la vittima perde la sua serie di uccisioni).
        victim.kill_streak = 0
        victim.death_streak += 1
        if killer_is_champ:
            killer.kill_streak += 1
            killer.death_streak = 0

    @staticmethod
    def _add_soul(p, n: int) -> None:
        if p.passive_id == "corpofunesto":
            p.soul_charges = min(config.SOUL_MAX, p.soul_charges + n)

    def _add_inversion(self, p, n: int = 5) -> None:
        """Passiva Inversione: cariche da colpo di spell. Tetto 50, che sale a
        100 durante il sovraccarico (stare dentro VerdeBufera)."""
        if getattr(p, "passive_id", "") != "inversione":
            return
        p.inversion_last_gain = self.time   # colpire tiene viva la finestra anti-decadimento
        cap = 100 if p.inversion_oc_until > self.time else 50
        if p.inversion_stacks < cap:
            p.inversion_stacks = min(cap, p.inversion_stacks + n)

    def shop_buy(self, unit, item_id: str) -> bool:
        if not isinstance(unit, Pokemon) or not unit.alive:
            return False
        if unit.pos.dist(unit.spawn_pos) > config.SHOP_RANGE:
            return False
        it = self.item_catalog.get(item_id)
        if it is None:
            return False
        price = self.item_catalog.effective_price(item_id, unit.items)
        if unit.gold < price:
            return False
        # Al massimo un oggetto Scelta, un paio di scarpe, un oggetto iniziale e
        # una Pietrastante; un Condividi Esp. per squadra; consumabili al massimo 2.
        if not self._item_buy_ok(unit, item_id):
            return False
        consumed = self.item_catalog.components_in(item_id, unit.items)
        if len(unit.items) - len(consumed) + 1 > ITEM_SLOTS:
            return False
        for c in consumed:
            unit.items.remove(c)
        unit.items.append(item_id)
        unit.gold -= price
        self._apply_items(unit)
        return True

    def shop_sell(self, unit, slot: int) -> bool:
        if not isinstance(unit, Pokemon) or not unit.alive:
            return False
        if unit.pos.dist(unit.spawn_pos) > config.SHOP_RANGE:
            return False
        if slot < 0 or slot >= len(unit.items):
            return False
        item_id = unit.items.pop(slot)
        unit.gold += self.item_catalog.sell_value(item_id)
        self._apply_items(unit)
        return True

    def _apply_items(self, unit) -> None:
        self._recompute_stats(unit, preserve_ratio=True)   # ricalcola anche item_bonus

    def _update_bot_shop(self, dt: float) -> None:
        """I bot comprano oggetti quando tornano in base con abbastanza oro."""
        for p in self.world.pokemons():
            if not p.alive or not p.is_bot or p.bot_standin:
                continue
            if p.pos.dist(p.spawn_pos) > config.SHOP_RANGE:
                continue
            st = self._ai.setdefault(p.id, {})
            st["shop_cd"] = st.get("shop_cd", 0.0) - dt
            if st["shop_cd"] <= 0.0:
                st["shop_cd"] = 1.5
                self._bot_buy(p)

    def _bot_buy(self, p) -> None:
        """Compra il prossimo pezzo dell'albero dei consigliati: l'oggetto se basta
        l'oro, altrimenti il primo componente (anche annidato) che si può pagare."""
        cat = self.item_catalog

        def can_buy(iid):
            if cat.get(iid) is None or not self._item_buy_ok(p, iid):
                return False
            slots_after = len(p.items) - len(cat.components_in(iid, p.items)) + 1
            return p.gold >= cat.effective_price(iid, p.items) and slots_after <= ITEM_SLOTS

        def pick(iid, need=1):
            if p.items.count(iid) >= need:
                return None
            if can_buy(iid):
                return iid
            comps = cat.get(iid).build_from
            for comp in dict.fromkeys(comps):
                got = pick(comp, comps.count(comp))
                if got is not None:
                    return got
            return None

        for target in (p.rec_items or cat.recommended_for(p.role)):
            it = cat.get(target)
            if it is None or target in p.items:
                continue
            if not self._item_buy_ok(p, target):
                continue   # regola "al massimo uno": passa al consigliato dopo
            got = pick(target)
            if got is not None:
                self.shop_buy(p, got)
            return  # un passo alla volta sull'oggetto corrente

    # --- rigenerazione -------------------------------------------------------
    def _update_regen(self, dt: float) -> None:
        """Cura rapida vicino alla propria base (fontana) e lieve rigen altrove."""
        for p in self.world.pokemons():
            if not p.alive or p.hp >= p.max_hp:
                continue
            if p.pos.dist(p.spawn_pos) <= 5.0:
                p.hp = min(p.max_hp, p.hp + p.max_hp * 0.25 * dt)   # ~25%/s
            else:
                p.hp = min(p.max_hp, p.hp + (p.max_hp * 0.015 + 2.0) * dt
                           * self._healcut_mult(p))   # cure ridotte (Sporcolancio)

    # --- mosse: esecuzione ---------------------------------------------------
    def _update_casts(self, dt: float) -> None:
        for p in self.world.pokemons():
            cast = p.pending_cast
            p.pending_cast = None
            if cast is None or not p.alive or self._immobile(p) or p.fear_until > self.time:
                continue
            if p.submerged:          # sott'acqua non si lanciano altre mosse
                continue
            if p.memento_until > self.time:   # Memento: invisibile, niente mosse
                continue
            if p.puddle_until > self.time or p.charm_until > self.time:
                continue                      # pozza (Acidarmatura) / affascinato
            if p.taunt_until > self.time:
                continue                      # provocato: solo attacchi base
            self._cast(p, cast[0], cast[1], cast[2])

    def _cast(self, p: Pokemon, slot: str, tx: float, ty: float) -> None:
        ms = p.moveset.get(slot)
        if ms is None or ms.chosen is None or not ms.ready:
            return
        if p.silence_until > self.time:   # Ripeti: silenziato, solo autoattacchi
            return
        if p.channel_until > self.time:   # Lanciafiamme/Volo: sta già canalizzando
            return
        mv = ms.chosen
        rank = ms.rank
        # Spiritomb: senza un bersaglio valido (nessuno da legare, nessuno spirito
        # nel raggio) la mossa non parte e la ricarica non si spende.
        # Linee con logica propria: senza un bersaglio valido (nessuno da legare,
        # Comete senza bersagli...) la mossa non parte e la ricarica non si spende.
        tgt0 = Vec2(tx, ty)
        for name in hook_names("precheck"):
            if not getattr(self, name)(p, mv, tgt0):
                return
        self._sneasel_on_cast(p, mv)   # Notte Fonda: lanciare una mossa toglie l'invisibilità
        # Lanciare annulla un attacco base in CARICAMENTO (il ciclo non si consuma);
        # se il colpo è già uscito il recupero continua da dove era (niente punizione).
        self._aa_cancel(p)
        if mv.d.get("aa_reset"):     # mosse che azzerano il ciclo: il prossimo attacco parte subito
            self._aa_reset(p)
        if mv.charges > 0:           # Erboscivolata: consuma una carica (scatto)
            if ms.charges_cur < 0:
                ms.charges_cur = float(mv.charges)
            ms.charges_cur = max(0.0, ms.charges_cur - 1.0)
        ms.cooldown_remaining = mv.cooldown_at(rank) * ability_haste_factor(p.ability_haste)
        if p.spite_until > self.time:   # Dispetto: ricariche allungate
            ms.cooldown_remaining *= p.spite_mult
        mv = self._item_on_cast(p, mv, slot)   # Scelta, Assorbisfera, Grandelente (+portata)
        self._hatt_on_cast(p)   # Quiete (Hatenna): lanciare vicino alla linea agita
        mv = self._mag_on_cast(p, mv)   # Campo Elettrico: +50% portata (linea Magnemite)
        if slot == "R":
            self._item_on_ult(p)   # Rocciaumida / Rocciacalda: pioggia o sole attorno a se'
        origin = Vec2(p.pos.x, p.pos.y)
        tgt = Vec2(tx, ty)
        to = tgt - origin
        dirv = to.normalized() if to.length() > 1e-6 else Vec2(1.0, 0.0)
        p.facing = direction_index(dirv.x, dirv.y)
        shape = mv.shape
        anim = CAST_ANIMS.get(shape)
        # Anche senza animazione di default: l'editor delle animazioni puo'
        # assegnarne una alla mossa ("move" = id, tag solo cosmetico).
        self.fx_events.append({"eid": p.id, "anim": anim or "", "move": mv.id})
        # Gli eventi visivi aggiunti dai _cast_* qui sotto vengono "taggati" con lo
        # stile FX della mossa (campo "fx" nel JSON), così il client può disegnare
        # una forma dedicata e ricolorata invece del raggio/cerchio generico.
        _fx0 = len(self.fx_events)
        if shape == "projectile":
            self._cast_projectile(p, mv, rank, origin, dirv)
        elif shape == "bubble":
            self._cast_bubble(p, mv, rank, origin, dirv)
        elif shape == "cone":
            self._cast_cone(p, mv, rank, origin, dirv)
        elif shape == "area":
            self._cast_area(p, mv, rank, origin, tgt)
        elif shape == "whirlpool":
            self._cast_whirlpool(p, mv, rank, origin, tgt)
        elif shape == "dash":
            self._cast_dash(p, mv, rank, origin, tgt, dirv)
        elif shape == "sub":
            self._cast_sub(p, mv, rank, origin, tgt, dirv)
        elif shape == "bolt":
            self._cast_bolt(p, mv, rank, origin, tgt)
        elif shape == "blink":
            self._cast_blink(p, mv, rank, origin, tgt, dirv)
        elif shape == "curse":
            self._cast_curse(p, mv, rank, origin, tgt)
        elif shape == "fan":
            self._cast_fan(p, mv, rank, origin, dirv)
        elif shape == "dashback":
            self._cast_dashback(p, mv, rank, origin, tgt, dirv)
        elif shape == "rain":
            self._cast_rain(p, mv, rank, origin, tgt)
        elif shape == "storm":
            self._cast_storm(p, mv, rank, origin, tgt)
        elif shape == "leap":
            self._cast_leap(p, mv, rank, origin, tgt, dirv)
        elif shape == "chain":
            self._cast_chain(p, mv, rank, origin, tgt, dirv)
        elif shape == "rancore":
            self._cast_rancore(p, mv, rank)
        elif shape == "arc":
            self._cast_arc(p, mv, rank, origin, tgt)
        elif shape == "raindance":
            self._cast_raindance(p, mv, rank, origin, tgt)
        elif shape == "healzone":
            self._cast_healzone(p, mv, rank, origin, tgt)
        elif shape == "throwback":
            self._cast_throwback(p, mv, rank, origin, tgt, dirv)
        elif shape == "bounce":
            self._cast_bounce(p, mv, rank, origin, tgt)
        elif shape == "perish":
            self._cast_perish(p, mv, rank, origin, tgt)
        elif shape == "flamethrower":
            self._cast_flamethrower(p, mv, rank, origin, dirv)
        elif shape == "smoke":
            self._cast_smoke(p, mv, rank, origin, tgt)
        elif shape == "fly":
            self._cast_fly(p, mv, rank, origin, tgt)
        elif shape == "rockwave":
            self._cast_rockwave(p, mv, rank, origin, dirv)
        elif shape == "smackdown":
            self._cast_smackdown(p, mv, rank, origin)
        elif shape == "sandboost":
            self._cast_sandboost(p, mv, rank, origin)
        elif shape == "spikes":
            self._cast_spikes(p, mv, rank, origin, tgt)
        elif shape == "rockcharge":
            self._cast_rockcharge(p, mv, rank, origin, tgt, dirv)
        elif shape == "avalanche":
            self._cast_avalanche(p, mv, rank, origin, tgt)
        elif shape == "explosion":
            self._cast_explosion(p, mv, rank, origin)
        elif any(getattr(self, name)(p, mv, rank, origin, tgt, dirv)
                 for name in hook_names("cast")):
            pass   # mossa di una linea con logica propria (engine/lines.py)
        elif shape == "buff":
            self._apply_support(p, mv, rank, origin)
            self.fx_events.append({"kind": "buff", "c": (origin.x, origin.y),
                                   "mtype": mv.type, "radius": 1.0})
        # Autodebuff (Zuffa): chi la lancia resta scoperto per qualche secondo.
        sdb = mv.effects.get("self_debuff")
        if sdb:
            dd = sdb.get("def_debuff")
            if dd:
                self._apply_def_debuff(p, float(dd.get("amount", 0)),
                                       float(dd.get("duration", 4.0)),
                                       float(dd.get("mult", 1.0)))
            sd = sdb.get("spdef_debuff")
            if sd:
                self._apply_spdef_debuff(p, float(sd.get("amount", 0)),
                                         float(sd.get("duration", 4.0)))
        # Autoblocco (Incendio): dopo il lancio Charizard resta inchiodato (non si
        # muove né attacca) per qualche istante. Usa uno stordimento su sé stesso.
        self._self_slow(p, mv)   # Martelpugno: chi lo usa rallenta
        sr = mv.effects.get("self_root")
        if sr:
            dur = float(sr.get("duration", 1.0))
            p.stop()
            p.stun_until = max(p.stun_until, self.time + dur)
        for ev in self.fx_events[_fx0:]:
            if "kind" in ev:
                if mv.fx:
                    ev.setdefault("fx", mv.fx)
                # Chi l'ha lanciata e quale mossa: solo per le regolazioni VISIVE
                # dell'editor delle animazioni (punto di partenza del proiettile).
                ev.setdefault("src", p.id)
                ev.setdefault("move", mv.id)

    def _cast_sub(self, p, mv, rank, origin, tgt, dirv):
        """SUB: si inabissa (non bersagliabile e inattaccabile) ma resta
        controllabile e si muove sott'acqua più in fretta. Riemerge a fine durata
        o ri-lanciando la R, con un'onda d'urto che fa danno attorno al punto
        raggiunto (stile Vladimir + Tham Kench)."""
        dur = float(mv.effects.get("untargetable", {}).get("duration", 3.0))
        p.submerged = True
        p.submerge_until = self.time + dur
        setattr(p, "_submerge_move", (mv, rank))
        self._apply_untargetable(p, dur + 0.05)
        self._apply_speed(p, 1.4, dur)   # più veloce sott'acqua, per riposizionarsi
        p.stop()                          # parte fermo: lo guida il giocatore
        self.fx_events.append({"kind": "dive", "c": (p.pos.x, p.pos.y),
                               "mtype": mv.type})

    # --- mosse della linea Gastly -------------------------------------------
    def _cast_bolt(self, p, mv, rank, origin, tgt):
        """Colpo ad aggancio automatico: prende il nemico più vicino al punto
        mirato entro gittata e lo colpisce subito (Velenoshock, Mangiasogni)."""
        want_sleep = "bonus_vs_sleep" in mv.effects   # Mangiasogni: preferisci i dormienti
        need_sleep = bool(mv.effects.get("requires_sleep"))  # colpisce SOLO i dormienti
        best, bk = None, 1e9
        for e in self._combat_hash.query_radius(origin.x, origin.y, mv.range + 1.0):
            if e.team == p.team or not getattr(e, "alive", False):
                continue
            if not isinstance(e, (Pokemon, Minion, Wild, Structure)):
                continue
            asleep = isinstance(e, Pokemon) and e.sleep_until > self.time
            if need_sleep and not asleep:
                continue   # Mangiasogni: niente danno se il bersaglio non dorme
            if origin.dist(e.pos) <= mv.range + getattr(e, "radius", 0.0):
                k = tgt.dist(e.pos)
                if want_sleep and asleep:
                    k -= 100.0   # i bersagli addormentati hanno priorità
                if k < bk:
                    best, bk = e, k
        if best is not None:
            self._hit(p, mv, rank, best)
            self.fx_events.append({"kind": "projectile", "a": (origin.x, origin.y),
                                   "b": (best.pos.x, best.pos.y), "mtype": mv.type})

    def _cast_blink(self, p, mv, rank, origin, tgt, dirv):
        if p.dashlock_until > self.time:   # Cucitombra: teletrasporti bloccati
            return
        """Spettrotuffo: teletrasporto attraverso i muri; la gittata cresce con le
        cariche di Corpofunesto; dopo `return_delay` torna al punto di partenza."""
        bonus = p.soul_charges * float(mv.effects.get("charge_range", 0.04))
        rng = mv.range + bonus
        to = tgt - origin
        dest = origin + dirv * min(to.length(), rng)
        dest = self._nearest_walkable(dest, origin)
        p.blink_return_pos = Vec2(origin.x, origin.y)
        p.blink_return_until = self.time + float(mv.effects.get("return_delay", 1.5))
        p.blink_return_fx = mv.fx   # solo grafica: stile del ritorno
        p.pos = dest
        p.stop()
        # Poof ai due capi + animazione Hop veloce (effetto teletrasporto).
        self.fx_events.append({"kind": "blink", "a": (origin.x, origin.y),
                               "b": (dest.x, dest.y), "mtype": mv.type})
        self.fx_events.append({"eid": p.id, "anim": "Hop", "anim_speed": 2.5})

    def _cast_curse(self, p, mv, rank, origin, tgt):
        """Maledizione: marchia un nemico; dopo `delay` subisce un danno alto che
        cresce con le cariche di Corpofunesto."""
        best, bk = None, 1e9
        for e in self._combat_hash.query_radius(origin.x, origin.y, mv.range + 1.0):
            if e.team == p.team or not getattr(e, "alive", False) or not isinstance(e, Pokemon):
                continue
            if origin.dist(e.pos) <= mv.range:
                k = tgt.dist(e.pos)
                if k < bk:
                    best, bk = e, k
        if best is None:
            return
        base = mv.power_at(rank) + mv.ratio * p.eff_sp_attack()
        best.curse_power = base + p.soul_charges * float(mv.effects.get("charge_power", 2.0))
        best.curse_until = self.time + float(mv.effects.get("delay", 3.0))
        best.curse_src = p.id
        setattr(best, "curse_type", mv.type)
        setattr(best, "curse_fx", mv.fx)   # solo grafica: stile dell'esplosione
        self.fx_events.append({"kind": "buff", "c": (best.pos.x, best.pos.y),
                               "mtype": mv.type, "radius": 1.2})

    def _update_blink_return(self, dt: float) -> None:
        for p in self.world.pokemons():
            if p.blink_return_pos is not None and p.blink_return_until > 0.0 \
                    and self.time >= p.blink_return_until:
                if p.alive:
                    src = Vec2(p.pos.x, p.pos.y)
                    p.pos = Vec2(p.blink_return_pos.x, p.blink_return_pos.y)
                    p.stop()
                    ev = {"kind": "blink", "a": (src.x, src.y),
                          "b": (p.pos.x, p.pos.y), "mtype": "Ghost"}
                    if getattr(p, "blink_return_fx", None):   # solo grafica
                        ev["fx"] = p.blink_return_fx
                        ev["ret"] = 1
                    self.fx_events.append(ev)
                    self.fx_events.append({"eid": p.id, "anim": "Hop", "anim_speed": 2.5})
                p.blink_return_until = -1.0
                p.blink_return_pos = None

    def _update_curses(self, dt: float) -> None:
        for p in self.world.pokemons():
            if p.curse_until > 0.0 and self.time >= p.curse_until:
                src = self.world.get(p.curse_src) if p.curse_src is not None else None
                if p.alive and src is not None:
                    dmg = self._mitigate_hit(src, p, p.curse_power, magic=True)
                    self._deal_damage(src, p, dmg, dmg_type=getattr(p, "curse_type", "Ghost"))
                    self._vamp(src, dmg, physical=False)
                    ev = {"kind": "area", "c": (p.pos.x, p.pos.y),
                          "radius": 1.6, "mtype": "Ghost"}
                    if getattr(p, "curse_fx", None):   # solo grafica
                        ev["fx"] = p.curse_fx + "pop"
                    self.fx_events.append(ev)
                p.curse_until = -1.0
                p.curse_src = None
                p.curse_power = 0.0

    # --- mosse della linea Rowlet -------------------------------------------
    def _cast_fan(self, p, mv, rank, origin, dirv):
        """Foglielama: ventaglio di 3 proiettili."""
        spread = math.radians(float(mv.d.get("spread", 16)))
        base = math.atan2(dirv.y, dirv.x)
        for a in (base - spread, base, base + spread):
            self._cast_projectile(p, mv, rank, origin, Vec2(math.cos(a), math.sin(a)))

    def _cast_dashback(self, p, mv, rank, origin, tgt, dirv):
        """Retromarcia: scatto verso il bersaglio; se colpisce, rimbalza indietro."""
        if p.dashlock_until > self.time:
            return
        dist = min((tgt - origin).length(), mv.range)
        dest = self._clamp_to_wall(origin, origin + dirv * dist)
        dest, grounded = self._intercept_dash(p, origin, dest)   # Abbattimento nemico attivo
        dist = origin.dist(dest)
        hit_any = False
        for e in self._combat_hash.query_radius((origin.x + dest.x) / 2,
                                                (origin.y + dest.y) / 2, dist + mv.radius + 1.0):
            if e.team == p.team or not getattr(e, "alive", False):
                continue
            if self._point_seg_dist(e.pos, origin, dest) <= mv.radius + getattr(e, "radius", 0.0):
                self._hit(p, mv, rank, e)
                hit_any = True
        p.pos = dest
        self.fx_events.append({"kind": "dash", "a": (origin.x, origin.y),
                               "b": (dest.x, dest.y), "mtype": mv.type, "hit": hit_any})
        if hit_any and not grounded:   # rimbalzo all'indietro (direzione opposta)
            back = self._clamp_to_wall(dest, dest - dirv * float(mv.d.get("recoil", mv.range)))
            self.fx_events.append({"kind": "dash", "a": (dest.x, dest.y),
                                   "b": (back.x, back.y), "mtype": mv.type, "ret": True})
            p.pos = back
        p.stop()

    def _cast_rain(self, p, mv, rank, origin, tgt):
        """Pioggia di Frecce: dopo un ritardo, l'area colpisce e rallenta a intervalli."""
        to = tgt - origin
        center = origin + to.normalized() * min(to.length(), mv.range) if to.length() > 1e-6 else Vec2(tgt.x, tgt.y)
        delay = float(mv.effects.get("delay", 1.0))
        self.area_effects.append({
            "center": Vec2(center.x, center.y), "radius": mv.radius,
            "start": self.time + delay, "end": self.time + delay + float(mv.effects.get("duration", 3.0)),
            "next": self.time + delay, "interval": 0.5,
            "caster_id": p.id, "mv": mv, "rank": rank, "mtype": mv.type, "fx": mv.fx})
        self.fx_events.append({"kind": "area", "c": (center.x, center.y),
                               "radius": mv.radius, "mtype": mv.type})

    def _cast_storm(self, p, mv, rank, origin, tgt):
        """VerdeBufera: zona persistente che pulsa danni a intervalli. Chi la
        lancia, restando dentro, accumula cariche extra di Inversione
        (sovraccarico: il tetto sale da 50 a 100)."""
        to = tgt - origin
        center = (origin + to.normalized() * min(to.length(), mv.range)
                  if to.length() > 1e-6 else Vec2(tgt.x, tgt.y))
        delay = float(mv.effects.get("delay", 0.0))
        self.area_effects.append({
            "center": Vec2(center.x, center.y), "radius": mv.radius,
            "start": self.time + delay,
            "end": self.time + delay + float(mv.effects.get("duration", 4.0)),
            "next": self.time + delay,
            "interval": float(mv.effects.get("interval", 0.75)),
            "caster_id": p.id, "mv": mv, "rank": rank, "mtype": mv.type,
            "fx": "", "inversion_owner": True})
        # Zona visibile fin dal lancio (fx "" così resta un evento "zone", non uno
        # stile, a meno che la mossa non abbia uno stile suo: tag solo cosmetico).
        self.fx_events.append({"kind": "zone", "c": (center.x, center.y),
                               "radius": mv.radius, "mtype": mv.type, "fx": mv.fx or ""})

    def _cast_leap(self, p, mv, rank, origin, tgt, dirv):
        """VastoImpatto: Serperior salta e, atterrando, infligge ingenti danni
        nell'area del punto d'arrivo (lo slow è negli effects della mossa)."""
        to = tgt - origin
        dist = min(to.length(), mv.range)
        dest = self._nearest_walkable(origin + dirv * dist, origin)
        dest, _ = self._intercept_dash(p, origin, dest)   # Abbattimento nemico attivo
        p.pos = dest
        p.stop()
        for e in self._combat_hash.query_radius(dest.x, dest.y, mv.radius + 1.0):
            if e.team == p.team or not getattr(e, "alive", False):
                continue
            if dest.dist(e.pos) <= mv.radius + getattr(e, "radius", 0.0):
                self._hit(p, mv, rank, e)
        self.fx_events.append({"kind": "dash", "a": (origin.x, origin.y),
                               "b": (dest.x, dest.y), "mtype": mv.type})
        self.fx_events.append({"kind": "area", "c": (dest.x, dest.y),
                               "radius": mv.radius, "mtype": mv.type})

    def _cast_chain(self, p, mv, rank, origin, tgt, dirv):
        """Aeroassalto: scatta tra un massimo di `max_targets` nemici vicini,
        colpendoli in sequenza; durante la sequenza non è bersagliabile."""
        if p.dashlock_until > self.time:
            return
        max_t = int(mv.d.get("max_targets", 4))
        hop = float(mv.effects.get("hop", 4.5))
        cur = Vec2(origin.x, origin.y)
        pool = []
        for e in self._combat_hash.query_radius(origin.x, origin.y, mv.range + hop * max_t):
            if e.team == p.team or not getattr(e, "alive", False):
                continue
            if not isinstance(e, (Pokemon, Minion, Wild, Structure)):
                continue
            if isinstance(e, Pokemon) and e.is_decoy:
                continue
            pool.append(e)
        chain, used = [], set()
        for _ in range(max_t):
            best, bd = None, 1e9
            for e in pool:
                if id(e) in used:
                    continue
                d = cur.dist(e.pos)
                if d <= hop + getattr(e, "radius", 0.0) and d < bd:
                    best, bd = e, d
            if best is None:
                break
            used.add(id(best))
            chain.append(best)
            cur = Vec2(best.pos.x, best.pos.y)
        if not chain:   # nessun bersaglio: breve scatto in avanti
            dest = self._clamp_to_wall(origin, origin + dirv * hop)
            p.pos = dest
            p.stop()
            self.fx_events.append({"kind": "dash", "a": (origin.x, origin.y),
                                   "b": (dest.x, dest.y), "mtype": mv.type, "hit": False})
            return
        prev = origin
        for i, e in enumerate(chain):
            tp = Vec2(e.pos.x, e.pos.y)
            # hit/tgt/i: solo grafica (graffi sul bersaglio, balzi in sequenza)
            self.fx_events.append({"kind": "dash", "a": (prev.x, prev.y),
                                   "b": (tp.x, tp.y), "mtype": mv.type,
                                   "hit": True, "tgt": e.id, "i": i})
            self._hit(p, mv, rank, e)
            prev = tp
        p.pos = self._nearest_walkable(prev, origin)
        p.stop()
        self._apply_untargetable(p, 0.15 * len(chain) + 0.2)   # intargetabile durante
        self.fx_events.append({"eid": p.id, "anim": "Hop", "anim_speed": 2.5})

    def _cast_rancore(self, p, mv, rank):
        """Rancore: attiva subito la forma Shedinja (condivide la ricarica con
        Muta). La punizione al killer è gestita in _cheat_death."""
        self._enter_shedinja(p, rancore=True)

    # --- mosse / passive della linea Poliwag ---------------------------------
    def _cast_arc(self, p, mv, rank, origin, tgt):
        """Bolla: una bolla in ARCO verso il punto mirato (telegrafato). Dopo un
        breve ritardo atterra, fa pochi danni nell'area e fa saltare (knock-up =
        breve stordimento, negli effects)."""
        to = tgt - origin
        center = (origin + to.normalized() * min(to.length(), mv.range)
                  if to.length() > 1e-6 else Vec2(tgt.x, tgt.y))
        delay = float(mv.effects.get("delay", 0.55))
        # Impatto singolo ritardato (riusa le aree persistenti con un solo tick).
        self.area_effects.append({
            "center": Vec2(center.x, center.y), "radius": mv.radius,
            "start": self.time + delay, "end": self.time + delay + 0.05,
            "next": self.time + delay, "interval": 1e9,
            "caster_id": p.id, "mv": mv, "rank": rank, "mtype": mv.type,
            "fx": mv.fx})
        # Bolla in ARCO + telegraph del punto d'atterraggio (un unico evento con
        # la durata dell'arco; l'impatto lo emette l'area_effect quando atterra).
        self.fx_events.append({"kind": "arcbubble", "a": (origin.x, origin.y),
                               "b": (center.x, center.y), "radius": mv.radius,
                               "delay": delay, "mtype": mv.type})

    def _cast_raindance(self, p, mv, rank, origin, tgt):
        """Pioggiadanza: zona di pioggia a tempo centrata su di sé. Dentro, le
        mosse Acqua degli alleati sono potenziate e conta come pioggia per
        Nuotovelox (gestito in _hit / _in_rain via self.rain_zones)."""
        dur = float(mv.effects.get("duration", 3.0)) * self._rain_dur_mult(p)   # Rocciaumida
        rad = mv.radius if mv.radius > 0 else 3.0
        self.rain_zones.append({"center": Vec2(origin.x, origin.y), "radius": rad,
                                "end": self.time + dur, "team": p.team})
        self.fx_events.append({"kind": "zone", "c": (origin.x, origin.y),
                               "radius": rad, "mtype": mv.type, "fx": mv.fx,
                               "dur": dur})   # dur: solo grafica

    def _cast_healzone(self, p, mv, rank, origin, tgt):
        """Acquanello: cura nel tempo a sé e agli alleati vicini (la zona segue
        chi la lancia)."""
        dur = float(mv.effects.get("duration", 4.0))
        rad = mv.radius if mv.radius > 0 else 3.0
        amt = (self._rank_val(mv.effects.get("heal", []), rank)
               + float(mv.effects.get("heal_scaling", 0.0)) * p.eff_sp_attack())   # per tick
        interval = float(mv.effects.get("interval", 1.0))
        self.area_effects.append({
            "center": Vec2(origin.x, origin.y), "radius": rad,
            "start": self.time, "end": self.time + dur + 1e-3,
            "next": self.time, "interval": interval,
            "caster_id": p.id, "heal_ally": amt, "follow": p.id,
            "mtype": mv.type, "fx": mv.fx})
        self.fx_events.append({"kind": "zone", "c": (origin.x, origin.y),
                               "radius": rad, "mtype": mv.type, "fx": mv.fx,
                               "dur": dur})   # dur: solo grafica

    def _cast_throwback(self, p, mv, rank, origin, tgt, dirv):
        """Ribaltiro: afferra il nemico più vicino a portata e lo scaraventa
        DIETRO di sé, con danno e breve confusione (negli effects)."""
        best, bd = None, 1e9
        for e in self._combat_hash.query_radius(origin.x, origin.y, mv.range + 1.0):
            if e.team == p.team or not getattr(e, "alive", False) or not isinstance(e, Pokemon):
                continue
            if e.is_decoy:
                continue
            d = origin.dist(e.pos)
            if d <= mv.range + e.radius and d < bd:
                best, bd = e, d
        if best is None:
            return
        can_move = best.untargetable_until <= self.time
        start = Vec2(best.pos.x, best.pos.y)
        self._hit(p, mv, rank, best)   # danno + confusione (negli effects)
        if best.alive and can_move:
            behind = self._nearest_walkable(origin - dirv * float(mv.d.get("toss", 2.5)), origin)
            # Non teletrasporta: il nemico si BLOCCA, viene sollevato in arco e
            # atterra dietro al lanciatore (vedi _update_tosses). Durante il volo
            # è fermo (stun) così si vede il movimento.
            dur = float(mv.effects.get("toss_time", 0.4))
            best.stop()
            best.stun_until = max(best.stun_until, self.time + dur)
            self.tosses.append({"pid": best.id, "start": start,
                                "end": Vec2(behind.x, behind.y),
                                "t0": self.time, "dur": dur, "peak": 2.2})
            self.fx_events.append({"kind": "dash", "a": (origin.x, origin.y),
                                   "b": (behind.x, behind.y), "mtype": mv.type})

    def _cast_bounce(self, p, mv, rank, origin, tgt):
        """Rimbalzo: salta in aria (non bersagliabile e fermo) per `air_time` e
        poi atterra nell'area scelta (vedi _update_air_landings)."""
        dur = float(mv.effects.get("air_time", 1.0))
        to = tgt - origin
        dest = (origin + to.normalized() * min(to.length(), mv.range)
                if to.length() > 1e-6 else Vec2(tgt.x, tgt.y))
        dest = self._nearest_walkable(dest, origin)
        p.stop()
        self._apply_untargetable(p, dur + 0.05)
        p.stun_until = max(p.stun_until, self.time + dur)   # in aria: fermo e inattivo
        self.air_landings.append({"pid": p.id, "pos": Vec2(dest.x, dest.y),
                                  "mv": mv, "rank": rank, "at": self.time + dur})
        self.fx_events.append({"eid": p.id, "anim": "Hop", "anim_speed": 1.5})
        self.fx_events.append({"kind": "area", "c": (dest.x, dest.y),
                               "radius": mv.radius, "mtype": mv.type,
                               "dur": dur})   # dur: solo grafica (mirino d'arrivo)

    def _update_air_landings(self, dt: float) -> None:
        pend = []
        for al in self.air_landings:
            if self.time < al["at"]:
                pend.append(al)
                continue
            p = self.world.get(al["pid"])
            if p is None or not p.alive:
                continue
            mv, rank, dest = al["mv"], al["rank"], al["pos"]
            p.pos = Vec2(dest.x, dest.y)
            p.untargetable_until = self.time   # atterra: torna bersagliabile
            p.stop()
            self._apply_support(p, mv, rank, dest)   # cura + scudo agli alleati
            for e in self._combat_hash.query_radius(dest.x, dest.y, mv.radius + 1.0):
                if e.team == p.team or not getattr(e, "alive", False):
                    continue
                if dest.dist(e.pos) <= mv.radius + getattr(e, "radius", 0.0):
                    self._hit(p, mv, rank, e)   # danno + stordimento (negli effects)
            self.fx_events.append({"kind": "area", "c": (dest.x, dest.y),
                                   "radius": mv.radius, "mtype": mv.type, "fx": mv.fx,
                                   "land": True})   # land: solo grafica
        self.air_landings = pend

    def _update_tosses(self, dt: float) -> None:
        """Ribaltiro: muove il nemico scaraventato lungo un arco (parabola) dal
        punto di presa fino a dietro il lanciatore, sollevandolo visivamente
        (air_z) durante il volo. Finito l'arco, lo posa e azzera il sollevamento."""
        pend = []
        for ts in self.tosses:
            p = self.world.get(ts["pid"])
            if p is None or not getattr(p, "alive", False):
                continue
            t = (self.time - ts["t0"]) / ts["dur"] if ts["dur"] > 0 else 1.0
            if t >= 1.0:
                p.pos = Vec2(ts["end"].x, ts["end"].y)
                p.air_z = 0.0
                p.stop()
                continue
            s, e = ts["start"], ts["end"]
            p.pos = Vec2(s.x + (e.x - s.x) * t, s.y + (e.y - s.y) * t)
            p.air_z = ts["peak"] * 4.0 * t * (1.0 - t)   # parabola (0 ai capi, max a metà)
            p.stop()
            pend.append(ts)
        self.tosses = pend

    def _update_rain_zones(self, dt: float) -> None:
        self.rain_zones = [rz for rz in self.rain_zones if self.time < rz["end"]]

    # --- pioggia / Piovischio / Nuotovelox / Roccia di Re --------------------
    def _iter_politoads(self):
        return [q for q in self.world.pokemons()
                if q.alive and q.passive_id == "piovischio"]

    def _in_rain(self, p) -> bool:
        """Vero se il Pokémon è in acqua o sotto la pioggia (Nuotovelox)."""
        if self.jungle.weather == "rain":
            return True
        tx, ty = int(p.pos.x), int(p.pos.y)
        if self.world.grid.tile_at(tx, ty) == TILE_RIVER:
            return True
        for rz in self.rain_zones:
            if p.pos.dist(rz["center"]) <= rz["radius"]:
                return True
        for pt in self._iter_politoads():
            if p.pos.dist(pt.pos) <= PIOVISCHIO_RADIUS:
                return True
        return False

    def _water_boost(self, pos, team) -> float:
        """+20% alle mosse Acqua lanciate da `team` dentro una sua pioggia
        (aura Piovischio di Politoad alleato o zona Pioggiadanza)."""
        for pt in self._iter_politoads():
            if pt.team == team and pos.dist(pt.pos) <= PIOVISCHIO_RADIUS:
                return WATER_BOOST
        for rz in self.rain_zones:
            if rz.get("team") == team and pos.dist(rz["center"]) <= rz["radius"]:
                return WATER_BOOST
        return 1.0

    def _fire_reduce(self, target_pos, target_team) -> float:
        """-20% alle mosse Fuoco che colpiscono un'unità dentro l'aura
        Piovischio della sua stessa squadra."""
        for pt in self._iter_politoads():
            if pt.team == target_team and target_pos.dist(pt.pos) <= PIOVISCHIO_RADIUS:
                return FIRE_REDUCE
        return 1.0

    def _give_heal(self, caster, target, amount: float) -> None:
        """Cura `target` e, se il curatore ha la Roccia di Re, dà all'alleato
        anche uno scudo pari al 30% della cura effettiva."""
        if amount <= 0.0 or not getattr(target, "alive", False):
            return
        amount *= self._heal_out_mult(caster) * self._heal_in_mult(target)   # Calmanella / Granradice
        before = target.hp
        target.hp = min(target.max_hp, target.hp + amount)
        healed = target.hp - before
        if healed > 0.0 and target is not caster:
            self._item_on_support(caster, target)   # Squama Bella
        if (healed > 0.0 and isinstance(caster, Pokemon) and target is not caster
                and "king_rock" in self._fx_items(caster)):
            self._give_shield(caster, target, healed * KINGROCK_SHIELD_FRAC, KINGROCK_SHIELD_DUR)

    def _apply_silence(self, p, dur):   # Ripeti: solo autoattacchi, niente mosse
        if self._cc_immune(p):
            return
        dur = self._tenacity_dur(p, dur)
        p.silence_until = max(p.silence_until, self.time + dur)

    def _cast_perish(self, p, mv, rank, origin, tgt):
        """Ultimocanto: area con conto alla rovescia; alla fine i nemici ancora
        dentro subiscono danno pari a una grossa % dei loro HP massimi."""
        to = tgt - origin
        center = (origin + to.normalized() * min(to.length(), mv.range)
                  if to.length() > 1e-6 else Vec2(tgt.x, tgt.y))
        countdown = float(mv.effects.get("countdown", 3.0))
        self.area_effects.append({
            "center": Vec2(center.x, center.y), "radius": mv.radius,
            "perish": float(mv.effects.get("hp_frac", 0.45)),
            "detonate_at": self.time + countdown,
            "next": self.time, "interval": 0.5,
            "caster_id": p.id, "mtype": mv.type, "fx": mv.fx})
        self.fx_events.append({"kind": "zone", "c": (center.x, center.y),
                               "radius": mv.radius, "mtype": mv.type, "fx": mv.fx,
                               "left": countdown})   # left: solo grafica (conto alla rovescia)

    def _update_area_effects(self, dt: float) -> None:
        now = self.time
        alive = []
        for ae in self.area_effects:
            # Acquanello: cura nel tempo agli alleati (la zona segue il lanciatore).
            if ae.get("heal_ally") is not None:
                caster = self.world.get(ae["caster_id"])
                if now >= ae["end"] or caster is None or not caster.alive:
                    continue
                fol = ae.get("follow")
                if fol is not None:
                    src = self.world.get(fol)
                    if src is not None and src.alive:
                        ae["center"] = Vec2(src.pos.x, src.pos.y)
                c = ae["center"]
                if now >= ae["next"]:
                    ae["next"] = now + ae["interval"]
                    amt = ae["heal_ally"] * (caster.heal_power
                                             if isinstance(caster, Pokemon) else 1.0)
                    for a in self._combat_hash.query_radius(c.x, c.y, ae["radius"] + 1.0):
                        if (isinstance(a, Pokemon) and a.team == caster.team and a.alive
                                and c.dist(a.pos) <= ae["radius"] + a.radius):
                            self._give_heal(caster, a, amt)
                    self.fx_events.append({"kind": "zone", "c": (c.x, c.y),
                                           "radius": ae["radius"], "mtype": ae["mtype"],
                                           "fx": ae.get("fx", "")})
                alive.append(ae)
                continue
            # Muro di Fumo: nube che accieca i nemici al suo interno.
            if ae.get("smoke"):
                if now >= ae["end"]:
                    continue
                c = ae["center"]
                caster = self.world.get(ae["caster_id"])
                cteam = caster.team if caster is not None else ae.get("team", -1)
                # Ogni frame: marca i NEMICI dentro (il raggio visivo cala in
                # _update_status finché smoke_until è nel futuro).
                for e in self._combat_hash.query_radius(c.x, c.y, ae["radius"] + 1.0):
                    if (isinstance(e, Pokemon) and getattr(e, "alive", False)
                            and e.team != cteam
                            and c.dist(e.pos) <= ae["radius"] + e.radius):
                        e.smoke_until = now + 0.3
                if now >= ae["next"]:
                    ae["next"] = now + ae["interval"]
                    self.fx_events.append({"kind": "zone", "c": (c.x, c.y),
                                           "radius": ae["radius"], "mtype": ae["mtype"],
                                           "fx": ae.get("fx", "")})
                alive.append(ae)
                continue
            # Levitoroccia: rocce aguzze a terra; ogni nemico dentro subisce danno
            # al massimo una volta ogni `interval` secondi (timer per nemico).
            if ae.get("spikes"):
                if now >= ae["end"]:
                    continue
                caster = self.world.get(ae["caster_id"])
                if caster is not None:
                    nh = ae["next_hit"]
                    for e in self._rock_targets(ae["team"], ae["center"], ae["radius"]):
                        if now >= nh.get(e.id, -1.0):
                            nh[e.id] = now + ae["interval"]
                            self._hit(caster, ae["mv"], ae["rank"], e)
                            self.fx_events.append({"kind": "rockspike",
                                                   "c": (e.pos.x, e.pos.y), "dir": (0.0, -1.0),
                                                   "radius": 0.6, "mtype": "Rock",
                                                   "fx": "rockspike"})
                alive.append(ae)
                continue
            # Ultimocanto: conto alla rovescia, poi danno %HP max ai nemici dentro.
            if ae.get("perish") is not None:
                caster = self.world.get(ae["caster_id"])
                c = ae["center"]
                if now >= ae["detonate_at"]:
                    if caster is not None:
                        frac = float(ae["perish"])
                        for e in self._combat_hash.query_radius(c.x, c.y, ae["radius"] + 1.0):
                            if e.team == caster.team or not getattr(e, "alive", False):
                                continue
                            if c.dist(e.pos) <= ae["radius"] + getattr(e, "radius", 0.0):
                                self._deal_damage(caster, e,
                                                  getattr(e, "max_hp", 0.0) * frac, dmg_type=None)
                    self.fx_events.append({"kind": "area", "c": (c.x, c.y),
                                           "radius": ae["radius"], "mtype": ae["mtype"],
                                           "fx": ae.get("fx", "")})
                    continue   # detonata: rimuovi
                if now >= ae["next"]:
                    ae["next"] = now + ae["interval"]
                    self.fx_events.append({"kind": "zone", "c": (c.x, c.y),
                                           "radius": ae["radius"], "mtype": ae["mtype"],
                                           "fx": ae.get("fx", ""),
                                           "left": ae["detonate_at"] - now})   # solo grafica
                alive.append(ae)
                continue
            if self.time >= ae["end"]:
                continue
            if self.time >= ae["start"] and self.time >= ae["next"]:
                ae["next"] = self.time + ae["interval"]
                caster = self.world.get(ae["caster_id"])
                c = ae["center"]
                if caster is not None:
                    # VerdeBufera: se chi l'ha lanciata è dentro la zona, va in
                    # sovraccarico (tetto 100) e accumula cariche d'Inversione.
                    if (ae.get("inversion_owner")
                            and c.dist(caster.pos) <= ae["radius"] + getattr(caster, "radius", 0.0)):
                        caster.inversion_oc_until = self.time + ae["interval"] + 0.25
                        self._add_inversion(caster, int(ae["mv"].effects.get("overcharge", 5)))
                    slow = ae["mv"].effects.get("slow") if ae.get("slow_only") else None
                    for e in self._combat_hash.query_radius(c.x, c.y, ae["radius"] + 1.0):
                        if e.team == caster.team or not getattr(e, "alive", False):
                            continue
                        if c.dist(e.pos) <= ae["radius"] + getattr(e, "radius", 0.0):
                            if slow is not None:   # Mulinello: solo rallentamento ricorrente
                                if isinstance(e, Pokemon):
                                    self._apply_slow(e, float(slow.get("mult", 1.0)),
                                                     float(slow.get("duration", 1.0)))
                            else:
                                self._hit(caster, ae["mv"], ae["rank"], e)
                if ae.get("silent"):
                    pass   # Mulinello: il vortice è un unico FX persistente, niente anelli
                elif ae.get("inversion_owner"):
                    # VerdeBufera: ripulsa la zona intera a ogni tick (campo animato continuo).
                    self.fx_events.append({"kind": "zone", "c": (c.x, c.y),
                                           "radius": ae["radius"], "mtype": ae["mtype"],
                                           "fx": getattr(ae.get("mv"), "fx", "") or ""})
                else:
                    self.fx_events.append({"kind": "area", "c": (c.x, c.y),
                                           "radius": ae["radius"] * 0.5, "mtype": ae["mtype"],
                                           "fx": ae.get("fx", "")})
            alive.append(ae)
        self.area_effects = alive

    def _is_hidden(self, p) -> bool:
        """Nascosto ai nemici: in un cespuglio o non visibile alla squadra avversaria."""
        tx, ty = int(p.pos.x), int(p.pos.y)
        if self.world.grid.tile_at(tx, ty) == TILE_BUSH:
            return True
        return not self.vision.is_enemy_visible(1 - p.team, p.id)

    def _auto_reach(self, u) -> float:
        """Gittata dell'auto-attacco (passiva Distacco: molto più larga se nascosti)."""
        r = u.attack_range
        if isinstance(u, Pokemon) and "zoom_lens" in self._fx_items(u):
            r *= 1.15   # Zoomlente: +15% portata degli attacchi base
        if isinstance(u, Pokemon) and u.passive_id == "distacco" and self._is_hidden(u):
            r += config.DISTACCO_RANGE_BONUS
        return self._claw_reach(u, r)   # Ombrartigli: +50% portata

    def _nearest_walkable(self, dest, fallback):
        if self.world.grid.is_walkable_point(dest.x, dest.y):
            return dest
        d = fallback - dest
        steps = int(d.length() * 4) + 1
        for i in range(1, steps + 1):
            pt = dest + d * (i / steps)
            if self.world.grid.is_walkable_point(pt.x, pt.y):
                return pt
        return Vec2(fallback.x, fallback.y)

    def _update_submerge(self, dt: float) -> None:
        """Gestisce lo stato SUB: riemersione a tempo o su ri-lancio della R."""
        for p in self.world.pokemons():
            if not p.submerged:
                continue
            # Ri-premere la R mentre si è immersi fa riemergere subito.
            early = p.pending_cast is not None and p.pending_cast[0] == "R"
            if early:
                p.pending_cast = None
            if not p.alive:
                p.submerged = False
                p.untargetable_until = self.time
                setattr(p, "_submerge_move", None)
                continue
            if early or self.time >= p.submerge_until:
                self._emerge(p)

    def _emerge(self, p) -> None:
        p.submerged = False
        p.untargetable_until = self.time   # torna bersagliabile all'istante
        data = getattr(p, "_submerge_move", None)
        setattr(p, "_submerge_move", None)
        if data is None:
            return
        mv, rank = data
        center = Vec2(p.pos.x, p.pos.y)
        # Doppioteam: niente onda d'urto; riemerge lasciando un clone esca.
        if "decoy" in mv.effects:
            dirv = DIR_VECTORS[p.facing].normalized()
            self._spawn_decoy(p, dirv, float(mv.effects["decoy"].get("duration", 4.0)))
            p.stop()
            ev = {"kind": "area", "c": (center.x, center.y), "radius": 1.0, "mtype": mv.type}
            if mv.fx:   # solo grafica: sbuffo del clone (linea Nincada)
                ev["fx"] = mv.fx
            self.fx_events.append(ev)
            return
        for e in self._combat_hash.query_radius(center.x, center.y, mv.radius + 1.0):
            if e.team == p.team or not getattr(e, "alive", False):
                continue
            if center.dist(e.pos) <= mv.radius + getattr(e, "radius", 0.0):
                self._hit(p, mv, rank, e)   # onda d'urto in riemersione
        p.stop()
        ev = {"kind": "area", "c": (center.x, center.y), "radius": mv.radius, "mtype": mv.type}
        if mv.fx:   # solo grafica: geyser della riemersione (linea Totodile)
            ev["fx"] = mv.fx
        self.fx_events.append(ev)

    # --- mosse della linea Charmander ----------------------------------------
    def _cast_flamethrower(self, p, mv, rank, origin, dirv):
        """Lanciafiamme: un cono di fuoco CONTINUO davanti a sé per tutta la
        durata. Non colpisce subito: registra una canalizzazione che a ogni tick
        rifà il cono dalla posizione/direzione attuali del lanciatore (così
        Charizard può muoversi LENTAMENTE mentre lo usa) finché non scade o viene
        interrotto da uno stordimento/sonno/morte."""
        dur = float(mv.effects.get("duration", 1.5))
        interval = float(mv.effects.get("interval", 0.2))
        move_mult = float(mv.effects.get("move_mult", 0.5))
        p.channel_until = self.time + dur
        self._apply_speed(p, move_mult, dur)   # si muove lentamente mentre canalizza
        self.channels.append({
            "type": "flame", "caster_id": p.id, "mv": mv, "rank": rank,
            "end": self.time + dur, "next": self.time, "interval": interval})

    def _cast_smoke(self, p, mv, rank, origin, tgt):
        """Muro di Fumo: una nube persistente. Chi (nemico) ci sta dentro ha il
        raggio visivo ridotto (vedi _update_status) per tutta la durata."""
        to = tgt - origin
        center = (origin + to.normalized() * min(to.length(), mv.range)
                  if to.length() > 1e-6 and mv.range > 0 else Vec2(origin.x, origin.y))
        dur = float(mv.effects.get("duration", 4.0))
        self.area_effects.append({
            "center": Vec2(center.x, center.y), "radius": mv.radius,
            "smoke": True, "end": self.time + dur, "team": p.team,
            "next": self.time, "interval": float(mv.effects.get("interval", 0.5)),
            "caster_id": p.id, "mtype": mv.type, "fx": mv.fx})
        self.fx_events.append({"kind": "zone", "c": (center.x, center.y),
                               "radius": mv.radius, "mtype": mv.type, "fx": mv.fx})

    def _cast_fly(self, p, mv, rank, origin, tgt):
        """Volo (stile ulti di Galio): Charizard canalizza sul posto per qualche
        istante, poi si teletrasporta al punto scelto facendo danno ad area
        all'arrivo. Il punto è telegrafato per tutta la canalizzazione e può
        essere interrotto da uno stordimento/sonno o dalla morte."""
        dest = self._nearest_walkable(Vec2(tgt.x, tgt.y), origin)
        channel = float(mv.effects.get("channel", 1.5))
        p.stop()
        p.channel_until = self.time + channel
        p.root_until = max(p.root_until, self.time + channel)   # inchiodato mentre canalizza
        self.channels.append({
            "type": "fly", "caster_id": p.id, "mv": mv, "rank": rank,
            "end": self.time + channel, "dest": Vec2(dest.x, dest.y),
            "origin": Vec2(origin.x, origin.y)})
        # Telegraph iniziale del punto d'arrivo (ripetuto in _update_channels).
        self.fx_events.append({"kind": "zone", "c": (dest.x, dest.y),
                               "radius": mv.radius, "mtype": mv.type, "fx": mv.fx})

    def _update_channels(self, dt: float) -> None:
        """Avanza le canalizzazioni (Lanciafiamme, Volo). Interruzione = chi
        canalizza muore o viene stordito/addormentato (non il suo stesso root)."""
        alive = []
        now = self.time
        for ch in self.channels:
            p = self.world.get(ch["caster_id"])
            if p is None or not getattr(p, "alive", False):
                continue   # morto: canalizzazione annullata
            interrupted = self._immobile(p)   # stun / sonno / lanciato in aria
            if ch["type"] == "explode":
                if interrupted:
                    p.channel_until = now
                    p.root_until = now
                    continue   # interrotta: niente esplosione (la ricarica resta spesa)
                if now >= ch["end"]:
                    p.channel_until = now
                    p.root_until = now
                    self._explode(p, ch["mv"], ch["rank"])
                    continue
                # Telegraph visibile ai nemici per tutta la carica (come Volo).
                self.fx_events.append({"kind": "zone", "c": (p.pos.x, p.pos.y),
                                       "radius": ch["mv"].radius, "mtype": ch["mv"].type,
                                       "fx": "explodecharge",
                                       "t": (now - ch["start"]) / max(1e-6, ch["end"] - ch["start"])})
                alive.append(ch)
                continue
            if ch["type"] == "flame":
                if interrupted or now >= ch["end"]:
                    if interrupted:
                        p.channel_until = now   # interrotto: libera subito
                    continue
                mv, rank = ch["mv"], ch["rank"]
                if now >= ch["next"]:
                    ch["next"] = now + ch["interval"]
                    dirv = DIR_VECTORS[p.facing % 8].normalized()
                    self._cast_cone(p, mv, rank, Vec2(p.pos.x, p.pos.y), dirv)
                alive.append(ch)
            elif ch["type"] == "fly":
                if interrupted:
                    p.channel_until = now
                    p.root_until = now
                    continue   # interrotto prima del decollo: niente teletrasporto
                if now >= ch["end"]:
                    mv, rank, dest = ch["mv"], ch["rank"], ch["dest"]
                    src = Vec2(p.pos.x, p.pos.y)
                    p.pos = Vec2(dest.x, dest.y)
                    p.channel_until = now
                    p.root_until = now
                    p.stop()
                    for e in self._combat_hash.query_radius(dest.x, dest.y, mv.radius + 1.0):
                        if e.team == p.team or not getattr(e, "alive", False):
                            continue
                        if dest.dist(e.pos) <= mv.radius + getattr(e, "radius", 0.0):
                            self._hit(p, mv, rank, e)
                    self.fx_events.append({"kind": "blink", "a": (src.x, src.y),
                                           "b": (dest.x, dest.y), "mtype": mv.type,
                                           "fx": mv.fx})
                    self.fx_events.append({"kind": "area", "c": (dest.x, dest.y),
                                           "radius": mv.radius, "mtype": mv.type, "fx": mv.fx})
                    continue
                # Telegraph del punto d'arrivo durante la canalizzazione
                # ("t" = avanzamento della carica, solo grafica).
                ch_t = ch["end"] - float(ch["mv"].effects.get("channel", 1.5))
                self.fx_events.append({"kind": "zone", "c": (ch["dest"].x, ch["dest"].y),
                                       "radius": ch["mv"].radius, "mtype": ch["mv"].type,
                                       "fx": ch["mv"].fx,
                                       "t": (now - ch_t) / max(1e-6, ch["end"] - ch_t)})
                alive.append(ch)
        self.channels = alive

    # --- linea Roggenrola: pilastri, Sabbiafiume, mosse ----------------------
    def _rock_targets(self, team, c, r):
        """Nemici colpibili dalle mosse di roccia entro r da c (Pokémon, minion,
        neutrali; niente strutture)."""
        out = []
        for e in self._combat_hash.query_radius(c.x, c.y, r + 1.0):
            if e.team == team or not getattr(e, "alive", False):
                continue
            if not isinstance(e, (Pokemon, Minion, Wild)):
                continue
            if c.dist(e.pos) <= r + getattr(e, "radius", 0.0):
                out.append(e)
        return out

    def _pillar_at(self, tx, ty):
        for pl in self.world.pillars:
            if pl.tx == tx and pl.ty == ty:
                return pl
        return None

    def _place_pillar(self, p, pos, explode=None):
        """Crea un pilastro di roccia sul tile di `pos` (se libero): il tile
        diventa TILE_PILLAR (blocca come un muro) per PILLAR_DURATION secondi.
        Chi ci stava sopra viene spinto fuori."""
        g = self.world.grid
        tx, ty = int(math.floor(pos.x)), int(math.floor(pos.y))
        if not g.is_walkable(tx, ty) or self._pillar_at(tx, ty) is not None:
            return None
        c = Vec2(tx + 0.5, ty + 0.5)
        for s in self.world.structures():
            if s.alive and c.dist(s.pos) < s.radius + 0.9:
                return None   # mai sopra torri/nucleo
        pl = Pillar(pos=c, tx=tx, ty=ty, team=p.team, born=self.time,
                    until=self.time + PILLAR_DURATION, prev_tile=g.tiles[ty][tx],
                    explode=explode)
        g.tiles[ty][tx] = TILE_PILLAR
        self.world.pillars.append(pl)
        self._eject_from_tile(tx, ty)
        self.fx_events.append({"kind": "pillar", "c": (c.x, c.y), "mtype": "Rock",
                               "fx": "pillar", "explosive": explode is not None})
        return pl

    def _eject_from_tile(self, tx, ty):
        """Spinge fuori dal tile (appena diventato pilastro) le unità che ci stanno."""
        g = self.world.grid
        c = Vec2(tx + 0.5, ty + 0.5)
        for u in self.world.alive_units():
            if int(math.floor(u.pos.x)) != tx or int(math.floor(u.pos.y)) != ty:
                continue
            away = u.pos - c
            base = math.atan2(away.y, away.x) if away.length() > 1e-6 else 0.0
            for k in range(8):   # prima la direzione "naturale", poi a ventaglio
                a = base + (k + 1) // 2 * (math.pi / 4) * (1 if k % 2 else -1)
                pt = c + Vec2(math.cos(a), math.sin(a)) * (0.75 + u.radius)
                if g.is_walkable_point(pt.x, pt.y):
                    u.pos = pt
                    break
            u.stop()

    def _remove_pillar(self, pl, crumble=True):
        g = self.world.grid
        if g.tiles[pl.ty][pl.tx] == TILE_PILLAR:
            g.tiles[pl.ty][pl.tx] = pl.prev_tile
        if pl in self.world.pillars:
            self.world.pillars.remove(pl)
        if crumble:
            self.fx_events.append({"kind": "pillarbreak", "c": (pl.pos.x, pl.pos.y),
                                   "mtype": "Rock", "fx": "pillarbreak"})

    def _update_pillars(self, dt: float) -> None:
        """Scadenza dei pilastri. Un cambio meteo rigenera la griglia dal terreno
        base (cancellando i pilastri): qui si ri-timbrano, salvando il tile nuovo."""
        g = self.world.grid
        for pl in list(self.world.pillars):
            if self.time >= pl.until:
                self._remove_pillar(pl)
                continue
            t = g.tiles[pl.ty][pl.tx]
            if t != TILE_PILLAR:
                pl.prev_tile = t
                g.tiles[pl.ty][pl.tx] = TILE_PILLAR

    def _pillar_burst(self, pl) -> None:
        """Pietrataglio: il pilastro contro cui è stato schiantato un nemico
        esplode, danneggiando i nemici vicini, e sparisce."""
        emv, rank, cid, r = pl.explode
        caster = self.world.get(cid)
        if caster is not None:
            for e in self._rock_targets(pl.team, pl.pos, r):
                self._hit(caster, emv, rank, e)
        self.fx_events.append({"kind": "area", "c": (pl.pos.x, pl.pos.y), "radius": r,
                               "mtype": "Rock", "fx": "rockburst"})
        self._remove_pillar(pl, crumble=False)

    # Sabbiafiume (passiva di Gigalith) / Terrempesta
    def _sand_radius(self, p):
        """(raggio, moltiplicatore) della tempesta di sabbia attorno a p; raggio 0
        = nessuna tempesta. Terrempesta la raddoppia (e la accende anche prima
        dell'evoluzione in Gigalith)."""
        boost = p.sandstorm_boost_until > self.time
        if not (boost or (p.passive_id == "sabbiafiume" and p.stage >= 2)):
            return 0.0, 1.0
        mult = 2.0 if boost else 1.0
        return SABBIAFIUME_RADIUS * mult, mult

    def _update_sandstorms(self, dt: float) -> None:
        """Tempesta di sabbia PERMANENTE attorno a Gigalith (stesso schema
        dell'aura di pioggia di Politoad, ma con danno): ogni SABBIAFIUME_TICK i
        nemici dentro (Pokémon e minion) subiscono un piccolo danno senza tipo."""
        now = self.time
        for p in self.world.pokemons():
            if not p.alive or p.is_decoy:
                p.sand_radius = 0.0
                continue
            r, mult = self._sand_radius(p)
            p.sand_radius = r
            if r <= 0.0 or now < p.sand_next:
                continue
            p.sand_next = now + SABBIAFIUME_TICK
            dmg = ((SABBIAFIUME_DPS_BASE + SABBIAFIUME_DPS_LVL * p.level)
                   * SABBIAFIUME_TICK * mult)
            for e in self._combat_hash.query_radius(p.pos.x, p.pos.y, r + 1.0):
                if e.team == p.team or not getattr(e, "alive", False):
                    continue
                if not isinstance(e, (Pokemon, Minion)):
                    continue
                if p.pos.dist(e.pos) <= r + e.radius:
                    self._deal_damage(p, e, dmg, dmg_type=None)

    def _cast_sandboost(self, p, mv, rank, origin):
        """Terrempesta: per qualche secondo la tempesta di Sabbiafiume raddoppia
        raggio e danno."""
        dur = float(mv.effects.get("duration", 4.0))
        p.sandstorm_boost_until = self.time + dur
        p.sand_next = self.time   # primo tick subito
        self.fx_events.append({"kind": "zone", "c": (origin.x, origin.y),
                               "radius": SABBIAFIUME_RADIUS * 2.0, "mtype": mv.type,
                               "fx": "sandburst"})

    # Q: Rocciotomba / Pietrataglio / Abbattimento
    def _cast_rockwave(self, p, mv, rank, origin, dirv):
        """Rocciotomba: un'onda di rocce corre in linea retta (danno + rallenta
        ogni nemico una volta); a fine corsa spunta un pilastro di roccia.
        Pietrataglio: il pilastro è esplosivo (vedi _pillar_burst)."""
        explode = None
        ex = mv.effects.get("explode")
        if ex:
            emv = Move(slot=mv.slot, max_rank=mv.max_rank, d={
                "id": mv.id + "_burst", "name": mv.name, "type": mv.type,
                "category": mv.category, "power": ex.get("power", [60]),
                "ratio": ex.get("ratio", 0.3), "scale": mv.d.get("scale", "")})
            explode = (emv, rank, p.id, float(ex.get("radius", 2.0)))
        self.rock_waves.append({
            "pos": Vec2(origin.x, origin.y), "start": Vec2(origin.x, origin.y),
            "dir": Vec2(dirv.x, dirv.y), "left": mv.range, "speed": mv.speed,
            "radius": mv.radius, "team": p.team, "caster_id": p.id, "mv": mv,
            "rank": rank, "hit": set(), "fx_next": 0.0, "explode": explode})

    def _cast_smackdown(self, p, mv, rank, origin):
        """Abbattimento: colpo ad area attorno a sé (annulla Volo in corso e
        blocca gli scatti dei colpiti). La zona resta ATTIVA per un po': uno
        scatto nemico che la attraversa si ferma al bordo, subisce il colpo e
        non può più scattare (vedi _intercept_dash)."""
        hit = set()
        for e in self._rock_targets(p.team, origin, mv.radius):
            self._hit(p, mv, rank, e)
            hit.add(e.id)
        dur = float(mv.effects.get("zone", 2.0))
        self.smackdown_zones.append({"c": Vec2(origin.x, origin.y), "radius": mv.radius,
                                     "end": self.time + dur, "team": p.team,
                                     "caster_id": p.id, "mv": mv, "rank": rank})
        self.fx_events.append({"kind": "area", "c": (origin.x, origin.y),
                               "radius": mv.radius, "mtype": mv.type, "fx": "smackdown"})
        self.fx_events.append({"kind": "zone", "c": (origin.x, origin.y),
                               "radius": mv.radius, "dur": dur, "mtype": mv.type,
                               "fx": "smackzone"})

    def _intercept_dash(self, p, origin, dest):
        """Abbattimento attivo: uno scatto di `p` che entra in una zona nemica si
        ferma al bordo e `p` subisce il colpo (niente scatti per qualche
        secondo). Ritorna (destinazione, intercettato)."""
        for z in self.smackdown_zones:
            if z["team"] == p.team or self.time >= z["end"]:
                continue
            c, r = z["c"], z["radius"] + getattr(p, "radius", 0.0)
            if self._point_seg_dist(c, origin, dest) > r:
                continue
            d = dest - origin
            n = int(d.length() * 8) + 1
            stop = Vec2(origin.x, origin.y)
            for i in range(n + 1):
                pt = origin + d * (i / n)
                if pt.dist(c) <= r:
                    stop = pt
                    break
            caster = self.world.get(z["caster_id"])
            if caster is not None:
                self._hit(caster, z["mv"], z["rank"], p)
            self.fx_events.append({"kind": "area", "c": (stop.x, stop.y), "radius": 1.0,
                                   "mtype": z["mv"].type, "fx": "smackdown"})
            return stop, True
        return dest, False

    # W: Levitoroccia
    def _cast_spikes(self, p, mv, rank, origin, tgt):
        """Levitoroccia: sparge rocce aguzze in un'area per `duration`; ogni
        nemico che ci sta dentro subisce danno, al massimo una volta ogni
        `interval` secondi (vedi _update_area_effects, ramo "spikes")."""
        to = tgt - origin
        center = (origin + to.normalized() * min(to.length(), mv.range)
                  if to.length() > 1e-6 else Vec2(origin.x, origin.y))
        dur = float(mv.effects.get("duration", 8.0))
        self.area_effects.append({
            "center": Vec2(center.x, center.y), "radius": mv.radius, "spikes": True,
            "end": self.time + dur, "interval": float(mv.effects.get("interval", 1.5)),
            "next_hit": {}, "team": p.team, "caster_id": p.id, "mv": mv, "rank": rank,
            "mtype": mv.type, "fx": mv.fx})
        self.fx_events.append({"kind": "zone", "c": (center.x, center.y),
                               "radius": mv.radius, "dur": dur, "mtype": mv.type,
                               "fx": "spikefield"})

    # E: Pesobomba (+ Bottintesta / Lucidatura)
    def _cast_rockcharge(self, p, mv, rank, origin, tgt, dirv):
        """Pesobomba: carica in avanti per `range` (tick per tick, vedi
        _update_rock_moves). Trascina i nemici colpiti; se sbatte contro un muro
        o un pilastro genera un'onda d'urto che lancia in aria i nemici vicini."""
        if p.dashlock_until > self.time:
            return
        dur = mv.range / max(0.1, mv.speed)
        p.stop()
        p.channel_until = self.time + dur + 0.1   # niente mosse/attacchi mentre carica
        self.rock_charges.append({"pid": p.id, "dir": Vec2(dirv.x, dirv.y),
                                  "left": mv.range, "speed": mv.speed, "mv": mv,
                                  "rank": rank, "hit": set(), "drag": [],
                                  "fx_next": 0.0})

    # R: Frana / Esplosione
    def _cast_avalanche(self, p, mv, rank, origin, tgt):
        """Frana: una grande valanga di massi compare in un punto lontano e
        rotola verso Gigalith (rallenta + lancia in aria chi attraversa). Se
        Gigalith la colpisce con Pesobomba la rilancia nella direzione della
        carica, più veloce e con un lancio in aria più lungo."""
        to = tgt - origin
        dist = max(float(mv.effects.get("min_range", 4.0)), min(to.length(), mv.range))
        dirv = to.normalized() if to.length() > 1e-6 else DIR_VECTORS[p.facing % 8].normalized()
        spawn = self._nearest_walkable(origin + dirv * dist, origin)
        back = origin - spawn
        roll = back.normalized() if back.length() > 1e-6 else Vec2(-dirv.x, -dirv.y)
        delay = float(mv.effects.get("delay", 0.6))
        travel = back.length() + float(mv.effects.get("overshoot", 2.5))
        self.avalanches.append({
            "pos": Vec2(spawn.x, spawn.y), "dir": roll, "left": travel,
            "speed": mv.speed, "radius": mv.radius, "start": self.time + delay,
            "team": p.team, "caster_id": p.id, "mv": mv, "rank": rank, "hit": set(),
            "air": float(mv.effects.get("air", 0.75)), "relaunched": False,
            "spin": 0.0, "born": self.time})
        self.fx_events.append({"kind": "zone", "c": (spawn.x, spawn.y),
                               "b": (spawn.x + roll.x * travel, spawn.y + roll.y * travel),
                               "radius": mv.radius, "delay": delay, "mtype": mv.type,
                               "fx": "avalanchewarn"})

    def _relaunch_avalanche(self, av, dirv) -> None:
        mv = av["mv"]
        av["dir"] = Vec2(dirv.x, dirv.y)
        av["speed"] = mv.speed * float(mv.effects.get("relaunch_speed", 1.6))
        av["air"] = float(mv.effects.get("relaunch_air", 1.4))
        av["left"] = float(mv.effects.get("relaunch_range", 9.0))
        av["hit"] = set()
        av["relaunched"] = True
        self.fx_events.append({"kind": "area", "c": (av["pos"].x, av["pos"].y),
                               "radius": av["radius"] + 0.4, "mtype": mv.type,
                               "fx": "shockwave"})

    def _cast_explosion(self, p, mv, rank, origin):
        """Esplosione: Gigalith si carica per `channel` secondi (visibile ai
        nemici, interrompibile da stordimento/sonno/lancio in aria come il Volo),
        poi esplode (vedi _explode)."""
        ch = float(mv.effects.get("channel", 1.5))
        p.stop()
        p.channel_until = self.time + ch
        p.root_until = max(p.root_until, self.time + ch)
        self.channels.append({"type": "explode", "caster_id": p.id, "mv": mv, "rank": rank,
                              "start": self.time, "end": self.time + ch})
        self.fx_events.append({"kind": "zone", "c": (origin.x, origin.y),
                               "radius": mv.radius, "mtype": mv.type,
                               "fx": "explodecharge", "t": 0.0})

    def _explode(self, p, mv, rank) -> None:
        """Fine carica di Esplosione: danno ad area + lancio in aria (negli
        effects) a tutti i nemici vicini; Gigalith perde metà degli HP attuali."""
        c = Vec2(p.pos.x, p.pos.y)
        for e in self._rock_targets(p.team, c, mv.radius):
            self._hit(p, mv, rank, e)
        cost = float(mv.effects.get("hp_cost_frac", 0.5))
        p.hp = max(1.0, p.hp * (1.0 - cost))
        self.fx_events.append({"kind": "area", "c": (c.x, c.y), "radius": mv.radius,
                               "mtype": mv.type, "fx": "explosion"})
        self.fx_events.append({"eid": p.id, "anim": "Hurt"})

    def _update_rock_moves(self, dt: float) -> None:
        now = self.time
        self.smackdown_zones = [z for z in self.smackdown_zones if now < z["end"]]
        self._update_rock_waves(dt)
        self._update_rock_charges(dt)
        self._update_avalanches(dt)

    def _update_rock_waves(self, dt: float) -> None:
        g = self.world.grid
        alive = []
        for w in self.rock_waves:
            caster = self.world.get(w["caster_id"])
            step = w["speed"] * dt
            nxt = w["pos"] + w["dir"] * min(step, max(0.0, w["left"]))
            blocked = not g.is_walkable_point(nxt.x, nxt.y)
            if not blocked:
                w["pos"] = nxt
                w["left"] -= step
            if caster is not None:
                for e in self._rock_targets(w["team"], w["pos"], w["radius"]):
                    if e.id not in w["hit"]:
                        w["hit"].add(e.id)
                        self._hit(caster, w["mv"], w["rank"], e)   # danno + rallentamento
            if self.time >= w["fx_next"]:   # rocce che spuntano dietro al fronte dell'onda
                w["fx_next"] = self.time + 0.045
                self.fx_events.append({"kind": "rockspike", "c": (w["pos"].x, w["pos"].y),
                                       "dir": (w["dir"].x, w["dir"].y), "radius": w["radius"],
                                       "mtype": "Rock", "fx": "rockspike"})
            if blocked or w["left"] <= 0.0:
                # Fine corsa: pilastro di roccia (non a ridosso di chi l'ha lanciata).
                if caster is not None and w["pos"].dist(w["start"]) >= 1.2:
                    self._place_pillar(caster, w["pos"], w["explode"])
                continue
            alive.append(w)
        self.rock_waves = alive

    def _update_rock_charges(self, dt: float) -> None:
        g = self.world.grid
        alive = []
        for ch in self.rock_charges:
            p = self.world.get(ch["pid"])
            if p is None or not p.alive or self._immobile(p):
                if p is not None:
                    p.channel_until = self.time
                continue   # carica interrotta (morto, stordito, lanciato in aria)
            mv, dirv = ch["mv"], ch["dir"]
            # Abbattimento nemico attivo: la carica si ferma lì.
            if any(z["team"] != p.team and p.pos.dist(z["c"]) <= z["radius"] + p.radius
                   for z in self.smackdown_zones):
                self._intercept_dash(p, p.pos, p.pos)
                self._end_charge(p, ch)
                continue
            step = min(ch["speed"] * dt, max(0.0, ch["left"]))
            newpos = p.pos + dirv * step
            # Impatto: Gigalith sbatte contro un muro/pilastro, oppure un nemico
            # TRASCINATO davanti a lui viene schiantato contro l'ostacolo.
            blocked_at = newpos + dirv * (p.radius * 0.8)
            if g.is_walkable_point(blocked_at.x, blocked_at.y):
                blocked_at = None
                for eid in ch["drag"]:
                    e = self.world.get(eid)
                    if e is None or not e.alive:
                        continue
                    front = newpos + dirv * (p.radius + 2.0 * e.radius + 0.05)
                    if not g.is_walkable_point(front.x, front.y):
                        blocked_at = front
                        break
            if blocked_at is not None:
                self._charge_impact(p, ch, blocked_at)
                continue
            p.pos = newpos
            p.stop()
            ch["left"] -= step
            # Colpisce (una volta) e TRASCINA i nemici sul percorso.
            for e in self._rock_targets(p.team, p.pos, p.radius + 0.45):
                if e.id in ch["hit"]:
                    continue
                ch["hit"].add(e.id)
                self._hit(p, mv, ch["rank"], e)
                if (e.alive and isinstance(e, (Pokemon, Minion))
                        and not self._cc_immune(e)
                        and not (isinstance(e, Pokemon) and e.untargetable_until > self.time)):
                    ch["drag"].append(e.id)
            for eid in ch["drag"]:
                e = self.world.get(eid)
                if e is None or not e.alive:
                    continue
                pt = p.pos + dirv * (p.radius + e.radius + 0.05)
                if g.is_walkable_point(pt.x, pt.y):
                    e.pos = pt
                e.stop()
                if isinstance(e, Pokemon):
                    e.stun_until = max(e.stun_until, self.time + 0.12)   # spostato: non agisce
            # Frana: Gigalith colpisce la propria valanga e la rilancia.
            for av in self.avalanches:
                if (av["team"] == p.team and not av["relaunched"] and self.time >= av["start"]
                        and av["pos"].dist(p.pos) <= av["radius"] + p.radius + 0.3):
                    self._relaunch_avalanche(av, dirv)
                    ch["left"] = 0.0
            if self.time >= ch["fx_next"]:
                ch["fx_next"] = self.time + 0.05
                self.fx_events.append({"kind": "dash", "a": (p.pos.x - dirv.x * 0.6,
                                                             p.pos.y - dirv.y * 0.6),
                                       "b": (p.pos.x, p.pos.y), "mtype": "Rock",
                                       "fx": "chargedust"})
            if ch["left"] <= 0.0:
                self._end_charge(p, ch)
                continue
            alive.append(ch)
        self.rock_charges = alive

    def _end_charge(self, p, ch) -> None:
        p.channel_until = self.time
        p.stop()

    def _charge_impact(self, p, ch, ahead) -> None:
        """Pesobomba contro un muro o un pilastro: onda d'urto che lancia in aria
        i nemici vicini. Pietrataglio: se un nemico viene schiantato contro un
        pilastro esplosivo, il pilastro esplode. Lucidatura: +velocità."""
        mv = ch["mv"]
        c = Vec2(p.pos.x, p.pos.y)
        for e in self._rock_targets(p.team, c, SHOCKWAVE_RADIUS):
            if isinstance(e, Pokemon):
                self._apply_airborne(e, SHOCKWAVE_AIR, 1.1)
        self.fx_events.append({"kind": "area", "c": (c.x, c.y), "radius": SHOCKWAVE_RADIUS,
                               "mtype": mv.type, "fx": "shockwave"})
        pl = self._pillar_at(int(math.floor(ahead.x)), int(math.floor(ahead.y)))
        if pl is None:   # il punto d'urto può cadere a cavallo del tile: il più vicino
            near = [q for q in self.world.pillars if q.pos.dist(ahead) <= 1.0]
            pl = min(near, key=lambda q: q.pos.dist(ahead)) if near else None
        if pl is not None and pl.explode is not None:
            slammed = any(self.world.get(eid) is not None and self.world.get(eid).alive
                          for eid in ch["drag"])
            if not slammed:   # nemico incastrato tra Gigalith e il pilastro
                slammed = any(isinstance(e, Pokemon) and e.pos.dist(pl.pos) <= 1.3
                              for e in self._rock_targets(p.team, pl.pos, 1.3))
            if slammed:
                self._pillar_burst(pl)
        isp = mv.effects.get("impact_speed")
        if isp:
            self._apply_speed(p, float(isp.get("mult", 1.35)), float(isp.get("duration", 3.0)))
            self.fx_events.append({"kind": "buff", "c": (c.x, c.y), "mtype": mv.type,
                                   "radius": 1.0})
        self._end_charge(p, ch)

    def _update_avalanches(self, dt: float) -> None:
        g = self.world.grid
        alive = []
        for av in self.avalanches:
            if self.time < av["start"]:
                alive.append(av)
                continue
            caster = self.world.get(av["caster_id"])
            step = av["speed"] * dt
            nxt = av["pos"] + av["dir"] * step
            if not g.is_walkable_point(nxt.x, nxt.y):
                self.fx_events.append({"kind": "area", "c": (av["pos"].x, av["pos"].y),
                                       "radius": av["radius"], "mtype": "Rock",
                                       "fx": "rockburst"})
                continue   # si schianta contro un muro/pilastro
            av["pos"] = nxt
            av["left"] -= step
            av["spin"] += step / max(0.3, av["radius"])
            if caster is not None:
                for e in self._rock_targets(av["team"], av["pos"], av["radius"]):
                    if e.id in av["hit"]:
                        continue
                    av["hit"].add(e.id)
                    self._hit(caster, av["mv"], av["rank"], e)   # danno + rallentamento
                    if isinstance(e, Pokemon):
                        self._apply_airborne(e, av["air"], 1.4 if av["relaunched"] else 1.0)
            if av["left"] <= 0.0:
                self.fx_events.append({"kind": "area", "c": (av["pos"].x, av["pos"].y),
                                       "radius": av["radius"], "mtype": "Rock",
                                       "fx": "rockburst"})
                continue
            alive.append(av)
        self.avalanches = alive

    def _cast_bubble(self, p, mv, rank, origin, dirv):
        """Proiettile che VIAGGIA (Idropulsar, Fuocobomba). Con `burst` esplode in
        un'area al contatto o a fine gittata (Fuocobomba)."""
        self.projectiles.append({
            "pos": Vec2(origin.x, origin.y), "dir": Vec2(dirv.x, dirv.y),
            "speed": mv.speed, "left": mv.range, "radius": mv.radius,
            "team": p.team, "caster_id": p.id, "mv": mv, "rank": rank, "mtype": mv.type,
            "burst": float(mv.d.get("burst", 0.0)), "fx": mv.fx})
        muzzle = origin + dirv * 1.3
        self.fx_events.append({"kind": "projectile", "a": (origin.x, origin.y),
                               "b": (muzzle.x, muzzle.y), "mtype": mv.type})

    def _update_projectiles(self, dt: float) -> None:
        alive = []
        for pr in self.projectiles:
            prev = Vec2(pr["pos"].x, pr["pos"].y)
            step = pr["speed"] * dt
            pr["pos"] = pr["pos"] + pr["dir"] * step
            pr["left"] -= step
            burst = pr.get("burst", 0.0)
            if burst > 0.0:   # Fuocobomba lenta: scia di fuoco visibile mentre viaggia
                self.fx_events.append({"kind": "projectile", "a": (prev.x, prev.y),
                                       "b": (pr["pos"].x, pr["pos"].y),
                                       "mtype": pr["mtype"], "fx": pr["mv"].fx})
            if not self.world.grid.is_walkable_point(pr["pos"].x, pr["pos"].y):
                if burst > 0.0:   # Fuocobomba: esplode anche contro un muro
                    self._projectile_burst(pr)
                continue   # colpisce un muro: svanisce
            hit = None
            for e in self._combat_hash.query_radius(pr["pos"].x, pr["pos"].y, pr["radius"] + 1.0):
                if e.team == pr["team"] or not getattr(e, "alive", False):
                    continue
                if not isinstance(e, (Pokemon, Minion, Structure, Wild)):
                    continue
                if pr["pos"].dist(e.pos) <= pr["radius"] + getattr(e, "radius", 0.0):
                    hit = e
                    break
            if hit is not None:
                caster = self.world.get(pr["caster_id"])
                if burst > 0.0:        # Fuocobomba: esplosione ad area all'impatto
                    self._projectile_burst(pr)
                elif caster is not None:
                    self._hit(caster, pr["mv"], pr["rank"], hit)
                    ev = {"kind": "area", "c": (pr["pos"].x, pr["pos"].y),
                          "radius": pr["radius"], "mtype": pr["mtype"]}
                    if pr.get("fx"):   # Armor Cannon: esplosione dedicata
                        ev["fx"] = pr["fx"] + "hit"
                    self.fx_events.append(ev)
                continue   # proiettile consumato
            if pr["left"] > 0:
                alive.append(pr)
            elif burst > 0.0:          # Fuocobomba: esplode a fine gittata
                self._projectile_burst(pr)
        self.projectiles = alive

    def _projectile_burst(self, pr) -> None:
        """Esplosione ad area di un proiettile (Fuocobomba): danneggia tutti i
        nemici entro `burst` dal punto d'impatto + anello visivo."""
        caster = self.world.get(pr["caster_id"])
        c = pr["pos"]
        br = pr.get("burst", pr["radius"])
        if caster is not None:
            for e in self._combat_hash.query_radius(c.x, c.y, br + 1.0):
                if e.team == pr["team"] or not getattr(e, "alive", False):
                    continue
                if not isinstance(e, (Pokemon, Minion, Structure, Wild)):
                    continue
                if c.dist(e.pos) <= br + getattr(e, "radius", 0.0):
                    self._hit(caster, pr["mv"], pr["rank"], e)
        # Esplosione: anello generico colorato per tipo (fx di viaggio escluso).
        self.fx_events.append({"kind": "area", "c": (c.x, c.y),
                               "radius": br, "mtype": pr["mtype"], "fx": pr["mv"].fx})

    def _cast_projectile(self, p, mv, rank, origin, dirv):
        best, best_proj = None, mv.range + 1.0
        mid = origin + dirv * (mv.range * 0.5)
        for e in self._combat_hash.query_radius(mid.x, mid.y, mv.range + 2.0):
            if e.team == p.team or not getattr(e, "alive", False):
                continue
            rel = e.pos - origin
            proj = rel.x * dirv.x + rel.y * dirv.y
            if proj < 0 or proj > mv.range:
                continue
            perp = abs(rel.x * (-dirv.y) + rel.y * dirv.x)
            if perp <= mv.radius + getattr(e, "radius", 0.0):
                if not mv.wall_pierce and not self.world.grid.line_of_sight(
                        origin.x, origin.y, e.pos.x, e.pos.y):
                    continue
                if proj < best_proj:
                    best_proj, best = proj, e
        end = origin + dirv * (best_proj if best else mv.range)
        if best is not None:
            self._hit(p, mv, rank, best)
        ev = {"kind": "projectile", "a": (origin.x, origin.y),
              "b": (end.x, end.y), "mtype": mv.type, "hit": best is not None}
        if best is not None:
            ev["hit_id"] = best.id   # cosmetico: chi e' stato colpito (grafica)
        self.fx_events.append(ev)

    def _cast_cone(self, p, mv, rank, origin, dirv):
        half = math.radians(mv.angle)
        hits = []   # cosmetico: dove sono i colpiti (morsi della linea Totodile)
        for e in self._combat_hash.query_radius(origin.x, origin.y, mv.range + 1.0):
            if e.team == p.team or not getattr(e, "alive", False):
                continue
            rel = e.pos - origin
            d = rel.length()
            if d > mv.range + getattr(e, "radius", 0.0) or d < 1e-6:
                if d >= 1e-6:
                    continue
            if d >= 1e-6:
                cosang = (rel.x * dirv.x + rel.y * dirv.y) / d
                if math.acos(max(-1.0, min(1.0, cosang))) > half:
                    continue
            self._hit(p, mv, rank, e)
            if len(hits) < 8:   # piatta [x0, y0, x1, y1, ...]: passa online (enc_fx)
                hits += [round(e.pos.x, 2), round(e.pos.y, 2)]
        ev = {"kind": "cone", "o": (origin.x, origin.y), "dir": (dirv.x, dirv.y),
              "range": mv.range, "angle": mv.angle, "mtype": mv.type}
        if mv.fx:   # anche i tick del Lanciafiamme (fuori da _cast_move): solo grafica
            ev["fx"] = mv.fx
            if hits:
                ev["hits"] = hits
        self.fx_events.append(ev)

    def _cast_area(self, p, mv, rank, origin, tgt):
        to = tgt - origin
        d = to.length()
        if mv.target == "self" or mv.range <= 0:
            center = origin
        elif d > mv.range:
            center = origin + to.normalized() * mv.range
        else:
            center = tgt
        for e in self._combat_hash.query_radius(center.x, center.y, mv.radius + 1.0):
            if e.team == p.team or not getattr(e, "alive", False):
                continue
            if center.dist(e.pos) <= mv.radius + getattr(e, "radius", 0.0):
                self._hit(p, mv, rank, e)
        self._apply_support(p, mv, rank, center)
        self.fx_events.append({"kind": "area", "c": (center.x, center.y),
                               "radius": mv.radius, "mtype": mv.type})

    def _cast_whirlpool(self, p, mv, rank, origin, tgt):
        """Vortice PERSISTENTE a terra per tutta la sua durata (Mulinello,
        Turbofuoco). Registra una zona che pulsa a intervalli e un unico effetto
        visivo "vortex" che gira per lo stesso tempo (non un lampo).

        Mulinello (slow_only, default): danno UNA volta, poi la zona RI-APPLICA
        solo il rallentamento. Turbofuoco (slow_only=false): la zona pulsa DANNO
        (e gli effetti della mossa, p.es. radice) a ogni tick -> intrappola e
        brucia nel tempo chi ci resta dentro."""
        to = tgt - origin
        d = to.length()
        center = (origin + to.normalized() * mv.range if d > mv.range
                  else Vec2(tgt.x, tgt.y))
        dur = float(mv.effects.get("duration", 1.5))
        interval = float(mv.effects.get("interval", 0.3))
        slow_only = bool(mv.d.get("slow_only", True))
        for e in self._combat_hash.query_radius(center.x, center.y, mv.radius + 1.0):
            if e.team == p.team or not getattr(e, "alive", False):
                continue
            if center.dist(e.pos) <= mv.radius + getattr(e, "radius", 0.0):
                self._hit(p, mv, rank, e)
        self.area_effects.append({
            "center": Vec2(center.x, center.y), "radius": mv.radius,
            "start": self.time + interval, "end": self.time + dur,
            "next": self.time + interval, "interval": interval,
            "caster_id": p.id, "mv": mv, "rank": rank, "mtype": mv.type,
            "fx": mv.fx, "silent": True, "slow_only": slow_only})
        self.fx_events.append({"kind": "vortex", "c": (center.x, center.y),
                               "radius": mv.radius, "dur": dur, "mtype": mv.type})

    def _cast_dash(self, p, mv, rank, origin, tgt, dirv):
        if p.dashlock_until > self.time:   # Cucitombra: scatti bloccati
            return
        to = tgt - origin
        dist = min(to.length(), mv.range)
        dest = origin + dirv * dist
        if not mv.wall_pierce:
            dest = self._clamp_to_wall(origin, dest)
        dest, _ = self._intercept_dash(p, origin, dest)   # Abbattimento nemico attivo
        dist = origin.dist(dest)
        # Colpisce i nemici lungo il tragitto.
        hit_any = False
        for e in self._combat_hash.query_radius((origin.x + dest.x) / 2,
                                                (origin.y + dest.y) / 2,
                                                dist + mv.radius + 1.0):
            if e.team == p.team or not getattr(e, "alive", False):
                continue
            if self._point_seg_dist(e.pos, origin, dest) <= mv.radius + getattr(e, "radius", 0.0):
                self._hit(p, mv, rank, e)
                hit_any = True
        # Erboscivolata: passare attraverso un nemico rende subito una carica.
        if hit_any and mv.refund_on_hit and mv.charges > 0:
            ms = p.moveset.get(mv.slot)
            if ms is not None:
                if ms.charges_cur < 0:
                    ms.charges_cur = float(mv.charges)
                ms.charges_cur = min(float(mv.charges), ms.charges_cur + 1.0)
        p.pos = dest
        p.stop()
        self._apply_support(p, mv, rank, dest)
        # Nitrocarica: ogni scatto dà una carica di velocità (cumulabile fino a N).
        ss = mv.effects.get("speed_stack")
        if ss:
            self._apply_flamecharge(p, float(ss.get("per", 0.12)),
                                    int(ss.get("max", 3)), float(ss.get("duration", 4.0)))
        self.fx_events.append({"kind": "dash", "a": (origin.x, origin.y),
                               "b": (dest.x, dest.y), "mtype": mv.type, "hit": hit_any})

    def _apply_flamecharge(self, p, per, maxst, dur):
        """Nitrocarica: accumula una carica di velocità (fino a maxst). Le cariche
        si azzerano se scadono prima del prossimo scatto; il bonus è applicato in
        _update_status (p.speed)."""
        if p.fc_until <= self.time:
            p.fc_stacks = 0
        p.fc_stacks = min(maxst, p.fc_stacks + 1)
        p.fc_per = per
        p.fc_until = self.time + dur

    def _clamp_to_wall(self, origin, dest):
        d = dest - origin
        dist = d.length()
        if dist < 1e-6:
            return dest
        steps = int(dist * 4) + 1
        last = origin
        for i in range(1, steps + 1):
            t = i / steps
            pt = origin + d * t
            if not self.world.grid.is_walkable_point(pt.x, pt.y):
                return last
            last = pt
        return dest

    @staticmethod
    def _point_seg_dist(pt, a, b):
        ab = b - a
        l2 = ab.length_sq()
        if l2 < 1e-9:
            return pt.dist(a)
        t = max(0.0, min(1.0, ((pt.x - a.x) * ab.x + (pt.y - a.y) * ab.y) / l2))
        proj = Vec2(a.x + ab.x * t, a.y + ab.y * t)
        return pt.dist(proj)

    def _hit(self, caster, mv, rank, target, mult: float = 1.0) -> None:
        """`mult` scala il danno (tick di una zona, es. Bora)."""
        prev = self._dmg_label
        self._dmg_label = mv.name        # riepilogo della morte: danno "da mossa"
        try:
            self._hit_core(caster, mv, rank, target, mult)
        finally:
            self._dmg_label = prev

    def _hit_core(self, caster, mv, rank, target, mult: float = 1.0) -> None:
        # Bersaglio non bersagliabile (es. SUB sott'acqua): niente danno né CC.
        if isinstance(target, Pokemon) and target.untargetable_until > self.time:
            return
        if self._mien_parry(caster, target, mv.shape in PROJ_SHAPES):   # Individua
            return
        fx = mv.effects
        power = mv.power_at(rank)
        if power > 0:
            # Danno abilita' = (base + ratio × stat offensiva). Le speciali usano
            # l'AP (eff_sp_attack) e poi il moltiplicatore SpA (ability_mult); le
            # fisiche usano l'AD. Sopra: STAB ×1,15, efficacia dei tipi compressa
            # (×0,6..×1,5) e meteo. La mitigazione (con penetrazione e riduzione
            # fissa) ed eventuale critico sono in _mitigate_hit.
            magic = mv.category == "special"
            if mv.d.get("scale") == "defense" and isinstance(caster, Pokemon):
                # Linea Roggenrola: il danno delle MOSSE scala con la Difesa (non
                # con l'Attacco né con l'AP); niente moltiplicatore SpA. La
                # categoria (fisica/speciale) decide solo la difesa che la mitiga.
                raw = power + mv.ratio * caster.eff_defense()
            else:
                raw = power + mv.ratio * (caster.eff_sp_attack() if magic
                                          else caster.eff_attack())
                if magic and isinstance(caster, Pokemon):
                    raw *= caster.ability_mult * self._calm_mult(caster)   # Calmamente
                    raw *= self._spa_down_mult(caster)   # Dragobolide: Att. Sp. ridotta
            eff = (self.types.moba_multiplier(mv.type, target.types)
                   if isinstance(target, Pokemon) else 1.0)
            eff = self._moth_eff(caster, eff)   # Lentifume' (Venomoth): mai poco efficace
            has_plate = (isinstance(caster, Pokemon) and caster.judgment_until > self.time
                         and mv.type == caster.judgment_plate)
            stab = STAB_MULT if (mv.type in caster.types or has_plate) else 1.0
            raw *= eff * stab * self.jungle.weather_damage_mult(mv.type)
            # Pioggia locale (Piovischio/Pioggiadanza): +20% alle mosse Acqua
            # alleate; -20% alle mosse Fuoco che colpiscono un alleato nell'aura.
            if self.rain_zones or (mv.type in ("Water", "Fire")
                                   and any(q.passive_id == "piovischio"
                                           for q in self.world.pokemons())):
                if mv.type == "Water" and isinstance(caster, Pokemon):
                    raw *= self._water_boost(caster.pos, caster.team)
                elif mv.type == "Fire" and isinstance(target, Pokemon):
                    raw *= self._fire_reduce(target.pos, target.team)
            # Bonus contro bersagli con stati alterati (Sciagura) o addormentati (Mangiasogni).
            if isinstance(target, Pokemon):
                if "bonus_vs_status" in fx and self._has_status(target):
                    raw *= float(fx["bonus_vs_status"])
                if "bonus_vs_sleep" in fx and target.sleep_until > self.time:
                    raw *= float(fx["bonus_vs_sleep"])
                if "bonus_vs_attacking" in fx and target.aa_active:   # Sbigoattacco
                    raw *= float(fx["bonus_vs_attacking"])
            bat = fx.get("bonus_if_ally_type")   # Erbapatto: +% con un alleato Acqua/Fuoco
            if bat and isinstance(caster, Pokemon):
                need = set(bat.get("types", []))
                if any(a is not caster and a.team == caster.team and a.alive
                       and need & set(a.types) for a in self.world.pokemons()):
                    raw *= float(bat.get("mult", 1.05))
            if mv.d.get("fury") and isinstance(target, Pokemon):  # Sfuriate: ramp fino a 5×
                raw *= self._fury_mult(caster, target)
            # Fuocardore (Flareon): mosse Fuoco +30% dopo averne subita una.
            if (mv.type == "Fire" and isinstance(caster, Pokemon)
                    and caster.flashfire_until > self.time):
                raw *= 1.3
            # Fendifoglia: più danno quanti meno HP ha il bersaglio.
            if "execute" in fx and getattr(target, "max_hp", 0) > 0:
                raw *= 1.0 + float(fx["execute"]) * (1.0 - target.hp / target.max_hp)
            raw *= mult
            raw *= self._hatt_dmg_mult(caster, target, mv)   # Quiete / Strega della Foresta
            raw *= self._swine_dmg_mult(caster, target, mv)   # Valanga: x2 su chi l'ha colpito
            raw *= self._scy_move_mult(caster, mv, rank)   # Tecnico (Scizor): +40%
            raw *= self._deli_move_mult(caster, target, mv)   # pacchi di Delibird, Liofilizzazione
            raw *= self._sun_mult(caster, mv)   # sole locale (Rocciacalda): Fuoco +30%, Acqua -30%
            raw *= self._item_move_mult(caster, mv, target, eff)   # Scelta, Metronomo, ...
            crit_bonus = HIGH_CRIT_CHANCE if mv.d.get("high_crit") else 0.0
            crit_bonus += float(mv.d.get("crit_bonus", 0.0))   # Lacerazione: +20%
            # Finta: ignora gli scudi e le protezioni (difesa potenziata a tempo).
            guard = bool(fx.get("pierce_guard"))
            self._shield_break(target, fx)   # Bruciatutto: brucia parte dello scudo
            dmg = self._mitigate_hit(caster, target, raw, magic=magic,
                                     crit_bonus=crit_bonus,
                                     pierce_guard=guard or bool(fx.get("ignore_def_buffs")),
                                     can_crit=bool(mv.d.get("can_crit")),   # Aerasoio
                                     vs_defense=bool(mv.d.get("vs_defense")))   # Psicoshock
            self._deal_damage(caster, target, dmg, dmg_type=mv.type, ignore_shield=guard)
            self._vamp(caster, dmg, physical=not magic)
            self._item_after_move_hit(caster, mv, target, eff)   # Assorbisfera, Vulnerassicurazione
            if "heal_self_ratio" in fx and isinstance(caster, Pokemon):  # Mangiasogni
                caster.hp = min(caster.max_hp, caster.hp + dmg * float(fx["heal_self_ratio"])
                                * self._heal_in_mult(caster))
            if "recoil" in fx and isinstance(caster, Pokemon) and caster.alive:
                # Sdoppiatore / Fuococarica: contraccolpo pari a una % del danno.
                caster.hp = max(1.0, caster.hp - dmg * float(fx["recoil"]))
        # Effetti di controllo/debuff sui nemici colpiti.
        if isinstance(target, Pokemon) and target.alive:
            self._apply_hit_effects(caster, target, fx)
            # Sincronismo (Espeon): chi gli applica uno stato lo subisce a sua volta.
            if (target.passive_id == "sincronismo" and isinstance(caster, Pokemon)
                    and caster.team != target.team and caster.alive):
                sync = {k: v for k, v in fx.items() if k in _SYNC_STATUSES}
                if sync:
                    self._apply_hit_effects(target, caster, sync)
                    self.fx_events.append({"kind": "sync",
                                           "a": (target.pos.x, target.pos.y),
                                           "b": (caster.pos.x, caster.pos.y),
                                           "mtype": "Psychic", "fx": "sync"})
        # Passiva Fiammacoda: ogni colpo di una mossa Fuoco dà una carica di Calore
        # al nemico; a 3 cariche scatta la Scottatura (danno nel tempo + -25% ATT).
        if (isinstance(caster, Pokemon) and caster.passive_id == "fiammacoda"
                and mv.type == "Fire" and isinstance(target, Pokemon)
                and target.alive and target.team != caster.team):
            self._apply_heat(target, caster)
        # Passiva Inversione: ogni spell a segno su un Pokémon nemico dà cariche
        # (non sui minion né sulle strutture).
        if (isinstance(caster, Pokemon) and isinstance(target, Pokemon)
                and target.team != caster.team):
            self._add_inversion(caster)
        self._sneasel_on_hit(caster, target)   # Arraffalesto, colpo d'agguato di Notte Fonda
        self._swine_on_hit(caster, target, mv)   # Gelo (accumuli), Frana (stordimento)
        self._deli_on_hit(caster, target, mv)    # Propulsione Quark: mossa Elettro subita

    def _apply_hit_effects(self, caster, target, fx) -> None:
        """Controlli e debuff di una mossa sul Pokémon colpito."""
        fx = self._mirror_armor(caster, target, fx)   # Specchiarmatura (Corviknight)
        slow = fx.get("slow")
        if slow:
            self._apply_slow(target, float(slow.get("mult", 0.6)),
                             float(slow.get("duration", 1.0)))
        cc = fx.get("stun")
        if cc:
            self._apply_stun(target, float(cc.get("duration", 1.0)))
        fz = fx.get("freeze")
        if fz:
            self._apply_freeze(target, float(fz.get("duration", 1.0)))
        bl = fx.get("blind")
        if bl:
            self._apply_blind(target, float(bl.get("duration", 1.5)))
        cf = fx.get("confuse")
        if cf:
            self._apply_confuse(target, float(cf.get("duration", 1.0)))
        si = fx.get("silence")
        if si:
            self._apply_silence(target, float(si.get("duration", 2.0)))
        dd = fx.get("def_debuff")
        if dd:
            self._apply_def_debuff(target, float(dd.get("amount", 0)),
                                   float(dd.get("duration", 3.0)),
                                   float(dd.get("mult", 1.0)))
        sd = fx.get("spdef_debuff")
        if sd:
            self._apply_spdef_debuff(target, float(sd.get("amount", 20)),
                                     float(sd.get("duration", 2.0)),
                                     float(sd.get("mult", 1.0)))
        rt = fx.get("root")
        if rt:
            self._apply_root(target, float(rt.get("duration", 1.0)))
        sl = fx.get("sleep")
        if sl:
            self._apply_sleep(target, float(sl.get("duration", 1.5)),
                              float(sl.get("delay", 0.0)))
        po = fx.get("poison")
        if po:
            # Laccioerboso: il danno nel tempo può scalare sugli HP max del bersaglio.
            dps = float(po.get("dps", 0))
            hp_frac = float(po.get("hp_frac", 0.0))
            if hp_frac > 0.0:
                dps += target.max_hp * hp_frac
            self._apply_poison(target, caster, dps,
                               float(po.get("duration", 2.0)))
        fe = fx.get("fear")
        if fe and isinstance(caster, (Pokemon, Minion, Structure)):
            self._apply_fear(target, caster.pos, float(fe.get("duration", 1.0)))
        dl = fx.get("dashlock")
        if dl:
            self._apply_stun(target, float(dl.get("stun", 0.6)))
            self._apply_dashlock(target, float(dl.get("duration", 2.5)))
        ad = fx.get("atk_debuff")
        if ad:
            self._apply_atk_debuff(target, float(ad.get("amount", 20)),
                                   float(ad.get("duration", 3.0)))
        kn = fx.get("airborne")
        if kn:
            self._apply_airborne(target, float(kn.get("duration", 1.0)),
                                 float(kn.get("peak", 1.0)))
        if fx.get("ground"):   # Abbattimento: annulla Volo in corso
            self._cancel_fly(target)
        ta = fx.get("taunt")
        if ta and isinstance(caster, Pokemon) and caster.team != target.team:
            self._apply_taunt(target, caster, float(ta.get("duration", 1.5)))
        if fx.get("steal_buff") and caster.team != target.team:   # Spennata
            self._steal_buff(caster, target)
        self._charcadet_hit_effects(caster, target, fx)   # Lavasbuffo: scottatura
        self._sneasel_hit_effects(caster, target, fx)   # Furto, Dire Claw, cure ridotte
        self._hatt_hit_effects(caster, target, fx)   # Magifiamma: -Att. Sp.

    def _apply_support(self, caster, mv, rank, center) -> None:
        """Effetti su sé/alleati: cura, scudo, buff attacco/velocità."""
        eff = mv.effects
        if not eff:
            return
        if mv.target == "ally":
            targets = [a for a in self._combat_hash.query_radius(center.x, center.y, mv.radius + 1.0)
                       if isinstance(a, Pokemon) and a.team == caster.team and a.alive
                       and center.dist(a.pos) <= mv.radius + a.radius]
            if caster not in targets:
                targets.append(caster)
        else:
            targets = [caster]
        # Panciamburo: sacrifica una frazione degli HP ATTUALI prima del buff.
        hc = eff.get("hp_cost_frac")
        if hc and isinstance(caster, Pokemon):
            caster.hp = max(1.0, caster.hp * (1.0 - float(hc)))
        # Potere cure/scudi del lanciatore: cure e scudi × heal_power.
        hp_power = caster.heal_power if isinstance(caster, Pokemon) else 1.0
        heal = eff.get("heal")
        if heal:
            amt = self._rank_val(heal, rank)
            if isinstance(caster, Pokemon):
                amt += float(eff.get("heal_scaling", 0.0)) * caster.eff_sp_attack()
            amt *= hp_power
            if eff.get("weather_heal"):   # Sintesi: x2 col Sole, dimezzata con la Pioggia
                w = self.jungle.weather
                amt *= 2.0 if w == "sun" else (0.5 if w == "rain" else 1.0)
            for a in targets:
                self._give_heal(caster, a, amt)   # + scudo Roccia di Re sugli alleati
        if "heal_self_ratio" in eff:
            self._give_heal(caster, caster,
                            mv.power_at(rank) * float(eff["heal_self_ratio"]) * hp_power)
        sh = eff.get("shield")
        if sh:
            amt = self._rank_val(sh.get("amount", []), rank)
            if isinstance(caster, Pokemon):
                amt += float(eff.get("shield_scaling", 0.0)) * caster.eff_sp_attack()
            amt *= hp_power
            dur = float(sh.get("duration", 3.0))
            for a in targets:
                self._give_shield(caster, a, amt, dur)   # Creta Luce
        # Corpetto Assalto: le mosse di potenziamento statistiche (con bonus
        # passivo) non danno il potenziamento attivo a chi lo indossa.
        vest = (mv.passive is not None and isinstance(caster, Pokemon)
                and "assault_vest" in self._fx_items(caster))
        boosted = [a for a in targets if not (vest and a is caster)]
        atk = eff.get("atk")
        if atk and not vest:
            self._apply_atk(caster, float(atk.get("mult", 1.3)), float(atk.get("duration", 5.0)))
        spd = eff.get("speed")
        if spd:
            for a in boosted:
                self._apply_speed(a, float(spd.get("mult", 1.2)), float(spd.get("duration", 3.0)))
        dfb = eff.get("def")   # Indurimento / Ferroscudo: difesa temporanea
        if dfb:
            m = dfb.get("mult", 1.2)
            m = self._rank_val(m, rank) if isinstance(m, list) else float(m)   # per grado
            for a in boosted:
                self._apply_def_buff(a, m, float(dfb.get("duration", 4.0)))
        cm = eff.get("calm")
        if cm and not vest:   # Calmamente: +Att. Sp. e +Dif. Sp. temporanei
            self._apply_calm(caster, float(cm.get("mult", 1.15)), float(cm.get("duration", 5.0)))
        ram = eff.get("rampage")
        if ram:   # Oltraggio: inarrestabile per qualche secondo
            self._apply_rampage(caster, float(ram.get("duration", 4.0)))

    @staticmethod
    def _rank_val(arr, rank):
        if not arr:
            return 0.0
        return float(arr[min(max(1, rank), len(arr)) - 1])

    # --- stati ---------------------------------------------------------------
    def _cc_immune(self, p):
        return getattr(p, "cc_immune_until", -1) > self.time

    def _tenacity_dur(self, p, dur):
        """Tenacia: accorcia la durata dei CC (durata × (1 − tenacia))."""
        ten = getattr(p, "tenacity", 0.0) if isinstance(p, Pokemon) else 0.0
        dur = dur * (1.0 - ten) * self._eevee_cc_mult(p)   # Forza Interiore / Fogliamanto
        if isinstance(p, Pokemon) and p.items and "snowball" in self._fx_items(p):
            dur *= SNOWBALL_TEN   # Pallaneve: controlli -25%
        return dur

    def _apply_slow(self, p, mult, dur):
        if self._cc_immune(p) or self._ursa_slow_immune(p):   # Luna Piena
            return
        dur = self._tenacity_dur(p, dur)
        if isinstance(p, Pokemon) and p.items and "acro_bike" in self._fx_items(p):
            dur *= ACRO_SLOW   # Bici Cross: rallentamenti -30%
        p.slow_mult = min(p.slow_mult, mult) if self.time <= p.slow_until else mult
        p.slow_until = max(p.slow_until, self.time + dur)

    def _apply_stun(self, p, dur):
        if self._cc_immune(p):
            return
        dur = self._tenacity_dur(p, dur)
        p.stun_until = max(p.stun_until, self.time + dur)

    def _apply_confuse(self, p, dur):
        if self._cc_immune(p):
            return
        dur = self._tenacity_dur(p, dur)
        p.confuse_until = max(p.confuse_until, self.time + dur)

    def _apply_def_debuff(self, p, amt, dur, mult=1.0):
        p.def_debuff_amt = amt
        p.def_debuff_mult = mult
        p.def_debuff_until = max(p.def_debuff_until, self.time + dur)

    def _apply_spdef_debuff(self, p, amt, dur, mult=1.0):
        p.spdef_debuff_amt = amt
        p.spdef_debuff_mult = mult
        p.spdef_debuff_until = max(p.spdef_debuff_until, self.time + dur)

    def _apply_atk_debuff(self, p, amt, dur):   # Danzadipiume: -Attacco
        p.atk_debuff_amt = amt
        p.atk_debuff_until = max(p.atk_debuff_until, self.time + dur)

    def _apply_root(self, p, dur):          # paralisi (rooted): fermo ma può agire
        if self._cc_immune(p):
            return
        dur = self._tenacity_dur(p, dur)
        p.root_until = max(p.root_until, self.time + dur)

    def _apply_sleep(self, p, dur, delay=0.0):
        if self._cc_immune(p):
            return
        dur = self._tenacity_dur(p, dur)
        if delay > 0.0:
            p.sleep_arm_at = self.time + delay   # Ipnosi: dorme dopo il ritardo
            p.sleep_arm_dur = dur
        else:
            p.sleep_until = max(p.sleep_until, self.time + dur)

    def _apply_poison(self, p, src, dps, dur):
        if self._flor_status_immune(p):   # Campo Nebbioso
            return
        p.poison_dps = max(p.poison_dps, dps)
        p.poison_until = max(p.poison_until, self.time + dur)
        p.poison_src = src.id if src is not None else None

    def _apply_fear(self, p, src_pos, dur):     # spavento: fugge dalla fonte
        if self._cc_immune(p):
            return
        dur = self._tenacity_dur(p, dur)
        p.fear_until = max(p.fear_until, self.time + dur)
        p.fear_src = Vec2(src_pos.x, src_pos.y)

    def _apply_dashlock(self, p, dur):          # Cucitombra: niente scatti/tp
        p.dashlock_until = max(p.dashlock_until, self.time + dur)

    def _apply_airborne(self, p, dur, peak=1.0):
        """Lanciato in aria (knock-up): non si muove né lancia mosse/attacchi.
        A differenza degli altri CC NON è ridotto dalla tenacia e non viene
        accorciato né annullato da altri controlli (si tiene sempre la fine più
        lontana). Visivamente lo sprite sale e ricade (air_z, _update_status)."""
        if not isinstance(p, Pokemon) or not p.alive or p.is_decoy:
            return
        if self._cc_immune(p) or p.untargetable_until > self.time:
            return
        if "iron_ball" in self._fx_items(p):
            dur *= 0.4   # Ferropalla: lanci in aria subiti -60%
        dur *= self._octi_disp_mult(p)   # Ventose (Octillery): -50%
        now = self.time
        end = now + dur
        if p.airborne_until > now and end <= p.airborne_until:
            return   # già in aria più a lungo: il nuovo lancio non cambia nulla
        p.airborne_start = now
        p.airborne_until = end
        p.airborne_peak = peak
        p.stop()
        p.aa_active = False
        p.recalling = False
        self.fx_events.append({"kind": "knockup", "c": (p.pos.x, p.pos.y),
                               "dur": dur, "mtype": "Rock", "fx": "knockup"})

    def _cancel_fly(self, p) -> bool:
        """Abbattimento: se il bersaglio sta canalizzando Volo, lo annulla (resta
        a terra; la ricarica della R resta spesa). True se ha annullato."""
        if not isinstance(p, Pokemon):
            return False
        hit = [ch for ch in self.channels
               if ch["caster_id"] == p.id and ch["type"] == "fly"]
        if not hit:
            return False
        self.channels = [ch for ch in self.channels if ch not in hit]
        p.channel_until = self.time
        p.root_until = self.time
        self.fx_events.append({"kind": "area", "c": (p.pos.x, p.pos.y),
                               "radius": 1.2, "mtype": "Rock"})
        return True

    def _apply_heat(self, target, caster):
        """Fiammacoda: aggiunge una carica di Calore (max 3, 4s, rinnovata). Alla
        terza il bersaglio va in Scottatura (danno nel tempo + -25% Attacco) e le
        cariche si azzerano."""
        if self.time > target.heat_until:
            target.heat_stacks = 0   # cariche scadute: riparti da zero
        target.heat_stacks += 1
        target.heat_until = self.time + 4.0
        if target.heat_stacks >= 3:
            target.heat_stacks = 0
            target.heat_until = -1.0
            if not self._cc_immune(target):
                target.burn_until = self.time + 4.0
                # Danno nel tempo: 2,5% degli HP max al secondo (10% in 4s).
                target.burn_dps = max(target.burn_dps, 0.025 * target.max_hp)
                target.burn_src = caster.id
                self.fx_events.append({"kind": "area", "c": (target.pos.x, target.pos.y),
                                       "radius": 0.9, "mtype": "Fire", "fx": "charburn"})

    def _has_status(self, p) -> bool:
        """Vero se il Pokémon ha un qualunque stato alterato (per Sciagura)."""
        t = self.time
        return (p.slow_until > t or p.stun_until > t or p.sleep_until > t
                or p.root_until > t or p.poison_until > t or p.confuse_until > t
                or p.def_debuff_until > t or p.spdef_debuff_until > t
                or p.burn_until > t or p.airborne_until > t
                or p.frozen_until > t or p.charm_until > t or p.blind_until > t
                or p.taunt_until > t or p.healcut_until > t)

    def _apply_untargetable(self, p, dur):
        p.untargetable_until = max(p.untargetable_until, self.time + dur)

    def _apply_rampage(self, p, dur):
        p.rampage_until = self.time + dur
        p.cc_immune_until = self.time + dur   # Oltraggio = inarrestabile (immune ai CC)

    def _apply_shield(self, p, amt, dur):
        active = p.shield if p.shield_until > self.time else 0.0
        p.shield = max(active, amt)
        p.shield_until = self.time + dur

    def _apply_atk(self, p, mult, dur):
        p.atk_mult = mult
        p.atk_until = self.time + dur

    def _apply_def_buff(self, p, mult, dur):   # Indurimento: +difesa temporanea
        p.def_buff_mult = mult
        p.def_buff_until = self.time + dur

    def _apply_speed(self, p, mult, dur):
        p.speed_mult = mult
        p.speed_until = self.time + dur

    def _immobile(self, p) -> bool:
        """Non può né muoversi né agire: stun/congelamento, sonno o lanciato in aria."""
        return (p.stun_until > self.time or p.sleep_until > self.time
                or p.frozen_until > self.time
                or p.airborne_until > self.time)

    def _flee(self, p, dt) -> None:
        """Spavento: corre via dalla fonte (rallentato, non controllabile)."""
        p.target_id = None
        p.aa_active = False
        away = p.pos - p.fear_src
        if away.length() < 1e-6:
            away = Vec2(1.0, 0.0)
        self._move_towards(p, p.pos + away.normalized() * 3.0, dt, use_astar=False)

    def _update_status(self, dt: float) -> None:
        now = self.time
        world_get = self.world.get
        terrain = self.jungle.terrain_speed_mult
        corvi_vision = self._corvi_vision
        # Metodi dei moltiplicatori di velocita' legati una volta per tick (stesso
        # ordine di hook_names; x * 1.0 == x, quindi saltare i 1.0 non cambia nulla).
        speed_hooks = [getattr(self, name) for name in hook_names("speed_mult")]
        for p in self.world.pokemons():
            # Le ricariche (cooldown e cariche) scorrono anche da morti: al respawn
            # le mosse sono avanti/pronte come in un MOBA, non congelate.
            for ms in p.moveset.values():
                cd = ms.cooldown_remaining
                if cd > 0.0:
                    ms.cooldown_remaining = max(0.0, cd - dt)
                mv = ms.chosen
                if mv is not None:
                    # = mv.charges / mv.charge_time (proprieta' sul dict d), letti
                    # dal dict e solo se servono: quasi nessuna mossa ha cariche.
                    nch = mv.d.get("charges", 0)
                    if nch and int(nch) > 0:
                        nch = int(nch)
                        cur = ms.charges_cur
                        if cur < 0:
                            ms.charges_cur = float(nch)
                        elif cur < nch:
                            ct = float(mv.d.get("charge_time", 0.0))
                            if ct > 0.0:
                                ms.charges_cur = min(float(nch), cur + dt / ct)
            if not p.alive:
                continue
            if now > p.slow_until:
                p.slow_mult = 1.0
            if now > p.speed_until:
                p.speed_mult = 1.0
            if now > p.atk_until:
                p.atk_mult = 1.0
            if now > p.shield_until:
                p.shield = 0.0
            p.def_debuff_active = p.def_debuff_until > now
            p.spdef_debuff_active = p.spdef_debuff_until > now
            p.atk_debuff_active = p.atk_debuff_until > now
            p.def_buff_active = p.def_buff_until > now
            p.memento_weak = p.memento_weak_until > now   # Memento: Att./Att.Sp. dimezzati
            # Memento / Notte Fonda: invisibile ai nemici
            p.invisible = p.memento_until > now or p.nightfall_until > now
            if now > p.spite_until:
                p.spite_mult = 1.0
            # Sonno ritardato (Ipnosi): scatta dopo il ritardo.
            arm = p.sleep_arm_at
            if arm >= 0.0 and now >= arm:
                if not self._cc_immune(p):
                    p.sleep_until = now + p.sleep_arm_dur
                p.sleep_arm_at = -1.0
            # Veleno: danno nel tempo (attribuito alla sorgente, per le uccisioni).
            if p.poison_until > now and p.poison_dps > 0.0:
                src = world_get(p.poison_src) if p.poison_src is not None else None
                if src is not None and src.alive:
                    self._dmg_label = "Veleno"
                    self._deal_damage(src, p, p.poison_dps * dt)
                    self._dmg_label = None
            # Scottatura (Fiammacoda): danno nel tempo + Attacco ridotto (vedi
            # eff_attack, via burn_active).
            burn = p.burn_active = p.burn_until > now
            if burn and p.burn_dps > 0.0:
                src = world_get(p.burn_src) if p.burn_src is not None else None
                if src is not None and src.alive:
                    self._dmg_label = "Scottatura"
                    self._deal_damage(src, p, p.burn_dps * dt, dmg_type="Fire")
                    self._dmg_label = None
            elif not burn:
                p.burn_dps = 0.0
            passive = p.passive_id
            # Passiva Inversione: senza colpi per 10s le cariche calano di 2 ogni 0.75s.
            if passive == "inversione":
                if p.inversion_stacks > 0 and now - p.inversion_last_gain >= 10.0:
                    if now >= p.inversion_decay_next:
                        p.inversion_stacks = max(0, p.inversion_stacks - 2)
                        p.inversion_decay_next = now + 0.75
                else:
                    p.inversion_decay_next = now + 0.75   # finestra ferma finché combatte
            # Lanciato in aria: lo sprite sale e ricade a parabola (cosmetico).
            air_until = p.airborne_until
            if air_until > now:
                span = max(1e-6, air_until - p.airborne_start)
                t = (now - p.airborne_start) / span
                p.air_z = p.airborne_peak * 4.0 * t * (1.0 - t)
            elif p.airborne_start >= 0.0:
                p.airborne_start = -1.0
                p.air_z = 0.0
            # = self._immobile(p): stun/congelamento, sonno o lanciato in aria.
            immobile = (p.stun_until > now or p.sleep_until > now
                        or p.frozen_until > now or air_until > now)
            rooted = p.root_until > now   # paralisi: fermo ma può agire
            cc = (p.slow_until > now or immobile or rooted
                  or p.confuse_until > now or p.charm_until > now
                  or p.taunt_until > now)
            speed = p.move_speed_base * p.slow_mult * p.speed_mult * terrain(p)
            if passive == "nuotovelox" and self._in_rain(p):
                speed *= NUOTOVELOX_SPEED   # +20% in acqua/pioggia
            for h in speed_hooks:   # Distortozona, Scia Ghiacciata...
                m = h(p)
                if m != 1.0:
                    speed *= m
            # Nitrocarica: cariche di velocità cumulabili (scadono insieme).
            if p.fc_until > now and p.fc_stacks > 0:
                speed *= 1.0 + p.fc_per * p.fc_stacks
            elif p.fc_stacks != 0:
                p.fc_stacks = 0
            # Muro di Fumo: raggio visivo ridotto a 3 tile finché si resta nel fumo.
            if not p.is_decoy:
                p.vision = corvi_vision(p, 3.0 if p.smoke_until > now else 9.0)   # Sguardofermo: +30%
            if p.fear_until > now:
                speed *= 0.5   # spavento: fuga rallentata del 50%
                cc = True
            if immobile or rooted:
                speed = 0.0   # immobilizzato (stun/sonno) o paralizzato (rooted)
            p.cc_active = cc
            p.speed = speed

    # --- evoluzioni / scelte / punti (bot) -----------------------------------
    def _update_evolution(self, dt: float) -> None:
        for p in self.world.pokemons():
            if not p.alive:
                continue
            if evolution.check_evolution(p):
                self._recompute_stats(p, heal_delta=True)
                self.announcements.append((self.time, f"{p.name} si è evoluto!"))
            # Offerta della scelta scaglionata: Q@5, W@6, E@7, R@8.
            # Pietrastante: finche' la tiene l'ultimate non si sblocca (si offre
            # appena la vende); una R gia' in attesa torna nascosta.
            stone = "everstone" in p.items
            rms = p.moveset.get("R")
            if stone and "R" in p.pending_choices and rms is not None and rms.chosen is None:
                p.pending_choices.remove("R")
                p.offered_slots.discard("R")
            for slot, lvl in (SLOT_UPGRADE_LEVEL.items() if p.level >= 5 else ()):
                if stone and slot == "R":
                    continue
                if p.level >= lvl and slot not in p.offered_slots:
                    ms = p.moveset.get(slot)
                    ready = ms and ms.upgrades and (
                        slot == "R" or (ms.base is not None and ms.chosen is ms.base))
                    if ready:
                        p.offered_slots.add(slot)
                        p.pending_choices.append(slot)
            # Slot con un'unica opzione (ultimate fissa delle forme di Eevee):
            # nessuna scelta da fare, si applica subito (anche lontano dalla base).
            if p.pending_choices:
                for slot in list(p.pending_choices):
                    ms = p.moveset.get(slot)
                    if ms and len(ms.upgrades) == 1:
                        evolution.apply_choice(p, slot, 0)
            if p.is_bot:
                # il sostituto lascia le scelte al giocatore
                if not p.bot_standin and p.pending_choices:
                    for slot in list(p.pending_choices):
                        ms = p.moveset.get(slot)
                        if ms and ms.upgrades:
                            evolution.apply_choice(p, slot, random.randrange(len(ms.upgrades)))
                if p.level > p.skill_points_spent:   # filtro rapido: forse ha punti
                    self._bot_spend_points(p)

    def _bot_spend_points(self, p) -> None:
        for _ in range(16):
            if skills.available_points(p) <= 0:
                break
            for slot in ("R", "Q", "E", "W"):
                if skills.can_rank_up(p, slot):
                    skills.rank_up(p, slot)
                    break
            else:
                break

    def _bot_try_cast(self, p, target) -> None:
        if p.pending_cast is not None or p.silence_until > self.time:
            return
        for name in bot_cast_names(p.passive_id):   # IA delle linee (engine/lines.py)
            if getattr(self, name)(p, target):
                return
        for slot in ("R", "Q", "W", "E"):
            ms = p.moveset.get(slot)
            if ms and ms.ready and ms.chosen and ms.chosen.power_at(ms.rank) > 0:
                reach = max(self._bp(p).cast_min_reach, ms.chosen.range)
                if p.pos.dist(target.pos) <= reach:
                    p.pending_cast = (slot, target.pos.x, target.pos.y)
                    return

    # --- IA del jungler ------------------------------------------------------
    def assign_junglers(self):
        """Un jungler per squadra (l'unico con la Punizione): il primo GIOCATORE
        che ha scelto una linea da jungler (smite.JUNGLE_DEX); se nessuno, un bot
        (va in giungla a farmare)."""
        from .smite import JUNGLE_DEX
        for team in (0, 1):
            mates = [p for p in self.world.pokemons() if p.team == team]
            humans = [p for p in mates if not p.is_bot and p.dex in JUNGLE_DEX]
            bots = [p for p in mates if p.is_bot]
            if humans:
                humans[0].is_jungler = True
            elif bots:
                bots[len(bots) // 2].is_jungler = True

    def _ai_jungler(self, p, dt):
        if self._immobile(p):
            p.stop()
            return
        bp = self._bp(p)
        if not bp.utility and p.hp < p.max_hp * bp.retreat_hp:
            p.target_id = None
            self._ai.setdefault(p.id, {})["gank_id"] = None
            self._move_towards(p, p.spawn_pos, dt, use_astar=True)
            return
        st = self._ai.setdefault(p.id, {})
        # Gank in corso: lo si porta avanti finche' il bersaglio e' visibile e vivo.
        gid = st.get("gank_id")
        if gid is not None:
            t = self.world.get(gid)
            if (t is None or not t.alive or self.time > st.get("gank_until", 0.0)
                    or not self.vision.is_enemy_visible(p.team, t.id)
                    or (self._under_enemy_tower(p, t.pos) and not self._good_kill_chance(p, t))):
                st["gank_id"] = None
                st["gank_cd"] = self.time + bp.gank_cooldown
            else:
                p.target_id = t.id
                self._bot_try_cast(p, t)
                if self._within_attack(p, t):
                    p.stop(); self._face(p, t)
                else:
                    self._chase(p, t, dt)
                return
        # Minaccia: un CAMPIONE nemico vicino (i minion non si inseguono in giungla),
        # ma non sotto una torre senza buone possibilita'.
        enemy = self._bot_filter_target(p, self._acquire_enemy_champ(p, bp.aggro))
        if enemy is not None:
            p.target_id = enemy.id
            self._bot_try_cast(p, enemy)
            if self._within_attack(p, enemy):
                p.stop(); self._face(p, enemy)
            else:
                self._chase(p, enemy, dt)
            return
        # Gank mirato: solo se probabilmente fruttuoso.
        g = self._jungler_gank_target(p, st)
        if g is not None:
            st["gank_id"] = g.id
            st["gank_until"] = self.time + bp.gank_duration
            self._move_towards(p, g.pos, dt, use_astar=True)
            return
        # Farming: campo neutrale (o leggendario, se abbastanza forte) più vicino.
        wild = self._nearest_wild_for(p)
        if wild is not None:
            p.target_id = wild.id
            self._bot_smite(p, wild)
            self._bot_try_cast(p, wild)
            if self._within_attack(p, wild):
                p.stop(); self._face(p, wild)
            else:
                self._move_towards(p, wild.pos, dt, use_astar=True)
            return
        # Niente da fare: pattuglia verso il centro della giungla.
        p.target_id = None
        self._move_towards(p, Vec2(self.world.grid.width * 0.5,
                                   self.world.grid.height * 0.5), dt, use_astar=True)

    def _acquire_enemy_champ(self, p, radius):
        """Campione nemico visibile piu' vicino entro il raggio."""
        best, bd = None, 1e9
        for e in self._combat_hash.query_radius(p.pos.x, p.pos.y, radius):
            if not isinstance(e, Pokemon) or not e.alive or e.team == p.team:
                continue
            if not self.vision.is_enemy_visible(p.team, e.id):
                continue
            d = p.pos.dist(e.pos)
            if d <= radius and d < bd:
                best, bd = e, d
        return best

    def _acquire_enemy_unit(self, p, radius):
        """Nemico (campione/minion della squadra avversaria, non neutrale) più
        vicino e VISIBILE entro il raggio."""
        best, bk = None, None
        for e in self._combat_hash.query_radius(p.pos.x, p.pos.y, radius):
            if not isinstance(e, (Pokemon, Minion)) or not getattr(e, "alive", False):
                continue
            if e.team == p.team or e.team == 2:
                continue
            if not self.vision.is_enemy_visible(p.team, e.id):
                continue
            d = p.pos.dist(e.pos)
            if d > radius:
                continue
            key = (_prio(e), d)
            if bk is None or key < bk:
                best, bk = e, key
        return best

    def _nearest_wild_for(self, p):
        best, bd = None, 1e9
        for w in self.world.wilds():
            if not w.alive or w.kind == "arceus":
                continue
            if w.kind == "meteo" and p.level < 6:   # il leggendario serve un po' di livello
                continue
            d = p.pos.dist(w.pos)
            if d < bd:
                best, bd = w, d
        return best

    # --- respawn / vittoria --------------------------------------------------
    def _update_respawns(self, dt: float) -> None:
        for p in self.world.pokemons():
            if not p.alive:
                p.respawn_timer -= dt
                if p.respawn_timer <= 0.0:
                    p.alive = True
                    p.hp = p.max_hp
                    p.pos = Vec2(p.spawn_pos.x, p.spawn_pos.y)
                    p.stop()
                    p.target_id = None
                    p.idle_anchor = None   # l'ancora d'inseguimento riparte dalla base
                    p.move_dest = None     # niente ordini di movimento di prima della morte
                    p.aa_active = False

    def _check_win(self) -> None:
        for core in self.world.cores():
            if not core.alive and self.winner is None:
                self.winner = 1 - core.team
