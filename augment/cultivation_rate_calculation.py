"""
Cultivation-time distributions: the idealized exponential and the three
discretizations of it that the two compilers actually sample.

All times are in CUBES (= layers = logical cycles = CUBE_DISTANCE ECC rounds).
The single parameter `A` is the mean of the underlying continuous process, in
cubes -- it is `AVG_CULTIVATION_TIME` in ts-prod_d*.py and `1/magic_state_lambda`
in PureMagic's src/cultivation.rs.

The four processes
-----------------
Let W ~ Exp(rate 1/A) be the idealized cultivation time.

  exact      W                            the continuous ideal; nothing samples this
  ts_tick    (1/D) * ceil(W*D)            ts-prod_d15.py:2654, the fractional-tick loop:
                                          one Bernoulli per square per ECC cycle,
                                          p = 1 - exp(-(1/D)/A)
  ts_layer   ceil(W)                      ts-prod_d15.py:2357, the per-layer seeding block:
                                          one Bernoulli per square per cube,
                                          p = 1 - exp(-1/A)
  puremagic  round(W)                     PureMagic/src/cultivation.rs:51, draw-once:
                                          `rng_exp.sample().round() as i32`, then counted
                                          down by busy_counts

Both FlowRouter sites are the SAME process at two resolutions: drawing i.i.d.
Bernoulli(1 - exp(-d/A)) once per interval of width d gives a geometric count N
with P[N > k] = exp(-k*d/A) = P[W > k*d], i.e. exactly ceil(W/d)*d. That is why
(1 - p_tick)**CUBE_DISTANCE == 1 - p_layer holds identically.

Means (exact, closed form)
--------------------------
  ceil onto a grid of width d:   E = d / (1 - exp(-d/A))       = A + d/2 + d^2/(12A) + ...
  round onto a grid of width d:  E = d / (2*sinh(d/(2A)))      = A       - d^2/(24A) + ...

so ceiling carries a FIRST-order bias (+d/2) and rounding only a second-order one.
Inverting the first for a target mean T gives the calibration constant

  A = -d / ln(1 - d/T)        (equivalently: per-interval hazard p = d/T exactly)

which is what `avg_for_target` returns.

Usage
-----
  python plot_cultivation_cdfs.py                  # writes one PNG per calibration
  python plot_cultivation_cdfs.py --show           # also opens the windows
  python plot_cultivation_cdfs.py --target 2 3.5   # ad-hoc calibrations, matched means
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

RESCALED_CULTIVATION_PARAMETER = {11: 1.804104, 13: 1.706345, 15: 5.225329}

CUBE_DISTANCE = 15         # ECC rounds per cube; ts-prod_d15.py:112
CULTIVATION_LENGTH = 24     # the shortest possible real cultivation, in ECC rounds
PHYSICAL_FLOOR = CULTIVATION_LENGTH / CUBE_DISTANCE   # 1.6 cubes

_EPS = 1e-9   # keeps floor(t*D) from tripping on 2/15*15 = 2.0000000000000004



# --------------------------------------------------------- the measured cultivation chain

# Transcribed verbatim from ts-prod_d15.py / ts-prod_d13.py / ts-prod_d11.py (SINGLE_SITE_POSTSELECTION_STATS,
# GAP_DECODER_STATS, GAP_DECODER_CYCLES). Format per tuple:
# (round, n_detectors, fires_alone, first_to_fail-as-fraction-of-all-shots).
MEASURED = {
    11: dict(   # the old ts-prod_d11.py table, as it shipped (4 decimals; out/verify_d1_3_d2_11_r2_5_gap48p7 rounds to the same stats)
        stats=[(1, 7, 0.1468, 0.1468), (3, 7, 0.1911, 0.1096), (4, 6, 0.3009, 0.1125),
               (5, 3, 0.1185, 0.0488), (6, 2, 0.0886, 0.0385), (7, 1, 0.0487, 0.0228)],
        gap={'reached_gap_decision': 156_101, 'below_threshold_discards': 33_155},
        gap_cycles=10),
    13: dict(
        stats=[(1, 7, 0.146910, 0.146910), (3, 7, 0.191213, 0.109860), (4, 6, 0.300600, 0.112210),
               (5, 3, 0.119047, 0.049200), (6, 2, 0.088223, 0.038350), (7, 1, 0.048737, 0.022633)],
        gap={'reached_gap_decision': 156_050, 'below_threshold_discards': 48_275},
        gap_cycles=10),
    15: dict(
        stats=[(1, 7, 0.147767, 0.147767), (3, 12, 0.226713, 0.141803), (4, 14, 0.431957, 0.212730),
               (5, 18, 0.418930, 0.154513), (6, 18, 0.418803, 0.097653), (8, 20, 0.554160, 0.085253),
               (9, 18, 0.617980, 0.062590), (10, 8, 0.307403, 0.019853), (11, 6, 0.264343, 0.016163),
               (12, 6, 0.263093, 0.013380), (13, 6, 0.261383, 0.010120), (14, 2, 0.093673, 0.002990)],
        gap={'reached_gap_decision': 10567, 'below_threshold_discards': 1923},
        gap_cycles=10),
}
IC_ROUNDS = {11: (1, 3, 4), 13: (1, 3, 4), 15: (1, 3, 4, 5, 6, 8, 9)}   # rounds the in-patch race touches;
# everything after max(IC_ROUNDS) is the (always single-site) escape ablation + gap decision.
MULTIPLEX_N = 4           # sites racing inside ONE cultivation; ts-prod_d15.py:153
MEASURED_HORIZON = 8000   # rounds; the d15 tail is far below 1e-12 by here


def measured_checks(cube_distance=CUBE_DISTANCE):
    """single_site_cultivation_checks() -- per-round conditional pass probability, plus the round
    the run-and-retry chain finishes on."""
    cfg = MEASURED[cube_distance]
    checks, survival_before = {}, 1.0
    for round_i, _n_det, _alone, first_to_fail in cfg['stats']:
        checks[round_i] = 1.0 - first_to_fail / survival_before
        survival_before -= first_to_fail
    gap_round = max(checks) + cfg['gap_cycles']
    checks[gap_round] = 1.0 - cfg['gap']['below_threshold_discards'] / cfg['gap']['reached_gap_decision']
    return checks, gap_round


def measured_survival(cube_distance=CUBE_DISTANCE, horizon=MEASURED_HORIZON):
    """[t] = P(one site's chain has not produced a state within t ROUNDS). _survival_array()'s
    construction: advance one round on a passed check, restart at round 0 on a failed one."""
    checks, length = measured_checks(cube_distance)
    transition = np.zeros((length, length))
    for r in range(length):
        p = checks.get(r + 1, 1.0)
        transition[r][0] += 1.0 - p
        if r + 1 < length:
            transition[r][r + 1] += p
    state = np.zeros(length); state[0] = 1.0
    row = np.empty(horizon)
    for t in range(horizon):
        row[t] = state.sum()
        state = state @ transition
    return row


def _restart_survival(checks, length, horizon=MEASURED_HORIZON):
    """[t] = P(a restart-on-fail chain over `checks` has not reached `length` within t rounds)."""
    transition = np.zeros((length, length))
    for r in range(length):
        p = checks.get(r + 1, 1.0)
        transition[r][0] += 1.0 - p
        if r + 1 < length:
            transition[r][r + 1] += p
    state = np.zeros(length); state[0] = 1.0
    row = np.empty(horizon)
    for t in range(horizon):
        row[t] = state.sum()
        state = state @ transition
    return row


def cultivation_survival(cube_distance=CUBE_DISTANCE, n_ic=MULTIPLEX_N, horizon=MEASURED_HORIZON):
    """[t] = P(ONE complete cultivation has not produced a magic state within t ROUNDS).

    One cultivation is: `n_ic` sites racing through IC_ROUNDS (each restarting itself on its own
    failed check, first to reach IC_END wins), then the winner alone running the escape-ablation
    rounds and the gap decision. A failure anywhere downstream discards the state, so the WHOLE
    cultivation restarts -- a fresh race included; there is no state left to re-escape with. That
    whole-restart is the difference from multiplexed_expected_rounds(), which charges a fresh race
    per attempt but also lets the downstream chain restart on its own.

    n_ic = 1 degenerates exactly to the plain single-site chain, which is the test below."""
    checks, gap_round = measured_checks(cube_distance)
    ic_rounds = IC_ROUNDS[cube_distance]
    ic_end = max(ic_rounds)

    s_ic = _restart_survival({r: checks[r] for r in ic_rounds}, ic_end, horizon) ** n_ic
    a = np.zeros(horizon)                       # pmf of the race's duration
    a[1:] = s_ic[:-1] - s_ic[1:]

    down = {r - ic_end: checks[r] for r in checks if r not in ic_rounds}
    length_b = gap_round - ic_end
    b_succ = np.zeros(horizon); b_fail = np.zeros(horizon)
    alive = 1.0
    for r in range(1, length_b + 1):
        p = down.get(r, 1.0)
        b_fail[r] = alive * (1.0 - p)
        alive *= p
    b_succ[length_b] = alive                    # survived every downstream check

    c_succ = np.convolve(a, b_succ)[:horizon]   # one cycle that ends in a state
    c_fail = np.convolve(a, b_fail)[:horizon]   # one cycle that ends in a discard

    f = np.zeros(horizon)                       # pmf of the total time, by renewal recursion
    for t in range(horizon):
        f[t] = c_succ[t] + float(np.dot(c_fail[1:t + 1], f[t - 1::-1])) if t else c_succ[0]
    return 1.0 - np.cumsum(f)


_MEASURED_CACHE = {}


def _cached_survival(cube_distance, n_ic=MULTIPLEX_N):
    key = (cube_distance, n_ic)
    if key not in _MEASURED_CACHE:
        _MEASURED_CACHE[key] = cultivation_survival(cube_distance, n_ic)
    return _MEASURED_CACHE[key]


def cdf_measured(t, cube_distance=CUBE_DISTANCE, n=1, n_ic=MULTIPLEX_N):
    """P[at least one of `n` COMPLETE cultivations has a state by t cubes] = 1 - S(t)^n, where each
    cultivation is itself n_ic-way multiplexed through its IC stage."""
    s = _cached_survival(cube_distance, n_ic)
    idx = np.clip(np.floor(np.asarray(t, dtype=float) * cube_distance + _EPS).astype(int),
                  0, len(s) - 1)
    return np.where(np.asarray(t, dtype=float) < 0.0, 0.0, 1.0 - s[idx] ** n)


def mean_measured(cube_distance=CUBE_DISTANCE, n=1, n_ic=MULTIPLEX_N):
    """E[min of n complete cultivations], in cubes: sum_t S(t)^n rounds / CUBE_DISTANCE."""
    return float((_cached_survival(cube_distance, n_ic) ** n).sum()) / cube_distance


# --------------------------------------------------------------------------- CDFs

def cdf_exact(t, A):
    """P[W <= t] for the idealized continuous exponential with mean A."""
    t = np.asarray(t, dtype=float)
    return np.where(t <= 0.0, 0.0, 1.0 - np.exp(-np.maximum(t, 0.0) / A))


def cdf_ceil(t, A, delta):
    """P[delta*ceil(W/delta) <= t]: the exponential rounded UP to a grid of width delta.

    Support is {delta, 2*delta, ...}: nothing can complete before one whole interval
    has elapsed, which is why the per-layer sampler has an accidental one-cube floor."""
    t = np.asarray(t, dtype=float)
    n = np.floor(t / delta + _EPS)
    return np.where(t <= 0.0, 0.0, 1.0 - np.exp(-np.maximum(n, 0.0) * delta / A))


def cdf_round(t, A, delta):
    """P[delta*round(W/delta) <= t]: the exponential rounded to the NEAREST grid point.

    Support is {0, delta, 2*delta, ...} -- note the atom at 0, of mass
    1 - exp(-delta/(2A)), which is a state that is ready the instant it is asked for."""
    t = np.asarray(t, dtype=float)
    k = np.floor(t / delta + _EPS)
    return np.where(t < 0.0, 0.0, 1.0 - np.exp(-(np.maximum(k, 0.0) + 0.5) * delta / A))


def cdf_ts_tick(t, A, cube_distance=CUBE_DISTANCE):
    """ts-prod_d15.py:2654 -- one draw per square per ECC cycle."""
    return cdf_ceil(t, A, 1.0 / cube_distance)


def cdf_ts_layer(t, A):
    """ts-prod_d15.py:2357 -- one draw per square per cube."""
    return cdf_ceil(t, A, 1.0)


def cdf_puremagic(t, A):
    """PureMagic/src/cultivation.rs:51 -- one rounded draw per cultivation, then a countdown."""
    return cdf_round(t, A, 1.0)


# ------------------------------------------------------------------- moments / calibration

def mean_ceil(A, delta):
    """E[delta*ceil(W/delta)] = delta / (1 - exp(-delta/A)) = A + delta/2 + delta^2/(12A) + ..."""
    return delta / (1.0 - math.exp(-delta / A))


def mean_round(A, delta):
    """E[delta*round(W/delta)] = delta / (2 sinh(delta/2A)) = A - delta^2/(24A) + ..."""
    return delta / (2.0 * math.sinh(delta / (2.0 * A)))


def sd_ceil(A, delta):
    """delta*Geometric(p) on {1,2,...}: sd = delta*sqrt(1-p)/p."""
    p = 1.0 - math.exp(-delta / A)
    return delta * math.sqrt(1.0 - p) / p


def sd_round(A, delta, terms=8000):
    """No tidy closed form; summed over the pmf, which decays geometrically."""
    pmf = [1.0 - math.exp(-delta / (2.0 * A))]
    pmf += [math.exp(-(k - 0.5) * delta / A) - math.exp(-(k + 0.5) * delta / A)
            for k in range(1, terms)]
    mean = sum(k * delta * p for k, p in enumerate(pmf))
    return math.sqrt(sum((k * delta - mean) ** 2 * p for k, p in enumerate(pmf)))


def cdf_mixture(t, A, w, cube_distance=CUBE_DISTANCE):
    return ((1.0 - w) * cdf_ts_layer(t, A)
            + w * cdf_ts_tick(t, A, cube_distance))


def mean_mixture(A, w, cube_distance=CUBE_DISTANCE):
    """E[realized wait] = (1-w)*E_layer + w*E_tick = A + (1-w)*(1/2 + 1/12A) + w*(1/2D + ...)."""
    return (1.0 - w) * mean_ceil(A, 1.0) + w * mean_ceil(A, 1.0 / cube_distance)


def avg_for_mixture(target_mean, w, cube_distance=CUBE_DISTANCE):
    """The A making the REALIZED (w-mixture) mean equal `target_mean` cubes. No closed form;
    bisected on a monotone function. `avg_for_mixture(T, 1.0)` and `avg_for_target(T, 1/D)`
    agree, as do `avg_for_mixture(T, 0.0)` and `avg_for_target(T, 1.0)`."""
    lo, hi = 1e-6, max(4.0 * target_mean, 1.0)
    while mean_mixture(hi, w, cube_distance) < target_mean:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_mixture(mid, w, cube_distance) < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def avg_for_mixture_series(target_mean, w, cube_distance=CUBE_DISTANCE):
    """Second-order closed form for the same thing: with
       c1 = (1-w)/2 + w/(2D),  c2 = (1-w)/12 + w/(12 D^2),
       E = A + c1 + c2/A, so A = ((T-c1) + sqrt((T-c1)^2 - 4 c2)) / 2."""
    d = 1.0 / cube_distance
    c1 = (1.0 - w) * 0.5 + w * d * 0.5
    c2 = (1.0 - w) / 12.0 + w * d * d / 12.0
    r = target_mean - c1
    return 0.5 * (r + math.sqrt(r * r - 4.0 * c2))


def w_from_run(delay, height, frac_ticks=0.0, cube_distance=CUBE_DISTANCE):
    """The mixing weight of one run, from its own reported numbers: `delay` in cubes,
    `height` the final graph height, `frac_ticks` = delivered_fractional_ticks."""
    return cube_distance * delay / (cube_distance * height + frac_ticks)


def avg_for_target(target_mean, delta=1.0):
    """The A that makes the delta-grid CEILING sampler have mean `target_mean` cubes.

    A = -delta / ln(1 - delta/target).  Equivalently it sets the per-interval hazard to
    exactly delta/target: at delta=1, target=2 this is a fair coin per cube (A = 1/ln 2)."""
    return -delta / math.log(1.0 - delta / target_mean)


# ------------------------------------------------------------------------------ plotting

@dataclass(frozen=True)
class Curve:
    key: str
    label: str
    cdf: object          # f(t, A) -> P[T <= t]
    A: float
    grid: float          # jump spacing in cubes; 0.0 = the continuous ideal
    colour: str
    style: str = '-'
    lw: float = 1.8
    alpha: float = 1.0
    has_atom_at_zero: bool = False

    mixture_w: float = None   # set on the realized-FlowRouter curves; None for the pure grids
    or_k: int = 1             # complete cultivations run in parallel; the curve is the min of k

    def mean(self):
        if self.key.startswith('measured_d'):
            return mean_measured(int(self.key.split('_')[1][1:]), self.or_k)
        if self.grid == 0.0:
            return self.A
        if self.mixture_w is not None:
            return mean_mixture(self.A, self.mixture_w, int(round(1.0 / self.grid)))
        return mean_round(self.A, self.grid) if self.has_atom_at_zero else mean_ceil(self.A, self.grid)

    def sd(self):
        if self.grid == 0.0:
            return self.A
        if self.mixture_w is not None:
            return float('nan')   # a mixture of two geometrics; not reported here
        return sd_round(self.A, self.grid) if self.has_atom_at_zero else sd_ceil(self.A, self.grid)


def build_curves(a_topols, a_puremagic, cube_distance=CUBE_DISTANCE,
                 w=1, ideal_a=None, extra_as=(), or_ks=(1, 2, 3)):
    """The curves for one calibration.

    `a_topols`      AVG_CULTIVATION_TIME after recalibrating so the per-layer sampler's mean
                    hits the target (solid lines).
    `a_puremagic`   1/magic_state_lambda."""
    mix = lambda t, A: cdf_mixture(t, A, w, cube_distance)
    curves = []
    # ORing k parallel cultivations. For every exponential-based model this is exactly A -> A/k:
    # ceil() and round() are monotone, so the min of k discretized draws is the discretization of
    # the min, and min_i Exp(1/A) ~ Exp(k/A). (It is NOT 1-(1-F)^k, which would be the answer if
    # each copy had its own independent grid; they share the run's clock.) Only the measured chain
    # has no such rescaling -- its floor does not move -- which is the whole comparison.
    widths = {1: (1.8, 1.0), 2: (1.3, 0.8), 3: (1.0, 0.6), 4: (0.8, 0.5)}
    shades = ['#7f2704', '#d94801', '#fd8d3c', '#fdbe85']
    for k in or_ks:
        lw, al = widths.get(k, (0.7, 0.45))
        tag = '' if k == 1 else f' x{k} (OR)'
        curves += [
            Curve('exact' if k == 1 else f'exact_k{k}', f'Exp ideal{tag}', cdf_exact,
                  (a_topols if ideal_a is None else ideal_a) / k, 0.0, '#777777', ':', lw, al),
            Curve(f'ts_mix_k{k}', f'FlowRouter w={w:.2f}{tag}', mix, a_topols / k,
                  1.0 / cube_distance, '#1f77b4', '-', lw, al, False, w, k),
            Curve(f'puremagic_k{k}', f'PureMagic{tag}', cdf_puremagic,
                  a_puremagic / k, 1.0, '#d62728', '-', lw, al, True, None, k),
            Curve(f'measured_d{cube_distance}_k{k}',
                  f'MEASURED d{cube_distance} (IC n={MULTIPLEX_N}){tag}',
                  (lambda t, _A, _k=k, _d=cube_distance: cdf_measured(t, _d, _k)),
                  float('nan'), 1.0 / cube_distance, shades[(k - 1) % len(shades)], '-',
                  lw + 0.2, min(1.0, al + 0.15), False, None, k),
        ]
    # Diagnostics, at k = 1 only: what the file runs today, and the w = 0 constant.
    for i, (a_extra, note) in enumerate(extra_as):
        curves.append(Curve(f'ts_mix_extra{i}', f'FlowRouter w={w:.2f}, {note}', mix,
                            a_extra, 1.0 / cube_distance, '#9467bd', '-.', 1.4, 0.9, False, w))
    return curves


def _draw(ax, curve, t_max):
    if curve.grid == 0.0:
        x = np.linspace(0.0, t_max, 2000)
        ax.plot(x, curve.cdf(x, curve.A), color=curve.colour, ls=curve.style,
                lw=curve.lw, alpha=curve.alpha, zorder=2,
                label=f'{curve.label}  A={curve.A:.3f}, mu={curve.mean():.3f}')
        return
    x = np.arange(0.0, t_max + curve.grid, curve.grid)
    label = (f'{curve.label}  mu={curve.mean():.3f}' if math.isnan(curve.A)
             else f'{curve.label}  A={curve.A:.3f}, mu={curve.mean():.3f}')
    if curve.has_atom_at_zero:   # said in the legend rather than annotated onto a crowded plot
        label += f', atom {float(curve.cdf(0.0, curve.A)):.3f}'
    ax.step(x, curve.cdf(x + _EPS, curve.A), where='post', color=curve.colour,
            ls=curve.style, lw=curve.lw, alpha=curve.alpha, zorder=3, label=label)
    if curve.has_atom_at_zero:
        ax.plot([0.0], [float(curve.cdf(0.0, curve.A))], 'o', color=curve.colour, ms=6, zorder=4)

def figure_for(a_topols, a_puremagic, cube_distance=CUBE_DISTANCE,
               w=0.5, t_max=None, ideal_a=None, extra_as=(), or_ks=(1, 2, 3)):
    import matplotlib.pyplot as plt

    curves = build_curves(a_topols, a_puremagic, cube_distance, w, ideal_a,
                          extra_as, or_ks = (1,))
    if t_max is None:
        t_max = max(4.0 * a_topols, 6.0)

    fig, ax = plt.subplots(figsize=(12, 9))
    for curve in curves:
        _draw(ax, curve, t_max)
    floor = measured_checks(cube_distance)[1] / cube_distance   # CULTIVATION_LENGTH, in cubes
    ax.axvline(floor, color='#bbbbbb', lw=1.1, ls=':', zorder=0)
    ax.text(floor, 0.015, f'  d{cube_distance} physical floor ({floor:.2f} cubes)',
            fontsize=8, color='#888888', va='bottom')
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel('t [layers]')
    ax.set_ylabel('Pr[magic state ready by t]')
    ax.set_title(f'Cultivation-time CDFs: code distance = {cube_distance}, ', fontsize=10)
    # 14 entries: below the axes rather than over the curves it describes.
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.075), ncol=2, framealpha=0.95, borderaxespad=0.0)
    return fig


def table_for(a_topols, a_puremagic, cube_distance=CUBE_DISTANCE, w=0.5):
    procs = build_curves(a_topols, a_puremagic, cube_distance, w, or_ks=(1,))
    ts = [0, 1 / cube_distance, 0.5, 1, PHYSICAL_FLOOR, 2, 3, 4, 5,
          a_topols, 2 * a_topols, 3 * a_topols, 4 * a_topols]
    short = {'exact': 'exact', 'ts_mix_k1': 'FlowRouter', 'puremagic_k1': 'PureMagic'}
    procs = [p for p in procs if p.key in short]
    head = f"{'t (cubes)':>10} | " + ' '.join(f'{short[p.key]:>10}' for p in procs)
    lines = [head, '-' * len(head)]
    for t in sorted(set(round(x, 6) for x in ts)):
        lines.append(f'{t:10.4f} | ' + ' '.join(f'{float(p.cdf(t, p.A)):10.4f}' for p in procs))
    lines.append('-' * len(head))
    lines.append(f"{'mean':>10} | " + ' '.join(f'{p.mean():10.4f}' for p in procs))
    lines.append(f"{'sd':>10} | " + ' '.join(f'{p.sd():10.4f}' for p in procs))
    return '\n'.join(lines)


def main():
    """
    The code simulates the cultivation process by beginning a random draw with p = 1 - e^(-1/A) with A the rate parameter, and only enabling fractional delay when the program stalls for lack of T-states. w=0.5 attempts to model this due to PureMagic rounding its draws to integer times. This was enabled to affect a comparison with PureMagic that doesn't overestimate its T-state availability, whilst allowing for the fractional delay's (realistic) increased speed of T-state access.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', type=float, nargs='*', default=None,
                        help='effective means (in cubes) to calibrate BOTH models to; '
                             'FlowRouter gets A = -1/ln(1-1/T), PureMagic gets A = T')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--no-table', action='store_true')
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    w = 0.5

    # Each target is the mean cultivation time in cubes measured from the Stim cultivation chain (MEASURED below): ~2.1 at
    # d11, ~2.0 at d13, ~5.5 at d15. ts-prod's RESCALED_CULTIVATION_PARAMETER is the rate parameter rescaled so that its
    # finite exponential draws realize that mean; each target is plotted against the code distance it was measured at.
    default_cd = {2.1: 11, 2.0: 13, 5.5: 15}
    cases = []
    for target in (args.target if args.target is not None else [2.1, 2.0, 5.5]):
        cd = default_cd.get(target, CUBE_DISTANCE)
        cases.append((f'matched-{target:g}_d{cd}', avg_for_mixture(target, w, cd), target, cd))

    for name, a_ts, a_pm, cd in cases:
        fig = figure_for(a_ts, a_pm, cd, w, ideal_a=a_pm)   # a_pm == target: the ideal exponential has mean = target
        out = f'cultivation_cdf_{name}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'wrote {out}   (FlowRouter A={a_ts:.6f}, PureMagic A={a_pm:.6f})')
        if not args.no_table:
            print(table_for(a_ts, a_pm, cd, w), '\n')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
