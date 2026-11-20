# GTA VI trophy earn-rate forecaster (Python 3.13, standard library only).
#
# A trophy's rate is earners / players tracked. The two move for different reasons: earners only ever grow, while
# the players Exophase tracks grow for every trophy at once, and each new player dilutes every rate. So:
#   - players tracked P: one fit of ln P against ln(hours since launch) over the recent distinct refreshes, shared
#     by every trophy, extrapolated with damping;
#   - earners E_i = rate_i * P: per trophy, a fit of ln E_i on the same clock, its slope held at >= 0 (earners never
#     decrease) and, where a trophy has few earners, shrunk toward the median slope of the well-counted trophies;
#   - the forecast rate is E_i / P at the target, and its spread on the logit scale combines a random walk, the
#     size of the predicted move, the fit's residual and the counting noise of few earners (about 1 / sqrt(E)).
# A trophy almost nobody holds yet has no trend to fit: its next days are a matter of when players reach it. For
# those, the case itself says how fast rare trophies stop being rare: the trophies that were rare ANALOG_BACK_H
# hours before the last read, and how far their logit rate has moved since. Those moves, rescaled from that span of
# ln(time) to the forecast's, give the rare trophy's quantiles directly.
# Launch night's page (a stale snapshot of ~600 early players) is ignored once later refreshes exist.

import json
import math
import os
from datetime import datetime

INPUT_DIR = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
OUTPUT_DIR = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
LAUNCH = datetime.fromisoformat('2026-11-19T00:00:00+00:00')

LAUNCH_SKIP_H = 6  # reads before this are launch night's stale page
FIT_WINDOW_H = 30  # fit on the last this-many hours of refreshes
DAMP = 0.4  # growth in ln(time) slows as the launch crowd settles (tuned on simulated launches)
POOL_E = 30  # earners at which a trophy's own slope and the pooled slope weigh the same
WALK = 0.3  # logit units per sqrt(day)
MOVE_UNC = 0.45  # share of the predicted move that is uncertain
SD_MIN = 0.15  # logit spread floor
SLOPE_MAX = 6.0
RATE_CAP = 99.5  # a rate not already at the top stays below this
RARE = 0.05  # percent at or below which a trophy has no trend worth fitting
ANALOG_BACK_H = 24  # how far back to look for trophies that were rare then
ANALOG_MIN = 4  # analog trophies needed to use them
ANALOG_SCALE = 0.75  # damping of the rescaled analog moves (0.7-0.8 equally good on simulated launches)


def logit(pct):
    p = min(99.99, max(0.01, pct)) / 100  # the scorer's clamp
    return math.log(p / (1 - p))


def inv(l):
    return 100 / (1 + math.exp(-l))


def hours(iso):
    return (datetime.fromisoformat(iso.replace('Z', '+00:00')) - LAUNCH).total_seconds() / 3600


def refreshes(history):
    """Reads that show a new Exophase refresh, launch night dropped when later ones exist."""
    distinct = []
    for r in history:
        if distinct and distinct[-1]['players_tracked'] == r['players_tracked'] and distinct[-1]['rates'] == r['rates']:
            continue
        distinct.append(r)
    later = [r for r in distinct if hours(r['at']) >= LAUNCH_SKIP_H]
    return later if len(later) >= 2 else distinct[-1:]


def fit(pts, t_last):
    """Weighted least squares of y on x, weight decaying with the hours before the last read. (slope, residual sd)."""
    if len(pts) < 3:
        return 0.0, 0.0
    w = [math.exp((math.exp(x) - t_last) / FIT_WINDOW_H) for x, _ in pts]
    sw = sum(w)
    mx = sum(wk * x for wk, (x, _) in zip(w, pts)) / sw
    my = sum(wk * y for wk, (_, y) in zip(w, pts)) / sw
    sxx = sum(wk * (x - mx) ** 2 for wk, (x, _) in zip(w, pts))
    if sxx <= 1e-9:
        return 0.0, 0.0
    slope = sum(wk * (x - mx) * (y - my) for wk, (x, y) in zip(w, pts)) / sxx
    resid = math.sqrt(sum(wk * (y - my - slope * (x - mx)) ** 2 for wk, (x, y) in zip(w, pts)) / sw)
    return slope, resid


def quantile(xs, tau):
    """Linear-interpolated quantile of a sorted list."""
    k = tau * (len(xs) - 1)
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def forecast(c):
    reads = refreshes(c['history'])
    t_last = hours(reads[-1]['at'])
    window = [r for r in reads if hours(r['at']) >= t_last - FIT_WINDOW_H]
    h_target = max(hours(c['target']), t_last + 1)
    days = c['horizon_minutes'] / 1440
    dx = math.log(h_target) - math.log(max(t_last, 1))
    xs = [math.log(max(hours(r['at']), 1)) for r in window]

    n = len(c['trophies'])
    rate = lambda r, i: r['rates'][i] if i < len(r['rates']) else None  # older reads may lack a new trophy
    last = [next((rate(r, i) for r in reversed(c['history']) if rate(r, i) is not None), 0.5) for i in range(n)]

    # Players tracked: one shared trend.
    p_last = reads[-1]['players_tracked']
    slope_p, _ = fit([(x, math.log(r['players_tracked'])) for x, r in zip(xs, window)], t_last)
    p_target = p_last * math.exp(DAMP * max(0.0, slope_p) * dx)

    # Earners: each trophy's own trend.
    e_last = [lr / 100 * p_last for lr in last]
    own = []
    for i in range(n):
        pts = [(x, math.log(rate(r, i) / 100 * r['players_tracked'] + 0.5)) for x, r in zip(xs, window) if rate(r, i) is not None]
        own.append(fit(pts, t_last))
    counted = sorted(own[i][0] for i in range(n) if e_last[i] >= POOL_E and last[i] < 50)
    pooled = counted[len(counted) // 2] if counted else 0.0

    # Rare trophies: how far the trophies that were rare ANALOG_BACK_H hours ago have moved since, per unit ln(time).
    back = [r for r in reads if hours(r['at']) <= t_last - ANALOG_BACK_H]
    analog = []
    if back:
        then = back[-1]
        span = math.log(max(t_last, 1)) - math.log(max(hours(then['at']), 1))
        if span > 1e-6:
            analog = sorted((logit(last[i]) - logit(rate(then, i))) / span for i in range(n)
                            if rate(then, i) is not None and rate(then, i) <= RARE)

    rows = []
    for i, t in enumerate(c['trophies']):
        if last[i] <= RARE and len(analog) >= ANALOG_MIN:
            base = logit(last[i])
            q = [inv(base + ANALOG_SCALE * dx * quantile(analog, tau)) for tau in LEVELS]
            # Keep a floor of spread so a run of identical analog moves does not make a point forecast.
            mid = logit(q[2])
            q = [inv(min(logit(qk), mid - SD_MIN * abs(z)) if z < 0 else max(logit(qk), mid + SD_MIN * z)) for qk, z in zip(q, Z)]
            rows.append({'id': t['id'], 'q': [round(v, 4) for v in q]})
            continue
        slope_e, resid = own[i]
        k = e_last[i] / (e_last[i] + POOL_E)
        slope_e = min(SLOPE_MAX, max(0.0, k * slope_e + (1 - k) * pooled))
        e_target = e_last[i] * math.exp(DAMP * slope_e * dx)
        mid_rate = last[i] if last[i] >= RATE_CAP else min(RATE_CAP, 100 * e_target / p_target)
        mid = logit(mid_rate)
        move = mid - logit(last[i])
        sd = max(SD_MIN, math.sqrt(WALK ** 2 * days + (MOVE_UNC * move) ** 2 + resid ** 2 + 1 / (e_target + 1)))
        rows.append({'id': t['id'], 'q': [round(inv(mid + z * sd), 4) for z in Z]})
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name in sorted(f for f in os.listdir(INPUT_DIR) if f.endswith('.in')):
        with open(os.path.join(INPUT_DIR, name), encoding='utf-8') as f:
            c = json.load(f)
        if c['levels'] != LEVELS:
            raise ValueError(f'{name}: unexpected levels')
        if not c['history']:
            raise ValueError(f'{name} has no history')
        with open(os.path.join(OUTPUT_DIR, name[:-3] + '.out'), 'w', encoding='utf-8') as f:
            f.write(json.dumps({'trophies': forecast(c)}, separators=(',', ':')) + '\n')


main()
