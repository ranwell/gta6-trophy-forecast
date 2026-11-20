# GTA VI trophy earn-rate forecaster (Python 3.13, standard library only).
#
# For each trophy: take the distinct Exophase refreshes after launch night (the house's launch-night page was a stale
# snapshot of ~600 early players, so reads before LAUNCH_SKIP_H hours are ignored once later ones exist), fit
# logit(rate) against ln(hours since launch) by weighted least squares over the last FIT_WINDOW_H hours, and
# extrapolate the damped slope to the target. The spread on the logit scale combines a random walk, the
# extrapolation's own size (the further we move, the less sure we are) and the fit's residual.

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
DAMP = 0.75  # growth in ln(time) slows as the launch crowd settles
WALK = 0.3  # logit units per sqrt(day)
MOVE_UNC = 0.45  # share of the predicted move that is uncertain
SLOPE_MIN, SLOPE_MAX = -2.0, 4.0


def logit(pct):
    p = min(99.99, max(0.01, pct)) / 100  # the scorer's clamp
    return math.log(p / (1 - p))


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


def forecast(c):
    reads = refreshes(c['history'])
    t_last = hours(reads[-1]['at'])
    window = [r for r in reads if hours(r['at']) >= t_last - FIT_WINDOW_H]
    h_target = max(hours(c['target']), t_last + 1)
    days = c['horizon_minutes'] / 1440
    dx = math.log(h_target) - math.log(max(t_last, 1))

    rows = []
    for i, t in enumerate(c['trophies']):
        # A read taken before a trophy was added may carry a shorter rates list: treat it as missing.
        rate = lambda r: r['rates'][i] if i < len(r['rates']) else None
        pts = [(math.log(max(hours(r['at']), 1)), logit(rate(r))) for r in window if rate(r) is not None]
        last_rate = next((rate(r) for r in reversed(c['history']) if rate(r) is not None), 0.5)
        slope = resid = 0.0
        if len(pts) >= 3:
            # Weighted least squares: weight decays with hours before the last read, so a burst of
            # refreshes in one hour counts no more than its time span (Exophase refreshes irregularly).
            w = [math.exp((math.exp(x) - t_last) / FIT_WINDOW_H) for x, _ in pts]
            sw = sum(w)
            mx = sum(wk * x for wk, (x, _) in zip(w, pts)) / sw
            my = sum(wk * y for wk, (_, y) in zip(w, pts)) / sw
            sxx = sum(wk * (x - mx) ** 2 for wk, (x, _) in zip(w, pts))
            sxy = sum(wk * (x - mx) * (y - my) for wk, (x, y) in zip(w, pts))
            if sxx > 1e-9:
                slope = min(SLOPE_MAX, max(SLOPE_MIN, sxy / sxx))
            resid = math.sqrt(sum(wk * (y - my - slope * (x - mx)) ** 2 for wk, (x, y) in zip(w, pts)) / sw)
        move = DAMP * slope * dx
        mid = logit(last_rate) + move
        sd = math.sqrt(WALK ** 2 * days + (MOVE_UNC * move) ** 2 + resid ** 2)
        rows.append({'id': t['id'], 'q': [round(100 / (1 + math.exp(-(mid + z * sd))), 4) for z in Z]})
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
