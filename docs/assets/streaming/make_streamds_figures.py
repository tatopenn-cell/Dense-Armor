"""The figures of the page of streamDS: the score along a stream, which is what a number gives that a flag does not.

Run from the repo root:
    python docs/assets/streaming/make_streamds_figures.py
"""

import numpy as np
from make_streaming_filters_figures import GREY, INK, ORANGE, PINK, canvas, save, surface

from dense_armor.core.online.anomaly.deviation import streamDS


def stream():
    rng = np.random.default_rng(0)
    cost = 12.0 + 1.5 * rng.standard_normal(500)
    cost[[120, 250, 400]] = [3.0, 25.0, 19.0]
    return cost


def scores_of(cost, radius):
    sc = streamDS(radius=radius, ref_mult=3)
    out = []
    for x in cost:
        out.append(sc.score_one({"x": float(x)}))
        sc.learn_one({"x": float(x)})
    return np.array(out)


def main():
    cost = stream()
    days = np.arange(240, 261)
    radii = np.array([5, 10, 15, 20, 25, 30, 40, 50])
    Z = np.array([scores_of(cost, int(r))[days] for r in radii])
    surface(Z, days, radii, "streaming_deviation_scorer", "streamDS: the score along the stream", "the score of each day from day 240 to day 260, for 8 sizes of the window: the spike of day 250 stands out in every one",
            "day", "radius of the window", "score_one", (0, float(Z.max())))
    sc = scores_of(cost, 30)
    top = np.argsort(sc)[::-1][:4]
    fig, ax = canvas("streamDS: the score of every day", "500 days of the cost of executing an order; the dashed lines are the thresholds 3, 4 and 5; the pink points are the four highest scores", "day", "score_one")
    ax.plot(sc, color=ORANGE, lw=1.0)
    for n in (3, 4, 5):
        ax.axhline(n, color=GREY, ls="--", lw=0.9)
    ax.scatter(top, sc[top], color=PINK, s=50, zorder=5)
    save(fig, "streaming_deviation_scorer")
    print("top", top, np.round(sc[top], 1), [int(np.sum(sc > n)) for n in (3, 4, 5)])


if __name__ == "__main__":
    main()
