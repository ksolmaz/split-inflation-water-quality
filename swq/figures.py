import argparse
import os
import re

import numpy as np
import pandas as pd

from swq import data as D

MM = 1 / 25.4
WIDTH = 190 * MM
SCHEME = {"random": ("#D55E00", "o", "Random CV"), "station": ("#009E73", "^", "Station-wise CV"), "group": ("#0072B2", "s", "Blocked CV")}
INK, REF, GRAY = "#222222", "#999999", "#C8C8C8"
LAND, LEAK, TRAIN = "#ECEAE4", "#D2462F", "#A9A9A9"
REGION = (6.35, 7.45, 49.08, 49.70)
MAPS = os.path.join(D.ROOT, "maps")
LABEL = {"gloria/Chla": "Chl-a", "gloria/TSS": "TSS", "gloria/aCDOM440": "aCDOM(440)", "gloria/Secchi_depth": "Secchi depth",
         "caravan/NO3N": "NO$_3$-N", "caravan/TN": "TN", "caravan/TRP": "TRP", "caravan/TP": "TP", "caravan/TDP": "TDP",
         "caravan/DOC": "DOC", "caravan/TOC": "TOC", "ssc/SSC": "SSC", "coast/DOC": "DOC"}
FAMILY = {"lgbm": "LightGBM", "ridge": "Ridge regression", "knn": "kNN", "rf": "Random forest", "mlp": "MLP"}


def pyplot(big=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, l, t, g, ti = (10, 10, 9, 9, 10) if big else (8, 8, 7, 7, 8)
    plt.rcParams.update({"font.family": "Arial", "font.size": f, "axes.labelsize": l, "xtick.labelsize": t, "ytick.labelsize": t,
                         "legend.fontsize": g, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0, "axes.spines.top": False,
                         "axes.spines.right": False, "ps.fonttype": 42, "pdf.fonttype": 42, "savefig.dpi": 600,
                         "legend.frameon": False, "axes.titlesize": ti, "axes.unicode_minus": False, "mathtext.fontset": "custom",
                         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
    return plt


def save(plt, fig, folder, name):
    for ext in ("pdf", "eps", "png"):
        fig.savefig(os.path.join(folder, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(folder, name + ".png"))


def label_width():
    import matplotlib.pyplot as plt
    return round(66 * plt.rcParams["ytick.labelsize"] / 7)


def left_edge(fig, extra=8):
    return (label_width() + extra) / 72 / fig.get_figwidth()


def panel_title(ax, letter, title="", letter_x=0, title_x=11):
    import matplotlib.pyplot as plt
    size = plt.rcParams["axes.titlesize"]
    kw = dict(xy=(0, 1), xycoords="axes fraction", textcoords="offset points", ha="left", va="bottom", color=INK)
    ax.annotate(letter, xytext=(letter_x, 7), fontsize=size + 1, fontweight="bold", **kw)
    if title:
        ax.annotate(title, xytext=(title_x, 7), fontsize=size, **kw)


def positions(tasks, gap=0.35):
    y, heads, cur, prev = [], {}, 0.0, None
    for t in tasks:
        d = D.DATASET[t.split("/")[0]]
        if d != prev:
            if prev is not None:
                cur += gap
            heads[d] = cur
            cur += 1
            prev = d
        y.append(cur)
        cur += 1
    return np.array(y), heads


def task_axis(ax, tasks, y, heads, labels=True):
    ax.set_ylim(max(y) + 0.6, -0.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#EBEBEB", lw=0.5)
    ax.set_axisbelow(True)
    if not labels:
        return
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[t] for t in tasks])
    ax.tick_params(axis="y", pad=label_width() - 10)
    for lab in ax.get_yticklabels():
        lab.set_horizontalalignment("left")
    for name, yh in heads.items():
        ax.annotate(name, xy=(0, yh), xycoords=ax.get_yaxis_transform(), xytext=(-label_width(), 0), textcoords="offset points",
                    ha="left", va="center", fontweight="bold", fontsize=ax.yaxis.get_ticklabels()[0].get_fontsize() + 0.5,
                    color=INK, annotation_clip=False)


def separators(fig, axes, heads, gap=0.35):
    from matplotlib.lines import Line2D
    x0 = axes[0].get_position().x0 - label_width() / 72 / fig.get_figwidth()
    x1 = axes[-1].get_position().x1
    for yh in list(heads.values())[1:]:
        yf = fig.transFigure.inverted().transform(axes[0].transData.transform((0, yh - 0.5 - gap / 2)))[1]
        fig.add_artist(Line2D([x0, x1], [yf, yf], color="#CFCFCF", lw=0.6))


def read(tables, name, tasks=None):
    df = pd.read_csv(os.path.join(tables, name))
    return df if tasks is None else df.set_index("task").loc[tasks]


def rings(name):
    d = pd.read_csv(os.path.join(MAPS, name))
    return [g[["lon", "lat"]].values for _, g in d.groupby("ring", sort=False)]


def map_frame(ax, box, asp=1.0, anchor="N"):
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect(asp, adjustable="box", anchor=anchor)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color("#9A9A9A")


def fig1(folder, plt):
    from matplotlib.collections import PolyCollection
    land, lakes = rings("land_world.csv.gz"), rings("lakes.csv.gz")
    extent = {"ssc": (-170, -62, 17, 73)}
    pw = (WIDTH * 0.98 - 4 * MM) / 2
    ph = pw * 145 / 360
    H = 2 * ph + 20 * MM
    fig = plt.figure(figsize=(WIDTH, H))
    gs = fig.add_gridspec(2, 2, hspace=8 * MM / ph, wspace=0.03, left=0.01, right=0.99, top=1 - 7.5 * MM / H, bottom=2 * MM / H)
    for i, (dom, let) in enumerate(zip(D.TASKS, "abcd")):
        best = None
        for t in D.TASKS[dom]:
            task = D.get_task(f"{dom}/{t}")
            P = pd.DataFrame({"lat": task.lat, "lon": task.lon}).drop_duplicates()
            if best is None or len(P) > len(best):
                best = P
        ax = fig.add_subplot(gs[i // 2, i % 2])
        box = extent.get(dom, (-180, 180, -60, 85))
        inside = lambda r: r[:, 0].max() >= box[0] and r[:, 0].min() <= box[1] and r[:, 1].max() >= box[2] and r[:, 1].min() <= box[3]
        ax.add_collection(PolyCollection([r for r in land if inside(r)], facecolors=LAND, edgecolors=LAND, linewidths=0.25, zorder=0))
        ax.add_collection(PolyCollection([r for r in lakes if inside(r)], facecolors="white", edgecolors="white", linewidths=0.2, zorder=1))
        ax.scatter(best.lon, best.lat, s=2.4, color="#2E4A62", linewidths=0, zorder=2)
        if dom == "caravan":
            ax.plot((REGION[0] + REGION[1]) / 2, (REGION[2] + REGION[3]) / 2, marker="s", ms=7, mfc="none", mec=LEAK, mew=1.3, zorder=3)
        map_frame(ax, box)
        panel_title(ax, let, D.DATASET[dom])
    save(plt, fig, folder, "Fig1_map")


def fig2(folder, plt, k=1):
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    task = D.get_task("caravan/DOC")
    si = pd.read_csv(os.path.join(D.DATA, "caravan", "wqms_site_info.csv")).drop_duplicates("wqms_id").set_index("wqms_id")
    link = si.loc[task.ids, "LINKNO"].values
    has = pd.notna(link)
    box = REGION
    inb = (task.lon >= box[0]) & (task.lon <= box[1]) & (task.lat >= box[2]) & (task.lat <= box[3])
    asp = 1 / np.cos(np.radians(np.mean(box[2:])))
    km = 111.32 * np.cos(np.radians(np.mean(box[2:])))
    pw = (WIDTH * 0.97 - 6 * MM) / 3
    hm = pw * (box[3] - box[2]) * asp / (box[1] - box[0])
    H = hm + 20 * MM
    fig = plt.figure(figsize=(WIDTH, H))
    gs = fig.add_gridspec(1, 3, wspace=0.04, left=0.015, right=0.985, top=1 - 8.5 * MM / H, bottom=11 * MM / H)
    polys = rings("land_saarland.csv.gz")
    for c, (scheme, let) in enumerate(zip(("random", "station", "blocked"), "abc")):
        f = D.folds(task, scheme, 0)
        test, train = (f == k) & inb, (f != k) & inb
        leak = has & np.isin(link, link[(f != k) & has]) & test
        ax = fig.add_subplot(gs[0, c])
        ax.add_collection(PolyCollection(polys, facecolors=LAND, edgecolors="white", linewidths=0.8, zorder=0))
        ax.scatter(task.lon[train], task.lat[train], s=9, color=TRAIN, linewidths=0, zorder=2)
        ax.scatter(task.lon[test & ~leak], task.lat[test & ~leak], s=32, facecolor="white", edgecolor="#1F1F1F", linewidths=1.0, zorder=3)
        ax.scatter(task.lon[leak], task.lat[leak], s=36, color=LEAK, edgecolor="white", linewidths=0.6, zorder=4)
        map_frame(ax, box, asp, anchor="C")
        panel_title(ax, let)
        for txt, x, y in (("FRANCE", 6.86, 49.115), ("GERMANY", 7.27, 49.665), ("LUX.", 6.42, 49.665)):
            ax.text(x, y, txt, fontsize=8, color="#7A7A7A", ha="center", va="center", zorder=1)
        if c == 0:
            x0, y0 = box[0] + 0.05, box[2] + 0.035
            ax.plot([x0, x0 + 20 / km], [y0, y0], color="#2A2A2A", lw=1.3, solid_capstyle="butt", zorder=6)
            ax.text(x0 + 10 / km, y0 + 0.017, "20 km", ha="center", fontsize=8.5, color="#2A2A2A", zorder=6)
    handles = [Line2D([], [], marker="o", ls="", color=TRAIN, ms=3.6, label="Training station"),
               Line2D([], [], marker="o", ls="", mfc="white", mec="#1F1F1F", mew=1.0, ms=6, label="Test station"),
               Line2D([], [], marker="o", ls="", mfc=LEAK, mec="white", mew=0.6, ms=6.5,
                      label="Test station with a training station on the same river reach")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.8)
    save(plt, fig, folder, "Fig2_region")


def fig3(tables, folder, plt):
    I = read(tables, "inflation.csv")
    tasks = [t for t in D.ALL_TASKS if t in set(I.task)]
    I, G = I.set_index("task").loc[tasks], read(tables, "station_grouping.csv", tasks)
    y, heads = positions(tasks)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 128 * MM), sharey=True,
                                 gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.06, "right": 0.98, "top": 0.88, "bottom": 0.10})
    fig.subplots_adjust(left=left_edge(fig))
    for yi, r, g in zip(y, I.r2_random, I.r2_group):
        ax.plot([g, r], [yi, yi], color=GRAY, lw=1.2, zorder=2)
    hs = {}
    for k, v, size, z in (("random", I.r2_random, 40, 3), ("group", I.r2_group, 40, 3), ("station", G.r2_station, 32, 4)):
        c, m, lab = SCHEME[k]
        hs[k] = ax.scatter(v, y, color=c, marker=m, s=size, label=lab, zorder=z, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0.35, 0.9)
    ax.set_xlabel("R²")
    task_axis(ax, tasks, y, heads)
    panel_title(ax, "a", "", -label_width())
    for off, k, v, lo, hi in ((-0.17, "random", I.inflation, I.inflation_lo, I.inflation_hi),
                              (0.17, "station", G.extra_r2_station, G.extra_r2_station_lo, G.extra_r2_station_hi)):
        c, m, _ = SCHEME[k]
        bx.errorbar(v, y + off, xerr=[v - lo, hi - v], fmt=m, color=c, ms=5.5, mec="white", mew=0.5, elinewidth=1.1, capsize=0, zorder=3)
    bx.axvline(0, color=REF, lw=0.7, ls="--")
    bx.set_xlim(-0.05, 0.45)
    bx.set_xlabel("R² minus blocked-CV R²")
    task_axis(bx, tasks, y, heads, labels=False)
    panel_title(bx, "b")
    fig.legend(handles=[hs[k] for k in ("random", "station", "group")], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995),
               handletextpad=0.3, columnspacing=2.0)
    separators(fig, (ax, bx), heads)
    save(plt, fig, folder, "Fig3_inflation")


def fig4(tables, folder, plt):
    tasks = ["gloria/Chla", "caravan/DOC", "ssc/SSC", "coast/DOC"]
    unit = {"gloria/Chla": "log10 Chl-a", "caravan/DOC": "ln DOC", "ssc/SSC": "log10 SSC", "coast/DOC": "log10 DOC"}
    I = read(tables, "inflation.csv").set_index("task")
    P = read(tables, "fig4_predictions.csv.gz")
    fig, axs = plt.subplots(2, 4, figsize=(WIDTH, 112 * MM),
                            gridspec_kw={"wspace": 0.38, "hspace": 0.36, "left": 0.10, "right": 0.99, "top": 0.92, "bottom": 0.11})
    for j, t in enumerate(tasks):
        d = P[P.task == t]
        lo, hi = np.percentile(d.y, [0.5, 99.5])
        pad = 0.05 * (hi - lo)
        lim = (lo - pad, hi + pad)
        for i, (scheme, cmap, col) in enumerate((("random", "Oranges", "r2_random"), ("blocked", "Blues", "r2_group"))):
            ax = axs[i, j]
            x = d[d.scheme == scheme]
            ax.hexbin(x.y, x.p, gridsize=32, extent=(*lim, *lim), bins="log", cmap=cmap, mincnt=1, linewidths=0)
            ax.plot(lim, lim, color=INK, lw=0.7, ls="--")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ticks = [v for v in ax.get_xticks() if lim[0] <= v <= lim[1]]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.annotate(f"R² = {I.loc[t, col]:.2f}", xy=(0.05, 0.88), xycoords="axes fraction", fontsize=9, color=INK)
            ax.annotate("abcdefgh"[i * 4 + j], xy=(0, 1), xycoords="axes fraction", xytext=(-3, 5), textcoords="offset points", ha="right",
                        fontsize=11, fontweight="bold", color=INK)
            if i == 0:
                ax.set_title(f"{D.DATASET[t.split('/')[0]]} {LABEL[t]}" if t != "ssc/SSC" else "USGS SSC", color=INK, pad=7)
            else:
                ax.set_xlabel(f"Observed ({unit[t]})")
            if j == 0:
                ax.set_ylabel(f"{SCHEME['random' if scheme == 'random' else 'group'][2]}\nPredicted", linespacing=1.4)
    save(plt, fig, folder, "Fig4_observed_predicted")


def fig5(tables, folder, plt, lo=-1.0):
    from matplotlib.lines import Line2D
    L = read(tables, "location_only.csv")
    tasks = [t for t in D.ALL_TASKS if t in set(L.task)]
    L = L.set_index("task").loc[tasks]
    dist = read(tables, "distance_error_by_dataset.csv")
    bins = ["<0.1", "0.1-1", "1-10", "10-100", "100-1000", ">1000"]
    y, heads = positions(tasks)
    fig = plt.figure(figsize=(WIDTH, 125 * MM))
    ax = fig.add_subplot(fig.add_gridspec(1, 1, left=left_edge(fig), right=0.49, top=0.885, bottom=0.10)[0, 0])
    r, g, lg = L.location_r2_random.values, L.location_r2_group.values, L.lgbm_r2_random.values
    for yi, ri, gi in zip(y, r, g):
        ax.plot([max(ri, lo), max(gi, lo)], [yi, yi], color=GRAY, lw=1.2, zorder=2)
    ax.scatter(lg, y, marker="|", s=110, color="#6E6E6E", linewidths=1.6, zorder=4)
    for k, v in (("random", r), ("group", g)):
        c, m, _ = SCHEME[k]
        ax.scatter(np.maximum(v, lo), y, color=c, marker=m, s=40, zorder=3, edgecolor="white", linewidth=0.5)
        for yi, vi in zip(y, v):
            if vi < lo:
                ax.annotate(f"{vi:.2f}", (lo, yi), xytext=(0, 6), textcoords="offset points", fontsize=8.5, color=INK, ha="left", va="bottom")
    ax.axvline(0, color=REF, lw=0.7, ls="--")
    ax.set_xlim(lo - 0.05, 1.0)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("R²")
    task_axis(ax, tasks, y, heads)
    panel_title(ax, "a", "", -label_width())
    separators(fig, (ax,), heads)
    gr = fig.add_gridspec(2, 2, left=0.615, right=0.99, top=0.885, bottom=0.15, wspace=0.34, hspace=0.50)
    x = np.arange(len(bins)) + 0.5
    for j, dom in enumerate(D.TASKS):
        bx = fig.add_subplot(gr[j // 2, j % 2])
        name = D.DATASET[dom]
        for off, split in ((-0.2, "random"), (0.2, "group")):
            d = dist[(dist.dataset == name) & (dist.split == split)].set_index("km_bin").reindex(bins)
            bx.bar(x + off, 100 * d.share_of_test_rows.fillna(0).values, width=0.38, color=SCHEME[split][0], lw=0)
        bx.axvline(2, color=REF, lw=0.7, ls="--")
        bx.set_xlim(0, 6)
        bx.set_xticks([1, 2, 3, 4, 5])
        bx.set_xticklabels(["0.1", "1", "10", "", "1000"])
        bx.set_ylim(0, 100)
        bx.set_yticks([0, 50, 100])
        bx.grid(axis="y", color="#EBEBEB", lw=0.5)
        bx.set_axisbelow(True)
        panel_title(bx, "bcde"[j], name)
    fig.text(0.80, 0.012, "Distance to the nearest training sample (km)", ha="center", va="bottom")
    fig.text(0.548, 0.52, "Test samples (%)", ha="center", va="center", rotation=90)
    handles = [Line2D([], [], marker=SCHEME["random"][1], ls="", color=SCHEME["random"][0], ms=6.5, mec="white", mew=0.4, label=SCHEME["random"][2]),
               Line2D([], [], marker=SCHEME["group"][1], ls="", color=SCHEME["group"][0], ms=6.5, mec="white", mew=0.4, label=SCHEME["group"][2]),
               Line2D([], [], marker="|", ls="", color="#6E6E6E", ms=9, mew=1.6, label="LightGBM, random CV")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995), handletextpad=0.4, columnspacing=2.0)
    save(plt, fig, folder, "Fig5_location_only")


def fig6(tables, folder, plt):
    from matplotlib.lines import Line2D
    other_c = "#BDBDBD"
    task = D.get_task("gloria/Chla")
    wl = np.arange(400, 701, 25)
    key = pd.Series(task.key)
    vc = key.value_counts()
    cand = vc[(vc >= 4) & (vc <= 8)].index

    def spread(k):
        ii = np.where(key.values == k)[0]
        return np.mean([np.linalg.norm(task.X[p] - task.X[q]) for x, p in enumerate(ii) for q in ii[x + 1:]])

    sd = {k: spread(k) for k in cand}
    k0 = sorted(sd, key=sd.get)[len(sd) // 2]
    m = np.where(key.values == k0)[0]
    rng = np.random.RandomState(3)
    others = rng.choice(np.where(task.site != task.site[m[0]])[0], 5, replace=False)
    g = key.values
    idx = np.arange(task.n)
    same = []
    for k in vc[vc >= 2].index:
        ii = idx[g == k]
        for p in range(len(ii)):
            for q in range(p + 1, len(ii)):
                same.append(np.linalg.norm(task.X[ii[p]] - task.X[ii[q]]))
    pa, pb = rng.randint(0, task.n, 20000), rng.randint(0, task.n, 20000)
    keep = g[pa] != g[pb]
    rand = np.linalg.norm(task.X[pa[keep]] - task.X[pb[keep]], axis=1)
    print(f"Fig. 6: {len(m)} samples, Chl-a {10 ** task.y[m].min():.1f}-{10 ** task.y[m].max():.1f}; "
          f"median distance {np.median(same):.2f} (same water body and day), {np.median(rand):.2f} (random pairs)")
    fig = plt.figure(figsize=(WIDTH, 118 * MM))
    gl = fig.add_gridspec(2, 1, left=0.09, right=0.46, top=0.875, bottom=0.10, hspace=0.48)
    ax, bx = fig.add_subplot(gl[0]), fig.add_subplot(gl[1])
    for i in others:
        ax.plot(wl, 10 ** task.X[i], color=other_c, lw=1.1)
    for i in m:
        ax.plot(wl, 10 ** task.X[i], color=LEAK, lw=1.2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Rrs (1/sr)")
    edges = np.linspace(0, np.percentile(rand, 99), 40)
    bx.hist(rand, bins=edges, density=True, color=other_c, lw=0)
    bx.hist(same, bins=edges, density=True, color=LEAK, lw=0)
    bx.hist(rand, bins=edges, density=True, histtype="step", color="#8C8C8C", lw=0.7)
    bx.set_xlabel("Distance between log10 Rrs spectra")
    bx.set_ylabel("Density")
    panel_title(ax, "a")
    panel_title(bx, "b")
    fig.legend(handles=[Line2D([], [], color=LEAK, lw=2.5, label="Same water body and day"), Line2D([], [], color=other_c, lw=2.5, label="Other samples")],
               loc="upper left", ncol=2, bbox_to_anchor=(0.07, 0.995), handlelength=1.4, columnspacing=1.6)
    T = read(tables, "twin_removal.csv")
    T = T[T.definition == "narrow"]
    tasks = [t for t in D.ALL_TASKS if t in set(T.task)]
    y, heads = positions(tasks)
    T = T.set_index("task").reindex(tasks)
    cx = fig.add_subplot(fig.add_gridspec(1, 1, left=0.555 + label_width() / 72 / fig.get_figwidth(), right=0.985, top=0.875, bottom=0.10)[0, 0])
    ok = T.size_matched.fillna(False).astype(bool).values
    for yi, v, s, matched in zip(y, T.share_of_inflation.values, T.share_of_inflation_sd.values, ok):
        if matched:
            cx.barh(yi, 100 * v, height=0.55, color=LEAK, xerr=100 * s, error_kw={"elinewidth": 0.8, "ecolor": INK}, zorder=2)
        else:
            cx.text(1, yi, "n.m.", va="center", fontsize=8.5, color=REF)
    cx.set_xlim(0, 70)
    cx.set_xlabel("Inflation removed (%)")
    task_axis(cx, tasks, y, heads)
    panel_title(cx, "c", "", -label_width())
    separators(fig, (cx,), heads)
    save(plt, fig, folder, "Fig6_near_duplicates")


def fig7(tables, folder, plt):
    from matplotlib.lines import Line2D
    V = read(tables, "intervals.csv")
    V = V[(V.level == 0.9) & (V["eval"] == "group")]
    tasks = [t for t in D.ALL_TASKS if t in set(V.task)]
    y, heads = positions(tasks)
    fig, cx = plt.subplots(figsize=(140 * MM, 112 * MM), gridspec_kw={"right": 0.97, "top": 0.915, "bottom": 0.10})
    fig.subplots_adjust(left=left_edge(fig))
    handles = []
    for off, cal, lab in ((-0.17, "random", "Calibrated with random CV"), (0.17, "group", "Calibrated with blocked CV")):
        c, m, _ = SCHEME[cal]
        v = V[V.cal == cal].set_index("task").reindex(tasks)
        cx.errorbar(v.coverage, y + off, xerr=[v.coverage - v.lo, v.hi - v.coverage], fmt=m, color=c, ms=5.5, mec="white", mew=0.5,
                    elinewidth=1.1, capsize=0, zorder=3)
        handles.append(Line2D([], [], color=c, marker=m, ms=6, mec="white", mew=0.5, lw=1.1, label=lab))
    cx.axvline(0.9, color=REF, lw=0.7, ls="--")
    cx.set_xlim(0.6, 1.0)
    cx.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    cx.set_xlabel("Coverage in held-out blocks")
    task_axis(cx, tasks, y, heads)
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995), handletextpad=0.4, columnspacing=1.6)
    separators(fig, (cx,), heads)
    save(plt, fig, folder, "Fig7_intervals")


def graphical_abstract(tables, folder, plt):
    from matplotlib.patches import FancyBboxPatch
    I = pd.read_csv(os.path.join(tables, "inflation.csv")).sort_values("r2_random", ascending=False)
    W, H = 130.0, 52.0
    O, B, G, LG = SCHEME["random"][0], SCHEME["group"][0], "#BDBDBD", "#F2F2F2"
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.text(4, 6.3, "Near-duplicates inflate random cross-validation", fontsize=10.5, fontweight="bold", color=INK, va="center")

    def sites(y0, held, title, glyph, col, caption):
        ax.plot(5.2, y0 - 2.6, marker=glyph, ms=6, color=col, mec="white", mew=0.4, ls="")
        ax.text(7.2, y0 - 2.6, title, fontsize=8.5, fontweight="bold", color=INK, va="center")
        pts = [(3, 2.7), (7.2, 2.4), (11.4, 3.0), (3.6, 6.6), (7.8, 6.9), (12.0, 6.4)]
        for j in range(4):
            x0 = 5 + j * 18.6
            ax.add_patch(FancyBboxPatch((x0, y0), 15, 9.4, boxstyle="round,pad=0,rounding_size=1.6", fc=LG,
                                        ec=B if held == j else "none", lw=1.0 if held == j else 0))
            for k, (dx, dy) in enumerate(pts):
                if held == j:
                    ax.plot(x0 + dx, y0 + dy, marker="s", ms=5.2, color=B, mec="white", mew=0.4, ls="")
                else:
                    ax.plot(x0 + dx, y0 + dy, marker="o", ms=5.6, color=G, mec="white", mew=0.4, ls="")
                if held is None and k == (j % 3) + 3:
                    ax.plot(x0 + dx + 1.55, y0 + dy - 0.35, marker="o", ms=5.6, color=O, mec="white", mew=0.4, ls="")
        ax.text(5, y0 + 12.4, caption, fontsize=7.5, color=INK, va="center")

    sites(14.6, None, "Random CV", "o", O, "Test samples have near-duplicates in training")
    sites(35.0, 2, "Blocked CV", "s", B, "Linked sites held out together, as for an unmonitored site")
    ax.plot([81.5, 81.5], [11.5, 49.5], color="#DDDDDD", lw=0.6)
    x0, x1, y0, y1 = 86.0, 126.0, 11.5, 39.5
    bx = fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H])
    n = len(I)
    yy = np.arange(n)
    for yi, r, g in zip(yy, I.r2_random, I.r2_group):
        bx.plot([g, r], [yi, yi], color=G, lw=0.9, zorder=1)
    bx.plot(I.r2_random, yy, "o", ms=4.2, color=O, mec="white", mew=0.3, zorder=3)
    bx.plot(I.r2_group, yy, "s", ms=3.8, color=B, mec="white", mew=0.3, zorder=3)
    bx.set_xlim(0.4, 0.9)
    bx.set_ylim(n - 0.4, -0.6)
    bx.set_yticks([])
    bx.set_xticks([0.4, 0.6, 0.8])
    bx.tick_params(axis="x", labelsize=7, length=2, pad=1.5)
    bx.spines[["left", "top", "right"]].set_visible(False)
    bx.text(0.9, n - 0.4, "R$^2$", fontsize=7.5, ha="right", va="bottom", color=INK)
    ax.text(86, 45.6, "Lower R$^2$ at new sites", fontsize=8.5, fontweight="bold", color=INK, va="center")
    ax.text(86, 48.9, "13 of 13 targets, 4 datasets", fontsize=7.5, color=INK, va="center")
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(os.path.join(folder, "graphical_abstract.svg"))
    fig.savefig(os.path.join(folder, "graphical_abstract.pdf"))
    for ext in ("tiff", "png"):
        fig.savefig(os.path.join(folder, f"graphical_abstract.{ext}"), dpi=2656 / (W * MM))
    plt.close(fig)
    print("wrote", os.path.join(folder, "graphical_abstract.png"))


TABLE1 = {
    "gloria": dict(type="In situ remote-sensing reflectance and water quality; inland and coastal waters, global",
                   targets="Chlorophyll-a (mg m^{-3}), total suspended solids (g m^{-3}), aCDOM(440) (m^{-1}), Secchi depth (m)",
                   predictors="Remote-sensing reflectance, 13 bands from 400 to 700 nm",
                   dup="Same water body and day", station="Site (water-body) name",
                   blocks="Connected components of 1° grid cell and site name",
                   source="Lehmann et al. (2022), doi:10.1594/PANGAEA.948492; Lehmann et al. (2023), Sci. Data 10, 100"),
    "caravan": dict(type="Catchment and river-reach attributes with station-median water quality; rivers, global",
                    targets="NO_{3}-N, TN, TRP, TP, TDP, DOC, TOC (mg l^{-1}), station medians",
                    predictors="65 catchment and river-reach attributes",
                    dup="Identical predictor vector", station="Gauge identifier",
                    blocks="Connected components of gauge, reach and merged-reach identifiers",
                    source="Jones et al. (2026a), Caravan-Qual (lite) v1.1, doi:10.5281/zenodo.21337532; Jones et al. (2026b), Sci. Data 13, 1015"),
    "ssc": dict(type="Sentinel-2 reflectance matched to USGS suspended-sediment samples; rivers, United States",
                targets="Suspended sediment concentration (mg l^{-1})",
                predictors="17 Sentinel-2 band and index medians; sine and cosine of day of year",
                dup="Same site and day", station="USGS site number",
                blocks="Connected components of 1° grid cell and site number",
                source="Nookala et al. (2026), doi:10.5281/zenodo.22298970"),
    "coast": dict(type="Dissolved organic carbon with physicochemical measurements; coastal waters, global",
                  targets="Dissolved organic carbon (µmol l^{-1})",
                  predictors="Temperature, salinity, depth, nitrate plus nitrite, ammonium, phosphate",
                  dup="Same location and date", station="Location name",
                  blocks="Connected components of 1° grid cell and location name",
                  source="Lønborg et al. (2023), doi:10.1594/PANGAEA.964012; Lønborg et al. (2024), Earth Syst. Sci. Data 16, 1107-1119"),
}
MARKUP = re.compile(r"([\^_])\{([^}]*)\}")


def plain(v):
    return MARKUP.sub(r"\2", v) if isinstance(v, str) else v


def span(v):
    lo, hi = int(min(v)), int(max(v))
    return f"{lo:,}" if lo == hi else f"{lo:,}-{hi:,}"


def write(df, folder, name):
    df.to_csv(os.path.join(folder, name), index=False)
    print("wrote", os.path.join(folder, name))


def table1(folder):
    rows = []
    for dom, targets in D.TASKS.items():
        tasks = [D.get_task(f"{dom}/{t}") for t in targets]
        d = TABLE1[dom]
        rows.append({"Dataset": D.DATASET[dom], "Data type and region": d["type"], "Targets (unit)": d["targets"], "Predictors": d["predictors"],
                     "Samples per target": span([t.n for t in tasks]),
                     "Stations": span([t.n if dom == "caravan" else len(np.unique(t.site)) for t in tasks]),
                     "Blocks": span([len(np.unique(t.grp)) for t in tasks]),
                     "Near-duplicates (removal analysis)": d["dup"], "Station (station-wise CV)": d["station"],
                     "Blocks (blocked CV)": d["blocks"], "Source": d["source"]})
    write(pd.DataFrame(rows).apply(lambda c: c.map(plain)), folder, "Table1_datasets.csv")


def table2(tables, folder):
    def mr(v, d=3):
        return f"{np.median(v):.{d}f} ({np.min(v):.{d}f}-{np.max(v):.{d}f})"
    S = read(tables, "models_supplement.csv")
    n = S.task.nunique()
    rows = [{"Model family": FAMILY[m], "R2, random CV": mr(s.r2_random, 2), "R2, blocked CV": mr(s.r2_group, 2), "Inflation": mr(s.inflation),
             "Targets with CI above zero": f"{int((s.inflation_lo > 0).sum())}/{n}"} for m in FAMILY for s in [S[S.model == m]]]
    write(pd.DataFrame(rows), folder, "Table2_model_families.csv")


def table3(tables, folder):
    N = read(tables, "near_duplicate_share.csv").groupby(["task", "scheme"]).share.mean().unstack()
    rows, prev = [], None
    for t in [t for t in D.ALL_TASKS if t in N.index]:
        dom = t.split("/")[0]
        first = dom != prev
        prev = dom
        rows.append({"Dataset": D.DATASET[dom] if first else "", "Target": LABEL[t].replace("NO$_3$-N", "NO3-N"),
                     "Near-duplicate definition": TABLE1[dom]["dup"] if first else "",
                     "Random CV (%)": f"{100 * N.loc[t, 'random']:.0f}", "Station-wise CV (%)": f"{100 * N.loc[t, 'station']:.0f}",
                     "Blocked CV (%)": f"{100 * N.loc[t, 'blocked']:.0f}"})
    write(pd.DataFrame(rows), folder, "Table3_near_duplicates.csv")


def table_s1(tables, folder):
    S = read(tables, "models_supplement.csv")
    tasks = [t for t in D.ALL_TASKS if t in set(S.task)]
    S = S.assign(o_t=S.task.map(tasks.index), o_m=S.model.map(list(FAMILY).index)).sort_values(["o_t", "o_m"])
    out = pd.DataFrame({"dataset": S.dataset.values, "task": [LABEL[t].replace("NO$_3$-N", "NO3-N") for t in S.task], "model": S.model.map(FAMILY).values,
                        "r2_random": S.r2_random.values, "r2_blocked": S.r2_group.values, "inflation": S.inflation.values,
                        "inflation_lo": S.inflation_lo.values, "inflation_hi": S.inflation_hi.values})
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(3)
    write(out, folder, "S1_model_families.csv")


def table_s2(tables, folder):
    I, L, G = (read(tables, f).set_index("task") for f in ("inflation.csv", "location_only.csv", "station_grouping.csv"))
    T = read(tables, "twin_removal.csv")
    T = T[T.definition == "narrow"].set_index("task")
    V = read(tables, "intervals.csv")
    V = V[(V.level == 0.9) & (V["eval"] == "group")]
    dist = read(tables, "distance_error_by_dataset.csv")
    dist = dist[dist.km_bin.isin(["<0.1", "0.1-1"])]
    near = {k: dist[dist.split == k].groupby("dataset").share_of_test_rows.sum() for k in ("random", "group")}
    tasks = [t for t in D.ALL_TASKS if t in I.index]
    S = pd.DataFrame(index=tasks)
    S["dataset"] = I.dataset
    S["task"] = [LABEL[t].replace("NO$_3$-N", "NO3-N") for t in tasks]
    S["r2_random"], S["r2_blocked"] = I.r2_random, I.r2_group
    for c in ("inflation", "inflation_lo", "inflation_hi"):
        S[c] = I[c]
    for cal, name in (("random", "random"), ("group", "blocked")):
        v = V[V.cal == cal].set_index("task")
        S[f"coverage90_cal_{name}"], S[f"coverage90_cal_{name}_lo"], S[f"coverage90_cal_{name}_hi"] = v.coverage, v.lo, v.hi
    S["location_r2_random"], S["location_r2_blocked"] = L.location_r2_random, L.location_r2_group
    matched = T.size_matched.astype(bool)
    S["neardup_share_train_removed"] = T.share_train_removed
    S["neardup_share_of_inflation"] = T.share_of_inflation.where(matched)
    S["neardup_share_of_inflation_sd"] = T.share_of_inflation_sd.where(matched)
    S["neardup_size_matched"] = matched
    for c in ("station_definition", "r2_station", "extra_r2_station", "extra_r2_station_lo", "extra_r2_station_hi", "share_hidden_by_station_grouping"):
        S[c] = G[c]
    S["test_within_1km_random"] = S.dataset.map(near["random"]).fillna(0.0)
    S["test_within_1km_blocked"] = S.dataset.map(near["group"]).fillna(0.0)
    S = S.reset_index(drop=True)
    num = S.select_dtypes("number").columns
    S[num] = S[num].round(3)
    write(S, folder, "S2_per_target_numbers.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=os.path.join(D.ROOT, "paper_results", "tables"))
    ap.add_argument("--out", default=os.path.join(D.ROOT, "output", "publication"))
    a = ap.parse_args()
    figures, tables, abstract = (os.path.join(a.out, k) for k in ("figures", "tables", "graphical_abstract"))
    for d in (figures, tables, abstract):
        os.makedirs(d, exist_ok=True)
    plt = pyplot()
    fig1(figures, plt)
    fig2(figures, plt)
    for fig in (fig3, fig4, fig5, fig6, fig7):
        fig(a.tables, figures, plt)
    graphical_abstract(a.tables, abstract, pyplot(big=False))
    table1(tables)
    for table in (table2, table3, table_s1, table_s2):
        table(a.tables, tables)


if __name__ == "__main__":
    main()
