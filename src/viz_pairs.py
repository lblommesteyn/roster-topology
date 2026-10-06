"""Explainability panels: why does the model like or dislike a particular pair?

The predicted synergy z_i' M z_j decomposes exactly over the eigenvectors of M:

    z_i' M z_j = sum_k  lambda_k (v_k . z_i)(v_k . z_j)

so each mode contributes a signed amount, and each mode can be described in skill language by
projecting its direction back onto the standardized features. That turns an abstract number into
"these two both load on the interior-scoring pole of a mode where doubling up is penalised".
"""
import os, sys, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V

DIMS = 6
SKILLS = ["z_f_rim", "z_fg3a_pct", "z_usage", "z_ast100", "z_blk100", "z_oreb_fg", "z_dreb_fg",
          "z_assisted3", "z_ts", "z_stl100", "z_f_lmid", "z_tov100"]
PRETTY = {"z_f_rim": "rim rate", "z_fg3a_pct": "3pt rate", "z_usage": "usage",
          "z_ast100": "assists", "z_blk100": "blocks", "z_oreb_fg": "off reb",
          "z_dreb_fg": "def reb", "z_assisted3": "catch-and-shoot 3s", "z_ts": "true shooting",
          "z_stl100": "steals", "z_f_lmid": "long midrange", "z_tov100": "turnovers"}


def mode_language(ctx, n_modes=3):
    """Describe each eigen-mode of M in skill terms, with its sign convention."""
    ev, vec = np.linalg.eigh(ctx.M)
    order = np.argsort(-np.abs(ev))
    pl = ctx.players
    W = np.linalg.lstsq(pl[SKILLS].values, pl[[f"pc{i+1}" for i in range(DIMS)]].values,
                        rcond=None)[0]            # skills -> pc weights
    out = []
    for k in order[:n_modes]:
        d = vec[:, k]
        load = pd.Series(W @ d, index=SKILLS).sort_values()
        out.append(dict(idx=int(k), lam=float(ev[k]), vec=d,
                        high=[PRETTY[c] for c in load.index[-3:][::-1]],
                        low=[PRETTY[c] for c in load.index[:3]]))
    return out


def decompose(ctx, pid_a, pid_b, modes):
    za = ctx.Z.loc[pid_a].values[:DIMS]
    zb = ctx.Z.loc[pid_b].values[:DIMS]
    ev, vec = np.linalg.eigh(ctx.M)
    rows = []
    for m in modes:
        k = m["idx"]
        pa, pb = float(vec[:, k] @ za), float(vec[:, k] @ zb)
        row = dict(m)
        row.update(proj_a=pa, proj_b=pb, contrib=float(ev[k] * pa * pb))
        rows.append(row)
    total = float(za @ ctx.M @ zb)
    return rows, total


def comparable_pairs(ctx, season, pid_a, pid_b, k=5):
    """Other pairs whose two players sit near these two in skill space."""
    df = ctx.players[(ctx.players.season == season) &
                     (ctx.players.minutes >= ctx.min_minutes)].drop_duplicates("pid")
    za, zb = ctx.Z.loc[pid_a].values[:DIMS], ctx.Z.loc[pid_b].values[:DIMS]
    ids = [p for p in df.pid if p in ctx.Z.index]
    Zs = np.array([ctx.Z.loc[p].values[:DIMS] for p in ids])
    da = np.linalg.norm(Zs - za, axis=1)
    db = np.linalg.norm(Zs - zb, axis=1)
    name = dict(zip(df.pid, df.name))
    out = []
    excl = {pid_a, pid_b}
    for i, j in itertools.combinations(range(len(ids)), 2):
        if ids[i] in excl or ids[j] in excl:      # a comparable pair must be two other players
            continue
        d = min(da[i] + db[j], da[j] + db[i])
        out.append((d, ids[i], ids[j]))
    out.sort()
    return [(name[a], name[b], ctx.syn(a, b), d) for d, a, b in out[:k]]


def measured(ctx, pid_a, pid_b):
    """The observed pair coefficient and shared possessions, with its (large) uncertainty."""
    p = ctx.pairs
    lo, hi = (pid_a, pid_b) if pid_a < pid_b else (pid_b, pid_a)
    row = p[(p.p1 == lo) & (p.p2 == hi)]
    if not len(row):
        return None
    r = row.iloc[0]
    return dict(net=float(r.net), poss=float(r.poss))


def panel(ctx, season, pid_a, pid_b, path, modes=None):
    modes = modes or mode_language(ctx)
    all_modes = mode_language(ctx, n_modes=DIMS)
    rows, total = decompose(ctx, pid_a, pid_b, modes)
    all_rows, _ = decompose(ctx, pid_a, pid_b, all_modes)
    pl = ctx.players[ctx.players.season == season].drop_duplicates("pid").set_index("pid")
    a, b = pl.loc[pid_a], pl.loc[pid_b]
    fig = plt.figure(figsize=(13.5, 7.4), dpi=120)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.32)

    # 1. skill bars for the two players
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(SKILLS))
    ax.barh(y - 0.2, [a[c] for c in SKILLS], height=0.4, color="#2b6cb0", label=V.surname(a["name"]))
    ax.barh(y + 0.2, [b[c] for c in SKILLS], height=0.4, color="#dd6b20", label=V.surname(b["name"]))
    ax.set_yticks(y); ax.set_yticklabels([PRETTY[c] for c in SKILLS], fontsize=8)
    ax.axvline(0, color="#333", lw=0.8); ax.set_xlabel("standard deviations vs league", fontsize=9)
    ax.set_title("skill profiles", fontsize=10); ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # 2. mode contributions
    ax = fig.add_subplot(gs[0, 1])
    labs, vals = [], []
    for r in sorted(all_rows, key=lambda x: -abs(x["contrib"])):
        labs.append(f"mode {r['idx']}\n(lam {r['lam']:+.2f})")
        vals.append(r["contrib"])
    ax.bar(range(len(vals)), vals, color=[V.POS if v > 0 else V.NEG for v in vals], alpha=0.9)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=8)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylabel("contribution to fit (pts/100)", fontsize=9)
    ax.set_title(f"total predicted fit {total:+.2f} pts/100", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # 3. text explanation
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    lines = [f"{a['name']}  +  {b['name']}", f"{season}", "",
             f"predicted fit: {total:+.2f} pts/100", ""]
    m = measured(ctx, pid_a, pid_b)
    if m:
        se = 30.0 / np.sqrt(max(m["poss"], 1))       # rough scale of a pair estimate's noise
        lines += [f"measured together: {int(m['poss'])} possessions",
                  f"observed pair effect {m['net']:+.2f} +/- ~{se:.2f} (1 se)",
                  "(single-pair estimates are ~94% noise; the", " model value is the reliable one)", ""]
    else:
        lines += ["never shared the floor in this sample:", "fit comes from the skill model only", ""]
    for r in rows:
        sign = "same pole helps" if r["lam"] > 0 else "same pole hurts"
        who = ("both on the " + ("+" if r["proj_a"] > 0 else "-") + " side"
               if np.sign(r["proj_a"]) == np.sign(r["proj_b"]) else "opposite sides")
        lines.append(f"mode {r['idx']} ({sign}): {who}")
        lines.append(f"   + pole: {', '.join(r['high'])}")
        lines.append(f"   - pole: {', '.join(r['low'])}")
        lines.append(f"   contributes {r['contrib']:+.2f}")
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.2)

    # 4. comparable pairs
    ax = fig.add_subplot(gs[1, 0:2])
    comps = comparable_pairs(ctx, season, pid_a, pid_b, k=7)
    labels = [f"{V.surname(x)} + {V.surname(y)}" for x, y, _, _ in comps]
    vals = [s for _, _, s, _ in comps]
    yy = np.arange(len(vals))
    ax.barh(yy, vals, color=[V.POS if v > 0 else V.NEG for v in vals], alpha=0.85)
    ax.axvline(total, color="#333", ls="--", lw=1.2, label="this pair")
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted fit (pts/100)", fontsize=9)
    ax.set_title("most similar pairs elsewhere in the league", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)

    # 5. where the two sit in skill space
    ax = fig.add_subplot(gs[1, 2])
    lg = ctx.players[(ctx.players.season == season) & (ctx.players.minutes >= ctx.min_minutes)]
    ax.scatter(lg.pc1, lg.pc2, s=10, c="#ddd", edgecolor="none")
    ax.scatter([a.pc1], [a.pc2], s=140, c="#2b6cb0", edgecolor="k")
    ax.scatter([b.pc1], [b.pc2], s=140, c="#dd6b20", edgecolor="k")
    ax.annotate(V.surname(a["name"]), (a.pc1, a.pc2), fontsize=8, xytext=(0, 9),
                textcoords="offset points", ha="center")
    ax.annotate(V.surname(b["name"]), (b.pc1, b.pc2), fontsize=8, xytext=(0, 9),
                textcoords="offset points", ha="center")
    ax.plot([a.pc1, b.pc1], [a.pc2, b.pc2], color=(V.POS if total > 0 else V.NEG), lw=2.2)
    ax.set_xlabel("PC1  spacing <-> rim and glass", fontsize=9)
    ax.set_ylabel("PC2  off-ball <-> on-ball load", fontsize=9)
    ax.set_title("position in league skill space", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Why the model {'likes' if total > 0 else 'dislikes'} "
                 f"{a['name']} + {b['name']}   ({total:+.2f} pts/100)", fontsize=13, y=0.99)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def extremes(ctx, season, n=4, min_minutes=1200):
    df = ctx.players[(ctx.players.season == season) &
                     (ctx.players.minutes >= min_minutes)].drop_duplicates("pid")
    ids = [p for p in df.pid if p in ctx.Z.index]
    Zs = np.array([ctx.Z.loc[p].values[:DIMS] for p in ids])
    S = Zs @ ctx.M @ Zs.T
    iu = np.triu_indices(len(ids), 1)
    vals = S[iu]
    order = np.argsort(vals)
    worst = [(ids[iu[0][k]], ids[iu[1][k]], vals[k]) for k in order[:n]]
    best = [(ids[iu[0][k]], ids[iu[1][k]], vals[k]) for k in order[-n:][::-1]]
    return best, worst


def main(season=None):
    ctx = V.Ctx(); V.ensure_dirs()
    season = season or ctx.seasons[-1]
    out = os.path.join(V.VIZ, "pairs")
    modes = mode_language(ctx)
    print("interaction modes:")
    for m in modes:
        print(f"  mode {m['idx']} lambda {m['lam']:+.3f}  + pole: {', '.join(m['high'])}"
              f"   - pole: {', '.join(m['low'])}")
    best, worst = extremes(ctx, season)
    for tag, lst in (("best", best), ("worst", worst)):
        for a, b, s in lst:
            na = V.surname(ctx.players[ctx.players.pid == a].name.iloc[0])
            nb = V.surname(ctx.players[ctx.players.pid == b].name.iloc[0])
            panel(ctx, season, a, b, os.path.join(out, f"pair_{tag}_{na}_{nb}.png"), modes)
            print(f"  {tag}: {na} + {nb}  {s:+.2f}")
    # one pair that actually shares a floor a lot, for contrast with the extremes
    big = ctx.pairs.nlargest(200, "poss")
    for r in big.itertuples():
        if r.p1 in ctx.Z.index and r.p2 in ctx.Z.index:
            pa = ctx.players[(ctx.players.pid == r.p1) & (ctx.players.season == season)]
            pb = ctx.players[(ctx.players.pid == r.p2) & (ctx.players.season == season)]
            if len(pa) and len(pb):
                panel(ctx, season, r.p1, r.p2,
                      os.path.join(out, f"pair_highminutes_{V.surname(pa.name.iloc[0])}_"
                                        f"{V.surname(pb.name.iloc[0])}.png"), modes)
                print("  high-minutes pair:", pa.name.iloc[0], "+", pb.name.iloc[0])
                break


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
