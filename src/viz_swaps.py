"""Counterfactual roster swap visuals: before/after graph, edge deltas, skill-space landing spot."""
import os, sys, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V, viz_plotly as P, viz_static as VS, topology as TP, swaps as SW


def build_swap(ctx, team, season, outgoing, incoming, inc_season=None):
    """Roster before and after, with the incoming player inheriting the vacated minutes."""
    roster = ctx.roster(team, season)
    if outgoing not in set(roster.name):
        raise ValueError(f"{outgoing} not in {team} {season}: {', '.join(roster.name)}")
    cand = ctx.players[ctx.players.name.str.lower() == incoming.lower()]
    if inc_season:
        cand = cand[cand.season == inc_season]
    if not len(cand):
        raise ValueError(f"unknown incoming player {incoming}")
    inc = cand.sort_values("season").iloc[-1].copy()
    idx = roster.index[roster.name == outgoing][0]
    inc["minutes"] = roster.loc[idx, "minutes"]
    new = pd.concat([roster.drop(index=idx), pd.DataFrame([inc])], ignore_index=True)
    new = new.sort_values("minutes", ascending=False).reset_index(drop=True)
    return roster, new, inc


def edge_delta_chart(ctx, roster, new, outgoing, incoming, ax):
    """Per-team-mate change in fit when the outgoing player is swapped for the incoming one."""
    keep = [p for p in roster.pid if p != roster.loc[roster.name == outgoing, "pid"].iloc[0]]
    names = {p: n for p, n in zip(roster.pid, roster.name)}
    out_pid = roster.loc[roster.name == outgoing, "pid"].iloc[0]
    inc_pid = new.loc[new.name == incoming, "pid"].iloc[0]
    rows = []
    for p in keep:
        rows.append((names[p], ctx.syn(out_pid, p), ctx.syn(inc_pid, p)))
    d = pd.DataFrame(rows, columns=["name", "before", "after"])
    d["delta"] = d.after - d.before
    d = d.sort_values("delta")
    y = np.arange(len(d))
    ax.barh(y, d.delta, color=[V.POS if v > 0 else V.NEG for v in d.delta], alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels([V.surname(n) for n in d.name], fontsize=8)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("change in fit with team-mate (pts/100)", fontsize=9)
    ax.set_title(f"who gains and loses:  {V.surname(outgoing)} -> {V.surname(incoming)}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return d


def swap_figure(ctx, team, season, outgoing, incoming, path, inc_season=None):
    roster, new, inc = build_swap(ctx, team, season, outgoing, incoming, inc_season)
    fig = plt.figure(figsize=(17, 10.5), dpi=115)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.28, wspace=0.22)
    ax_b = fig.add_subplot(gs[0, 0]); ax_a = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[0, 2]); ax_e = fig.add_subplot(gs[1, :2])
    ax_t = fig.add_subplot(gs[1, 2]); ax_t.axis("off")

    for ax, r, tag in ((ax_b, roster, "before"), (ax_a, new, "after")):
        W = ctx.syn_matrix(r)
        pos = TP.synergy_mds(W)
        vmax = ctx.edge_scale
        for i, j, w in V.edge_list(r, W, 0.14):
            f = min(abs(w) / vmax, 1.4)
            ax.plot(*zip(pos[i], pos[j]), lw=0.4 + 3.4 * f,
                    color=(V.POS if w > 0 else V.NEG), alpha=min(0.9, 0.32 + 0.42 * f), zorder=1)
        s = 90 + 820 * (r.minutes / r.minutes.max())
        hl = (r.name == incoming) if tag == "after" else (r.name == outgoing)
        ax.scatter(pos[:, 0], pos[:, 1], s=s, c=r.pc1, cmap="coolwarm", vmin=-6, vmax=6,
                   edgecolor=np.where(hl, "#d95f02", "k"),
                   linewidth=np.where(hl, 3.0, 0.7), zorder=2)
        for k, l in enumerate(V.short_labels(list(r.name))):
            ax.annotate(l, pos[k], textcoords="offset points",
                        xytext=(0, -(np.sqrt(s.iloc[k]) / 2 + 7)), fontsize=7.4,
                        ha="center", va="top", zorder=3)
        st = ctx.structure(r)
        ax.set_title(f"{tag}: complementarity {st['complementarity']:+.3f} | "
                     f"roles {st['eff_roles']:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)

    d = edge_delta_chart(ctx, roster, new, outgoing, incoming, ax_d)

    # where the incoming player lands in league skill space
    lg = ctx.players[(ctx.players.season == season) & (ctx.players.minutes >= ctx.min_minutes)]
    ax_e.scatter(lg.pc1, lg.pc2, s=12, c="#ccc", edgecolor="none", zorder=1)
    ax_e.scatter(roster.pc1, roster.pc2, s=70, c=roster.pc1, cmap="coolwarm", vmin=-6, vmax=6,
                 edgecolor="k", linewidth=0.6, zorder=2)
    for _, row in roster.iterrows():
        ax_e.annotate(V.surname(row["name"]), (row.pc1, row.pc2), fontsize=7,
                      xytext=(0, 6), textcoords="offset points", ha="center")
    o = roster[roster.name == outgoing].iloc[0]
    ax_e.scatter([o.pc1], [o.pc2], s=250, facecolor="none", edgecolor="#777", linewidth=2.4,
                 zorder=3, label=f"out: {outgoing}")
    ax_e.scatter([inc.pc1], [inc.pc2], s=290, facecolor="none", edgecolor="#d95f02", linewidth=3,
                 zorder=4, label=f"in: {incoming}")
    ax_e.annotate("", xy=(inc.pc1, inc.pc2), xytext=(o.pc1, o.pc2),
                  arrowprops=dict(arrowstyle="->", color="#d95f02", lw=2.2, alpha=0.9), zorder=3)
    ax_e.set_xlabel("PC1   spacing <-> rim and glass")
    ax_e.set_ylabel("PC2   off-ball <-> on-ball load")
    ax_e.set_title("where the swap moves the roster in league skill space", fontsize=10)
    ax_e.legend(fontsize=8, frameon=False, loc="best")
    ax_e.spines[["top", "right"]].set_visible(False)

    # metric table + best predicted units
    sb, sa = ctx.structure(roster), ctx.structure(new)
    val_b = float(np.average(roster.value, weights=roster.minutes))
    val_a = float(np.average(new.value, weights=new.minutes))
    lines = [f"{team} {season}:  OUT {outgoing}  ->  IN {incoming} ({inc.season})", ""]
    for k in ("complementarity", "redundancy", "eff_roles", "fragility"):
        lines.append(f"{k:16s} {sb[k]:+.3f}  ->  {sa[k]:+.3f}   ({sa[k]-sb[k]:+.3f})")
    lines.append(f"{'value (mw)':16s} {val_b:+.3f}  ->  {val_a:+.3f}   ({val_a-val_b:+.3f})")
    lines.append(f"{'connector':16s} {sb['connector']}  ->  {sa['connector']}")
    lines += ["", "biggest fit gains:"]
    for _, r in d.tail(3)[::-1].iterrows():
        lines.append(f"   {V.surname(r['name']):<16s} {r.delta:+.3f}")
    lines += ["", "biggest fit losses:"]
    for _, r in d.head(3).iterrows():
        lines.append(f"   {V.surname(r['name']):<16s} {r.delta:+.3f}")
    best_b, _ = SW.predicted_top_lineup(roster, ctx.syn, ctx.eff_o.to_dict(), ctx.eff_d.to_dict(), top=2)
    best_a, _ = SW.predicted_top_lineup(new, ctx.syn, ctx.eff_o.to_dict(), ctx.eff_d.to_dict(), top=2)
    lines += ["", "best predicted unit before:",
              "   " + ", ".join(V.surname(x) for x in best_b[0][1]),
              "best predicted unit after:",
              "   " + ", ".join(V.surname(x) for x in best_a[0][1])]
    ax_t.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.6)
    fig.suptitle(f"Counterfactual swap:  {team} {season}   {outgoing}  ->  {incoming}",
                 fontsize=14, y=0.985)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, d, (sb, sa)


def trade_before_after(ctx, team, s0, s1, path):
    """A real transaction, both sides real: the team's topology the season before and after.

    Players who left are ringed grey in the first panel, arrivals ringed orange in the second.
    """
    lay = V.aligned_team_layouts(ctx, team, seasons=[s0, s1])
    if len(lay) < 2:
        return None
    r0, p0 = lay[s0]; r1, p1 = lay[s1]
    left = set(r0.pid) - set(r1.pid)
    joined = set(r1.pid) - set(r0.pid)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), dpi=115)
    for ax, r, pos, season, mark, col in ((axes[0], r0, p0, s0, left, "#777"),
                                          (axes[1], r1, p1, s1, joined, "#d95f02")):
        W = ctx.syn_matrix(r)
        vmax = ctx.edge_scale
        for i, j, w in V.edge_list(r, W, 0.14):
            f = min(abs(w) / vmax, 1.4)
            ax.plot(*zip(pos[i], pos[j]), lw=0.4 + 3.4 * f,
                    color=(V.POS if w > 0 else V.NEG), alpha=min(0.9, 0.32 + 0.42 * f), zorder=1)
        sz = 90 + 820 * (r.minutes / r.minutes.max())
        hl = r.pid.isin(mark).values
        ax.scatter(pos[:, 0], pos[:, 1], s=sz, c=r.pc1, cmap="coolwarm", vmin=-6, vmax=6,
                   edgecolor=np.where(hl, col, "k"), linewidth=np.where(hl, 3.0, 0.7), zorder=2)
        for k, l in enumerate(V.short_labels(list(r.name))):
            ax.annotate(l, pos[k], textcoords="offset points",
                        xytext=(0, -(np.sqrt(sz.iloc[k]) / 2 + 7)), fontsize=7.4,
                        ha="center", va="top", zorder=3)
        st = ctx.structure(r)
        tag = "departed ringed grey" if season == s0 else "arrivals ringed orange"
        ax.set_title(f"{season}   net {ctx.net_rating(team, season):+.1f} | "
                     f"comp {st['complementarity']:+.3f} | {tag}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    names0 = dict(zip(r0.pid, r0.name)); names1 = dict(zip(r1.pid, r1.name))
    fig.suptitle(f"{team}: {s0} -> {s1}   out: "
                 + ", ".join(V.surname(names0[p]) for p in list(left)[:5])
                 + "   in: " + ", ".join(V.surname(names1[p]) for p in list(joined)[:5]),
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


TRADES = [
    ("NYK", "2023-24", "2024-25", "Towns and Bridges arrive"),
    ("PHX", "2022-23", "2023-24", "Durant / Beal era begins"),
    ("BOS", "2022-23", "2023-24", "Porzingis and Holiday arrive"),
    ("DAL", "2024-25", "2025-26", "post-Doncic roster"),
    ("MIN", "2021-22", "2022-23", "Gobert arrives"),
]


CASES = [
    # team, season, outgoing, incoming, incoming season (None = latest), note
    ("MIN", "2025-26", "Rudy Gobert", "LaMelo Ball", None, "add a lead guard next to Edwards"),
    ("GSW", "2025-26", "Al Horford", "Nikola Jokic", None, "least complementary roster + a passing hub"),
    ("OKC", "2025-26", "Isaiah Hartenstein", "Draymond Green", None, "swap a rim big for a passing big"),
    ("MEM", "2025-26", "Javon Small", "Rudy Gobert", None, "guard-heavy roster gains a rim big"),
    ("DAL", "2025-26", "Daniel Gafford", "Myles Turner", None, "double-big roster, add spacing"),
]


def main():
    ctx = V.Ctx(); V.ensure_dirs()
    out = os.path.join(V.VIZ, "swaps")
    made = []
    for team, season, og, inc, isea, note in CASES:
        try:
            p, d, _ = swap_figure(ctx, team, season, og, inc,
                                  os.path.join(out, f"swap_{team}_{V.surname(og)}_to_{V.surname(inc)}.png"),
                                  inc_season=isea)
            made.append((team, og, inc, note))
            print("  swap:", team, og, "->", inc)
        except Exception as e:
            print("  SKIP", team, og, "->", inc, ":", e)
    for team, s0, s1, note in TRADES:
        try:
            trade_before_after(ctx, team, s0, s1,
                               os.path.join(out, f"trade_{team}_{s0}_to_{s1}.png"))
            print("  trade:", team, s0, "->", s1, f"({note})")
        except Exception as e:
            print("  SKIP trade", team, ":", e)
    return made


if __name__ == "__main__":
    main()
