"""Charts for the win-projection backtest and the league-trend analysis."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import viz_common as V

OUT, FIGS = os.path.join(V.ROOT, "out"), os.path.join(V.VIZ, "wins")
POS, NEG = "#1b7f4f", "#c0392b"


def backtest_figure(path):
    tab = pd.read_csv(os.path.join(OUT, "preseason_backtest.csv"))
    scores = pd.read_csv(os.path.join(OUT, "preseason_model_scores.csv")).sort_values("mae")
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), dpi=125,
                             gridspec_kw=dict(width_ratios=[1.05, 1.05, 1.25]))

    for ax, col, title in ((axes[0], "prior_wins82", "baseline: last season's wins, carried over as-is"),
                           (axes[1], "pred_best", "projection: talent + continuity")):
        y, p = tab.wins82.values, tab[col].values
        mae = np.abs(y - p).mean()
        seasons = sorted(tab.season.unique())
        cmap = plt.get_cmap("viridis", len(seasons))
        for k, s in enumerate(seasons):
            m = tab.season == s
            ax.scatter(tab.loc[m, col], tab.loc[m, "wins82"], s=42, color=cmap(k),
                       edgecolor="k", linewidth=0.5, label=s, alpha=0.9)
        lims = [8, 72]
        ax.plot(lims, lims, color="#888", lw=1, ls="--")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("predicted wins (82-game pace)"); ax.set_ylabel("actual wins")
        ax.set_title(f"{title}\nMAE {mae:.2f} wins", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=7.5, frameon=False, loc="upper left", title="target season",
                   title_fontsize=8)

    ax = axes[2]
    s = scores.sort_values("mae", ascending=False)
    colors = [POS if "talent" in m else "#888" for m in s.model]
    ax.barh(np.arange(len(s)), s.mae, color=colors, alpha=0.9)
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(s.model, fontsize=8.5)
    for k, (m, v) in enumerate(zip(s.model, s.mae)):
        ax.text(v + 0.08, k, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("mean absolute error, wins (leave-one-season-out)")
    ax.set_xlim(0, 11.2)
    ax.set_title("nothing structural beats talent + continuity", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Preseason win projection, backtested on 180 team-seasons (2020-21 to 2025-26)",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def trends_figure(path):
    t = pd.read_csv(os.path.join(OUT, "league_trends.csv"))
    panels = [
        ("fg3a_rate", "3-point attempt rate", True),
        ("rim_freq", "shots at the rim", True),
        ("lmid_freq", "long midrange rate", True),
        ("ts", "true shooting", True),
        ("mean_pair_distance", "within-roster skill distance\n(are team-mates alike?)", False),
        ("eff_roles", "effective roles per roster", False),
        ("rotation_size", "rotation size", False),
        ("top3_minutes_share", "minutes share of top 3", False),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), dpi=125)
    x = np.arange(len(t))
    step = max(1, len(t) // 10)
    from scipy.stats import spearmanr
    for ax, (col, lab, strong) in zip(axes.ravel(), panels):
        rho, p = spearmanr(x, t[col].values)
        sig = p < 0.05
        ax.plot(x, t[col], marker="o", ms=3.5, lw=1.8,
                color=("#1b4f7f" if sig else "#999"))
        ax.set_xticks(x[::step])
        ax.set_xticklabels([s[2:] for s in t.season[::step]], fontsize=8, rotation=45)
        ax.set_title(f"{lab}\nrho {rho:+.2f}" + ("  (monotone)" if sig else "  (no trend)"),
                     fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eee")
    fig.suptitle(f"League trends over {len(t)} seasons ({t.season.iloc[0]} to {t.season.iloc[-1]})",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def projection_table(season="2025-26"):
    tab = pd.read_csv(os.path.join(OUT, "preseason_backtest.csv"))
    d = tab[tab.season == season].copy()
    d["miss"] = d.wins82 - d.pred_best
    return d.sort_values("pred_best", ascending=False)[
        ["team", "pred_best", "wins82", "prior_wins82", "miss", "talent_mw", "continuity"]]


def main():
    os.makedirs(FIGS, exist_ok=True)
    backtest_figure(os.path.join(FIGS, "backtest.png"))
    trends_figure(os.path.join(FIGS, "league_trends.png"))
    print("wrote viz/wins/backtest.png and viz/wins/league_trends.png")
    d = projection_table()
    d.to_csv(os.path.join(OUT, "projection_2025-26.csv"), index=False, encoding="utf-8")
    print("\n=== 2025-26 projection vs actual (model saw nothing from this season) ===")
    print(d.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
