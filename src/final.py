"""Final end-to-end run on all seasons; prints every result used in FINDINGS.md.

  python src/final.py --from-step 3    # resume (earlier steps' artifacts are on disk)
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import synergy as S, direct as D, run_all as RA

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "proc")


def banner(n, txt):
    print("\n" + "=" * 70); print(f"STEP {n}  {txt}"); print("=" * 70, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-step", type=int, default=1)
    ap.add_argument("--to-step", type=int, default=7)
    a = ap.parse_args()
    lo, hi = a.from_step, a.to_step

    if lo <= 1 <= hi:
        banner(1, "pair ridge + two-stage bilinear + hypothesis patterns")
        RA.main()

    players = pd.read_parquet(os.path.join(PROC, "players.parquet"))
    st = S.load_stints(); st["players"] = st.lineup.str.split("-")

    if lo <= 2 <= hi:
        banner(2, "direct joint fit on all seasons")
        fo = D.fit(st, players, "off", dims=6)
        fd = D.fit(st, players, "def", dims=6)
        M = D.net_M(fo, fd); np.save(os.path.join(PROC, "M_direct.npy"), M)
        V = fo["Z"].values[:, :6]
        Sx = V @ M @ V.T; iu = np.triu_indices(len(V), 1)
        print("direct surface spread: %.3f pts/100 per pair (%.2f per 10-pair lineup)"
              % (Sx[iu].std(), Sx[iu].std() * np.sqrt(10)))
        print("M eigenvalues:", np.round(np.linalg.eigvalsh(M), 3))

    if lo <= 3 <= hi:
        banner(3, "reliability")
        import reliability as R
        R.report(st, players, label="pooled all seasons")
        R.bilinear_stability(st, players)
        R.direct_stability_by_size(st, players)

    if lo <= 4 <= hi:
        banner(4, "chronological validation")
        import run_validation_all as RV
        RV.main(min_train=2)

    if lo <= 5 <= hi:
        banner(5, "forward roster test")
        import run_rosters as RR
        RR.main()

    if lo <= 6 <= hi:
        banner(6, "player moves")
        import run_moves as RM
        RM.main()

    if lo <= 7 <= hi:
        banner(7, "topology figures + roster metrics")
        import run_topology as RT
        RT.main()
    print("\nDONE")


if __name__ == "__main__":
    main()
