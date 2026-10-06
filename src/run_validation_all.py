"""All chronological splits, one table. Primary question: do interaction terms help?"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import validate as V, synergy as S

OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def main(min_train=2):
    st = S.load_stints()
    seasons = sorted(st.season.unique())
    frames = []
    for i in range(min_train, len(seasons)):
        tr, te = seasons[:i], seasons[i]
        out, *_ = V.run(tr, te, verbose=False)
        out["test_season"] = te; out["n_train_seasons"] = i
        out["model"] = out.index
        frames.append(out.reset_index(drop=True))
        print(f"test {te} (train {i} seasons): " +
              "  ".join(f"{m}={out.loc[out.model == m, 'r2'].iloc[0]:.4f}" for m in
                        ["additive", "additive+ridgepair", "additive+bilinear",
                         "direct joint (M off)", "direct joint (M on)"]), flush=True)
    tab = pd.concat(frames, ignore_index=True)
    tab.to_csv(os.path.join(OUT, "validation.csv"), index=False)
    piv = tab.pivot_table(index="model", columns="test_season", values="r2")
    print("\n=== weighted out-of-sample R2 on held-out future lineups ===")
    print(piv.round(4).to_string())
    print("\nmean across splits:")
    print(piv.mean(axis=1).round(4).sort_values(ascending=False).to_string())
    pc = tab.pivot_table(index="model", columns="test_season", values="corr")
    print("\nmean correlation across splits:")
    print(pc.mean(axis=1).round(4).sort_values(ascending=False).to_string())
    return tab


if __name__ == "__main__":
    main()
