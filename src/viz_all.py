"""Regenerate the whole visualization set. Run after any change to the model or the shared scales.

  python src/viz_all.py            # everything
  python src/viz_all.py static 4d  # only the named stages
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(__file__))

STAGES = ["static", "interactive", "embedding", "evolution", "4d", "swaps", "pairs", "index"]


def main(stages=None):
    stages = stages or STAGES
    t0 = time.time()
    if "static" in stages:
        import viz_static; viz_static.main(); print(f"[{time.time()-t0:.0f}s] static done", flush=True)
    if "interactive" in stages:
        import viz_interactive; viz_interactive.main(); print(f"[{time.time()-t0:.0f}s] interactive done", flush=True)
    if "embedding" in stages:
        import viz_embedding; viz_embedding.main(); print(f"[{time.time()-t0:.0f}s] embedding done", flush=True)
    if "evolution" in stages:
        import viz_evolution; viz_evolution.main(); print(f"[{time.time()-t0:.0f}s] evolution done", flush=True)
    if "4d" in stages:
        import viz_4d; viz_4d.main(); print(f"[{time.time()-t0:.0f}s] 4d done", flush=True)
    if "swaps" in stages:
        import viz_swaps; viz_swaps.main(); print(f"[{time.time()-t0:.0f}s] swaps done", flush=True)
    if "pairs" in stages:
        import viz_pairs; viz_pairs.main(); print(f"[{time.time()-t0:.0f}s] pairs done", flush=True)
    if "index" in stages:
        import viz_index; viz_index.main(); print(f"[{time.time()-t0:.0f}s] index done", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
