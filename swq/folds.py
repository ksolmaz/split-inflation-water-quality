import os

import pandas as pd

from swq import data as D


def main():
    out = os.path.join(D.ROOT, "paper_results", "folds")
    os.makedirs(out, exist_ok=True)
    for t in D.ALL_TASKS:
        task = D.get_task(t)
        cols = {"row": range(task.n), "id": task.ids, "block": task.grp, "station": D.stations(task)}
        for scheme in D.SCHEMES:
            for s in D.SEEDS[task.domain]:
                cols[f"{scheme}_seed{s}"] = D.folds(task, scheme, s)
        path = os.path.join(out, t.replace("/", "_") + "_folds.csv.gz")
        pd.DataFrame(cols).to_csv(path, index=False)
        print("wrote", path)


if __name__ == "__main__":
    main()
