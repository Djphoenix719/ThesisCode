import copy
import os.path
import random
from typing import Dict
import json

from trials.Evolution import mutate_fn
from trials.Search import search
from trials.Selection import drop_oldest_fn
from trials.Selection import drop_random_fn
from trials.Selection import drop_tournament_fn
from trials.Selection import drop_worst_fn
from trials.Selection import sel_best_fn
from trials.Selection import sel_e_greedy_fn
from trials.Selection import sel_middle_fn
from trials.Selection import sel_random_fn
from trials.Selection import sel_tournament_fn
from trials.Utilities import banner
from trials.Utilities import build_experiment_results
from trials.Utilities import get_spec
from trials.Utilities import random_spec
from trials.Utilities import write_table_html


def main():
    # gpu time in seconds
    MAX_TIME = 60 * 60 * 24 * 7
    # number of best models to maintain
    N_BEST = 5
    # number of models to drop each generation
    N_DROP = 5
    # size of the population that should be maintained
    N_POP = 100
    # number of trials that should be performed with a single set of parameters
    # note trials are deterministic but may differ slightly due to multiple samples
    # present in the the nasbench dataset.
    N_EPOCH = 5
    # should we use halfway training values
    STOP_HALFWAY = True

    state = random.getstate()
    random.seed(42)

    INITIAL_POPULATION = [
        random_spec(stop_halfway=STOP_HALFWAY) for idx in range(N_POP)
    ]

    random.setstate(state)

    BASE_TRIAL_ARGS = {
        "max_time": MAX_TIME,
        "num_best": N_BEST,
        "num_epochs": N_EPOCH,
    }

    def trial_name(text: str) -> str:
        return text

    def extend_trial(**kwargs) -> Dict:
        population = copy.deepcopy(INITIAL_POPULATION)
        return {**BASE_TRIAL_ARGS, "initial_population": population, **kwargs}

    trials: dict = dict()

    # GREEDY SELECTION + OLDEST DROP
    trials[trial_name(f"Select 5 Greedy + Drop 5 Oldest")] = extend_trial(
        **{
            "mut_fn": mutate_fn(1.0),
            "sel_fn": sel_best_fn(N_BEST),
            "drp_fn": drop_oldest_fn(N_DROP),
        }
    )

    # RANDOM SELECTION + OLDEST DROP
    trials[trial_name(f"Select 5 Random + Drop 5 Oldest")] = extend_trial(
        **{
            "mut_fn": mutate_fn(1.0),
            "sel_fn": sel_random_fn(N_BEST),
            "drp_fn": drop_oldest_fn(N_DROP),
        }
    )

    # etc...

    # jury-rigged multi-threading, just re-run the program multiple times
    # can't pickle functions so we do it like this, rather than refactor
    # would have to create the functions on child threads to do it "right"
    # deterministic evaluation means even if we overwrite the files we're ok
    items = list(trials.items())
    random.shuffle(items)
    # items.reverse()
    for name, params in items:
        path = os.path.join('D://', "Graph Data", name)

        if os.path.exists(path):
            print(f"Skipping {name}, it already exists")
            continue

        banner(name)
        results = search(**params)

        for idx, [population, best, done, generations] in enumerate(results):
            epoch_path = os.path.join(path, f"Epoch {idx+1}")

            if os.path.exists(epoch_path):
                continue
            os.makedirs(epoch_path)

            generations_path = os.path.join(epoch_path, f"Heredity")
            os.makedirs(generations_path, exist_ok=True)
            for idx, generation in enumerate(generations):
                generation = [ind.get_hash() for ind in generation]
                json_path = os.path.join(generations_path, f"Generation{idx:06d}.json")

                with open(json_path, "w", encoding="utf8") as handle:
                    handle.write(json.dumps(generation))


if __name__ == "__main__":
    main()
