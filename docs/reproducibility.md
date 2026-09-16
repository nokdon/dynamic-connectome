# Reproducibility

## Three different operations

| Operation | Command | What it establishes |
|---|---|---|
| Small inference demo | `dc demo` | Bundled models load and run without training or a teacher. |
| Full checkpoint evaluation | `dc evaluate --output outputs/eval.json` | Regenerates the published 4,096-example diagnostic. |
| New training | `dc train --config configs/research.json --output outputs/run` | Runs an explicit protocol with repaired selection; does not reproduce historical selection. |

All commands use the installed package's artifacts by default, not a notebook or a machine-specific working directory. `dc evaluate --artifacts PATH` evaluates models produced by a new training run. `dc plot` needs the optional `plots` dependencies; inference and training do not import Matplotlib.

## Environment

`pyproject.toml` pins the core numerical packages. `requirements-lock.txt` pins their transitive dependencies plus the plotting and test tools. Python 3.13 on Linux CPU is the tested environment. Python 3.12+ is the declared package range; other Python/OS/backend combinations have not all been tested. The [CPU checks workflow](https://github.com/nokdon/dynamic-connectome/actions/workflows/tests.yml) runs installation, tests and the demo on GitHub.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e '.[dev,plots]'
JAX_PLATFORMS=cpu pytest -q
JAX_PLATFORMS=cpu dc demo
```

No GPU is required. Installing a GPU-specific JAX distribution is outside this verified setup. Numerical differences on other backends may change thresholded spikes; compare results with a stated tolerance rather than assuming bitwise portability.

## Artifact contract

`src/dynamic_connectome/_artifacts/` contains one numeric problem file, 26 numeric model files, and `catalog.json`. Each model stores the transition matrix and flat parameter arrays. Loading uses `allow_pickle=False`. The catalog records complete inference configs, tensor shapes, original checkpoint seed/family, and SHA-256 hashes. Validation checks finite float32 parameters, dimensions, candidate support, row normalization, and sparse degree constraints.

Historical local pickles were converted once in the trusted source workspace. The release does not load, download, or distribute those pickles. The imported-file manifest records original relative paths and hashes without personal home paths. SHA-256 detects accidental drift against the supplied manifest; it is not a signature or proof of upstream authorship.

## New training protocol

`configs/smoke.json` uses a two-dimensional payload and eight hidden units, three updates per structural stage, and two spiking updates. It checks the entire pipeline cheaply. `configs/research.json` uses the original 8-dimensional/64-hidden architecture and 300 Stage 1 updates, 200 Stage 2 updates, and 100 × 4 B3 updates. All values are stored in the output configuration.

Keys derive from `fold_in(PRNGKey(seed), stream_id)`. Streams 0/1/2/3 initialize and train the core; 100 is validation and 200 is final testing. Training batches use one-based step indices folded into their stream key; evaluation batches use `split`. Baselines use streams 0 and 50 for initialization/training, 51 for soft adjacency initialization, and the same 100/200 validation/test convention. Stream separation is pseudorandom key separation, not separate physical datasets.

Stage 1 selection uses structural reachability and density on the known pair set. Stage 2 and B3 select by mean validation MSE after updates, including their initial states. Test draws are generated only after selection finishes. Running repeated experiments and adapting to their test results would turn those results into a development set; reserve a new seed for a future confirmatory study.

Every run writes the effective config, selected Stage 1 parameters, numeric inference artifacts, validation history, selected step, test metrics, and stream IDs. Output directories are created exclusively to avoid accidentally mixing experiments. Optimizer state is not exported; the command is a complete run, not a resumable training service.

For matched new controls:

```bash
JAX_PLATFORMS=cpu dc train --config configs/research.json --output outputs/run
JAX_PLATFORMS=cpu dc train-baselines --config configs/research.json --reference-artifacts outputs/run --output outputs/controls
```

Without `--reference-artifacts`, random controls use the bundled canonical seed-0 graph. The heuristic independently obeys the configured row/edge budget; a newly learned graph may have fewer than 44 edges if some second gates close. Record actual edge counts before calling those new results exactly edge-matched.

## Verification scope

Tests cover graph support and isolated rows, finite surrogate gradients, message direction, hard spike forward values, teacher-free and batch-independent inference, numeric checkpoint round-trips, corruption detection, exact control graph recovery, and an end-to-end smoke that reloads selected models and recomputes their recorded validation/test scores. Historical checkpoint evaluations were compared against the original source implementation during curation.

See [local verification](../results/verification.md) for the checks actually run for this candidate. This is not a full retraining of the original research campaign.
