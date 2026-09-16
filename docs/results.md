# Results and interpretation

The README table is generated from [checkpoint_evaluation.json](../results/checkpoint_evaluation.json). It evaluates 26 existing checkpoints: five seeds for each of five structural families and one selected hybrid B3 checkpoint. The source inventory and hashes are in [the import manifest](../provenance/import_manifest.json); complete inference settings are embedded in the artifact catalog.

The diagnostic uses `PRNGKey(20260916)`, split into 16 batch keys, with 256 examples per batch. All models see identical examples. Error is averaged per coordinate, then over examples. Family standard deviations use `ddof=0` over checkpoint-level means. They describe these five seeds, not a population confidence interval. The evaluation key was chosen for the publication review and was not used to retrain or select any of these bundled checkpoints. It is public now and should not be treated as a hidden future test set.

## What the comparison supports

Learned sparse graphs score 0.50949 MSE, compared with 0.67875 for random sparse controls and 0.67457 for the shortest-path controls. The reductions are approximately 24.94% and 24.47%, respectively. Both controls have 44 edges. Random graphs match the row-degree pattern of one canonical graph, not every learned seed independently.

Fixed dense graphs score 0.33296, and dense soft graphs score 0.16357. The experiment therefore shows a real accuracy cost of the hard edge budget. It does not establish a sparse model superior to unconstrained dense models.

B3 scores 0.25972 on the same examples. It has 12 recurrent steps, analog messages, a synaptic trace, different input coding, and a richer readout. There is one selected checkpoint. Its result is a useful continuation, not an isolated demonstration that spikes improve accuracy or efficiency.

## Why the earlier headline was replaced

The historical learned-graph mean of about 0.272 was calculated from the minimum stochastic training-batch loss per seed. It was not held-out MSE of the saved checkpoint. The old loop also associated a pre-update loss with post-update weights. The original fixed-graph and dense-soft baseline selection shared the training-minimum issue. New-draw evaluation makes the bundled weights directly inspectable, but does not retroactively repair the old selection procedure or eliminate historical model-search bias.

A useful consistency check comes from graph reachability. The canonical seed-0 graph reaches about 54.6% of admissible pairs in three message steps. For unreachable pairs, the target representation cannot contain the independent unit-variance payload. The minimum population MSE for those episodes is one, so the overall population MSE is at least about 0.454. Its fresh score is 0.501, consistent with that bound. A small stochastic batch can score below the population bound; selecting its minimum is the problem.

## Boundaries

- This is one synthetic task and one fixed geometry. The project has not demonstrated generalization across graph sizes, geometry distributions, or real applications.
- Historical variants and hyperparameters were explored before this evaluation. The B3 checkpoint and arm were selected using a fixed six-batch validation set.
- The original frozen Stage 2 configuration was absent. Dimensions were recovered from checkpoint tensors, and inference defaults from the original loader. The released config records that reconstruction explicitly; it is not represented as a recovered original config.
- The new trainer's smoke run verifies execution and selection correspondence. It is intentionally too short to support a new performance claim. The larger schedule and baseline trainer are provided, but a new multi-seed study has not been run for this release candidate.
- Dense tensor kernels implement both sparse graphs and spikes. No speedup, energy saving, physical Loihi run, or biological connectome recovery is established.

See [research history](research-history.md) for later branches and why their claims are kept separate.
