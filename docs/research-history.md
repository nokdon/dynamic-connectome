# Research history and release scope

This release presents the development chain from learned graph structure to hybrid temporal computation. It is a curated implementation and evidence set, not a chronological dump of every notebook, report, or experiment directory.

| Phase | Question | Retained outcome |
|---|---|---|
| Early JAX routing notebooks | Can a neural model choose a useful hard communication graph? | Two-slot STE routing, graph objectives and budget controls. |
| Stage 1 → Stage 2 | Does the extracted graph support a target-local computation? | Five learned graphs and dense/sparse controls, with fresh checkpoint evaluation. |
| B3 continuation | Can a frozen graph support a recurrent membrane/spike model? | Executable hybrid model and one selected checkpoint; no isolated spike-efficiency claim. |
| Native spiking / teacher release | Can later native models continue after teacher release? | A separate historical multi-seed campaign, described below. |
| Fixed-point and Lava work | Which execution semantics survive a deployment-oriented translation? | Numerical debugging lessons; full hardware equivalence remains unestablished. |
| Physical sensing and telemetry | Does the approach help a real application? | No validated application or energy advantage in the inspected evidence. |

## Native follow-up

The later `native_k2_teacher_release_replication_24seed_01` campaign contains 84 completed training rows with corresponding checkpoints: 24 each for A0, A1 and A3, plus 12 A3_LONG runs. A3's stored selected-validation median is about 0.269. Its code implements recurrent spiking computation and selects within a teacher-free tail.

The [sanitized campaign table](../results/historical_native_campaign.csv) preserves the numerical rows and checkpoint-presence check. [Evidence fingerprints](../provenance/historical_evidence.json) identify the source table and the inspected code paths for this history.

This is a different study, including a different 48-edge graph. It is not pooled with the original 44-edge results. The campaign has not been independently retrained or newly tested for this release, and its large dependency chain is not bundled. It is a useful candidate for a future focused extension after portable extraction and fresh evaluation.

## What the Lava work establishes

Early full-graph simulation checks failed. A later decoder repair reports a full-equivalence success label, but the inspected runner feeds saved reference features into a decoder. That supports a narrower decoder-level comparison; it does not rerun raw input through the complete connected graph. A fresh end-to-end test is required before claiming complete equivalence. The inspected Lava work is CPU simulation, not execution on physical Loihi hardware. Fixed-point experiments also retain some spike drift.

## Experiments excluded from the headline

One P00 result path labeled as a membrane/spiking decoder actually constructs supplied route-witness features and solves least squares. It therefore cannot substantiate neural route discovery. The release omits that executable branch and its earlier headline.

The physical-sensing branch pools sessions before fitting normalization, uses overlapping windows, and lacks important simple controls. Its rough energy estimate uses two power samples around a short inference; the reported ratio is about 1.013 and fails the branch's own 2× target. The telemetry smoke records negative test R² for all three stored arms. Neither supports an application or hardware-efficiency claim here.

The useful methodological lesson is to keep graph selection, message computation, readout capacity, checkpoint selection, and backend execution separate in the evaluation. The present release makes those distinctions explicit and testable.
