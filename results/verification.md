# Local verification

Verified on 2026-09-16, Linux x86_64, Python 3.13.7, JAX/JAXLIB 0.9.2, NumPy 2.4.4 and CPU execution. The table records local checks; the current remote status is available in [GitHub Actions](https://github.com/nokdon/dynamic-connectome/actions/workflows/tests.yml).

| Check | Observed outcome |
|---|---|
| Build a wheel with an isolated setuptools build environment | Passed; all 26 model files, the problem/catalog and third-party notice included. |
| Install the wheel and locked dependencies into a fresh virtual environment | Passed; dependency compatibility check passed. |
| Run the installed `dc demo` from outside the checkout | Passed; imports resolved to the fresh environment's `site-packages`. |
| Full pytest suite against the installed wheel | **18 passed in 30.42 s**. Includes complete small core/control training and selected-checkpoint re-evaluation. |
| Re-evaluate all bundled models from the installed wheel | Passed; the same 4,096-example checkpoint table regenerated. |
| Compare all 26 curated checkpoint MSEs with the original implementation's diagnostic | Maximum absolute difference **0.0** in this environment. |
| Compare Stage 1 and B3 losses and parameter gradients with original kernels on a fixed small batch | Loss and maximum gradient differences **0.0** for both. |
| Regenerate and visually inspect both figures | Passed; SVG and PNG artifacts included. |
| Focused static Python checks | Passed (`ruff` 0.11.13, E9/F63/F7/F82). |
| Inspect proposed release files for personal absolute paths and common credential/private-key patterns | No matches found in the curated file set. This is a targeted scan, not an exhaustive security certification. |

The full research schedule and original multi-seed training campaign were **not** rerun. Smoke training verifies function, selection and serialization; its performance is not substituted for the historical checkpoint table. Other backends and operating systems remain unverified.

The bundled artifact hashes verify the shipped arrays against the catalog. [release_manifest.json](../provenance/release_manifest.json) fingerprints the candidate files, excluding itself and ignored local build/output directories. [Source provenance](../provenance/source-origin.md) reflects the owner's clarification that the early implementation was transferred from their own local repository. This documentation correction does not change model code, parameters, or evaluation results.
