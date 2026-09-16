# Dependency notices

The project owner clarified that the early JAX code originated in their own local repository and was transferred into DC. [Source provenance](provenance/source-origin.md) records this origin and links to the imported-file manifest.

A project-wide license has not yet been selected for this local release candidate.

The package depends on JAX/JAXLIB, NumPy, Optax and NetworkX. Optional tooling uses Matplotlib and pytest. Those packages are installed separately under their respective upstream terms; their source is not vendored here. The full resolved package list is in `requirements-lock.txt`.

No third-party papers, real-world datasets, browser history, chat archives, raw experimental logs, credentials, or private notebook outputs are included. The bundled examples are synthetic numerical artifacts from the DC workspace. Code and artifact paths imported into the candidate are listed in `provenance/import_manifest.json`.
