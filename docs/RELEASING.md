# Release and archival checklist

The manuscript must cite an immutable software release rather than the moving `main` branch.

For a release intended for citation:

1. ensure CI is green on the release commit;
2. ensure `python scripts/sync_bibliography.py --check` passes;
3. regenerate paper numerical assets and compile the manuscript;
4. update `CHANGELOG.md` and the version in `pyproject.toml` and `CITATION.cff`;
5. create an annotated Git tag such as `v0.1.0`;
6. create a GitHub Release from that tag;
7. attach the CI-compiled manuscript PDF and relevant reproducibility artifacts to the release;
8. archive the GitHub release with Zenodo (or an equivalent repository) and obtain the version-specific DOI;
9. add the DOI to `CITATION.cff`, `.zenodo.json`, the README, and the manuscript software citation;
10. verify that the DOI resolves to the exact tagged source archive.

No DOI is recorded in the repository until an archive has actually been created. Inventing or pre-allocating a DOI in source metadata would defeat the purpose of immutable citation.
