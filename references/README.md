# References

[`references.bib`](references.bib) is the canonical bibliography for the repository and manuscript. `paper/references.bib` is a generated/synchronized copy retained so the manuscript source tree remains self-contained.

Synchronize the paper copy with:

```bash
python scripts/sync_bibliography.py
```

CI verifies synchronization with `--check`.

This public directory intentionally contains bibliographic metadata rather than publisher PDFs. External articles should be referenced by DOI, stable URL, or another persistent identifier. A full-text file may be committed only when its redistribution license has been verified explicitly (for example, an open-access version or author manuscript whose license permits repository redistribution).

The three previously committed reference PDFs have been removed from the current tree. Removing them from historical commits requires a separate authenticated history rewrite with `git filter-repo`; deleting them from the current branch does not erase old Git objects.
