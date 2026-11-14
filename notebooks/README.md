# scikit-bio: five questions about representation

These independent, executed SQL walkthroughs use explicitly synthetic inputs and the real released scikit-bio worker. They cover **46 of 90 distinct public SQL names (51.1%)**, exceeding the 45-name target while keeping different analytical questions separate.

| Notebook | Question | Parameter experiment |
| --- | --- | --- |
| [01 · Same proportions, different total counts](01_same_proportions_different_counts.ipynb) | Does uniform count scaling change diversity or composition? | `DEPTH_MULTIPLIER`: 3 → 8 |
| [02 · Orientation before composition](02_orientation_before_composition.ipynb) | Can strand conventions change a word-profile comparison? | `KMER_K`: 3 → 2 |
| [03 · A base change is not a protein change](03_comparison_depends_on_representation.ipynb) | What do positional, aligned and translated comparisons retain? | `ALIGNMENT_MODE`: global → local |
| [04 · Coordinates do not recover totals](04_coordinates_do_not_recover_totals.ipynb) | Does changing the log-ratio reference change the composition? | `REFERENCE_FEATURE`: a → c |
| [05 · Tree scale is not topology](05_tree_scale_is_not_topology.ipynb) | Which tree comparisons detect uniform length scaling? | `LENGTH_SCALE`: 1 → 2 |

Notebook 01 preserves the original count-scale narrative and SQL reconstructions. Each book contains its own input, visible analytical SQL, interpretations, two SVG figures, checks and experiment. There is no organism identification, biological inference, medical interpretation or claim that these authored inputs model sampling uncertainty.

## Run locally or in Colab

Install [uv](https://docs.astral.sh/uv/) and Git. From `notebooks/`:

```sh
uv sync --locked
uv run --locked vgi-notebooks 01_same_proportions_different_counts.ipynb
uv run --locked vgi-notebooks 02_orientation_before_composition.ipynb
uv run --locked vgi-notebooks 03_comparison_depends_on_representation.ipynb
uv run --locked vgi-notebooks 04_coordinates_do_not_recover_totals.ipynb
uv run --locked vgi-notebooks 05_tree_scale_is_not_topology.ipynb
```

An editor can use `notebooks/.venv/bin/python` and `notebooks/` as its working directory. Restart/run all after changing parameters. The Colab badge opens the same notebook; its brief bootstrap installs only the shared **base** package. Local publication uses `[runner]`. These examples have no companion files to download or repository discovery logic.

Every notebook visibly launches the release in a separate Python 3.13 process:

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'skbio' (TYPE vgi,
 LOCATION 'uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-scikit-bio@v0.3.0 vgi-scikit-bio');
```

First use downloads the worker dependencies and community VGI extension. The release owns its dependency constraints; the old notebook constraints file and extra transitive pins are removed. The isolated worker's observed packages include scikit-bio 0.7.4, pandas 3.1.1 and vgi-python 0.51.1; exact observations are in [coverage/runtime-versions.json](coverage/runtime-versions.json). The worker source is pinned, but its declared dependency ranges permit later transitive versions. Local notebook dependencies are locked. scikit-bio, SciPy and the VGI Python SDK stay outside the notebook kernel.

Shared connection setup uses an empty temporary SQL secret directory with persistent secrets disabled. These computational examples create no SQL secrets, read no private accounts and send no messages. Cleanup closes each connection.

## Experiments preserve successful defaults

Run overrides into separate destinations:

```sh
uv run --locked vgi-notebooks 01_same_proportions_different_counts.ipynb \
  --parameters '{"DEPTH_MULTIPLIER":8}' --output-dir validation/counts-8
uv run --locked vgi-notebooks 02_orientation_before_composition.ipynb \
  --parameters '{"KMER_K":2}' --output-dir validation/words-2
uv run --locked vgi-notebooks 03_comparison_depends_on_representation.ipynb \
  --parameters '{"ALIGNMENT_MODE":"local"}' --output-dir validation/local
uv run --locked vgi-notebooks 04_coordinates_do_not_recover_totals.ipynb \
  --parameters '{"REFERENCE_FEATURE":"c"}' --output-dir validation/reference-c
uv run --locked vgi-notebooks 05_tree_scale_is_not_topology.ipynb \
  --parameters '{"LENGTH_SCALE":2.0}' --output-dir validation/scale-2
```

The runner rejects unknown parameters and overrides that would overwrite successful defaults. Each notebook validates its authored parameter domain before ATTACH. Inputs remain fixed in every experiment except the explicitly selected count/length multiplier; the notebooks state the outputs expected to remain invariant.

## What the comparisons mean

- **Counts and composition:** Uniform rescaling is an arithmetic control, not deeper biological sampling. Closure discards total size. Alpha aggregates receive integer counts; the complete raw and closed Bray–Curtis matrices are reconstructed in SQL.
- **Sequence representations:** Missing and malformed strings stay in the quality audit; ambiguity is not resolved into invented bases. Gap removal and reverse-complement canonicalization are explicit choices. All word-profile pairs, including zero-overlap pairs, remain in the result.
- **Alignment and translation:** Hamming comparisons are unavailable for unequal lengths. The worker's legacy global nucleotide defaults do not penalize terminal gaps; local mode can select a shorter span and changes nucleotide match/mismatch scoring from +1/−2 to +2/−3, so cross-mode score increases are not improvements under one objective. Protein and nucleotide scores have different scoring units and are not probabilities. Translation uses a stipulated frame; six-frame enumeration does not select a gene.
- **Log-ratio coordinates:** Positive complete inputs avoid a separate zero-treatment decision. CLR, ILR and ALR inverses recover proportions rather than counts. SQL verifies all 36 reconstructed parts and the sample variance of every feature log ratio. Feature variance is descriptive, not a significance test.
- **Trees:** Serialized-root height is root-dependent. The actual `cophenetic_distance` metric is scikit-bio's unit correlation distance `(1-r)/2`, so uniform positive scaling can give zero even when absolute lengths differ. This is more specific than the broad catalog wording. Controlled identical, scaled and topology-swapped trees distinguish that behavior from weighted RF and split RF. Neither a zero comparison nor algorithm agreement validates a phylogeny.

## Coverage and execution evidence

[coverage.json](coverage.json) maps each counted name to visible successful SQL, displayed results and reviewed interpretation. Full native functions/tables/views exports are retained, including empty table/view exports. The catalog denominator includes all five schemas and all 90 names; overloads count once. The remaining 44 names stay explicitly uncovered.

The release catalog was captured from v0.3.0 / `eea42cf85f63a79b06be0e08880fe3e9c3824daa`; the current catalog was independently captured from public main `bd12ddc16320c73980aad9729e8ec58842d79faa`. Their named interfaces agree. Changes between those worker revisions add metadata/composed examples. Default execution receipts identify the released runtime: reviewed coverage is 46/90 for each interface, with exact runtime coverage 46 for release and 0 for current. The notebook does not relabel a release run as an exact-current run.

All five defaults and notebook-only detached experiments execute through published Haybarn 2.0.0rc11. One fresh Python 3.12 kernel with pandas, Matplotlib, IPython, ipykernel and jupyter-client already imported exercises the shared bootstrap and complete alignment notebook, using a cold worker cache and preserving those versions. No runner, scikit-bio, SciPy or VGI Python SDK is installed in that kernel. Hosted Colab was not manually executed.

Charts use native SVG output plus committed image links for GitHub. Review PNGs, experiment copies and logs are ignored. The [shared standard](https://github.com/Query-farm/vgi-example-notebooks/blob/main/docs/standard.md) supplies presentation, branding and runner behavior. Existing [green source CI at the unchanged worker baseline](https://github.com/Query-farm/vgi-scikit-bio/actions/runs/43582965784) is reused; this notebook-only change does not rebuild the worker, repeat source suites or publish containers.
