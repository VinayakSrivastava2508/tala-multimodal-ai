# Notebooks excluded from the submission sequence

Two files in `notebooks/` are early, unexecuted scaffold stubs superseded by the numbered sequence documented in `notebook_manifest.csv`, and are **excluded** from the submission ZIP:

- `notebooks/03_creator_strategy_and_claim_divergence.ipynb` -- 26 cells, entirely commented-out placeholder code, 0 cells executed. Superseded by `03_video_pipeline_and_multimodal_evidence.ipynb` and `06_analytical_synthesis_and_creator_strategy.ipynb`.
- `notebooks/04_multimodal_rag_and_evaluation.ipynb` -- 25 cells, entirely commented-out placeholder code, 0 cells executed. Superseded by `04_primary_multimodal_fusion.ipynb` and `05_multimodal_rag.ipynb`.

Both are verified (via `scripts/_build_notebook_03.py` / `_build_notebook_04.py`, which generate the current, real, executed notebooks with the same 03/04 numbering) to be leftover Day-1-era scaffolding, not part of the final analytical pipeline. They are kept in the working repository (not deleted) but are not part of this submission's notebook sequence, so the ZIP is not misrepresented as containing two different '03' or '04' notebooks with conflicting content.
