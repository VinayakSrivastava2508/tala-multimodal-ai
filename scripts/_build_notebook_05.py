"""One-off generator for notebooks/05_multimodal_rag.ipynb (Day 3B).
Run once to (re)build the notebook skeleton, then execute with nbconvert."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# 1. Objective and scope
md("""# 05 — Multimodal RAG for TALA Claim-Experience Divergence (Day 3B)

## 1. Objective and scope

**Primary objective:** determine whether multimodal evidence (text, image, video, reference)
reveals divergence between TALA's official quality/responsibility claims and externally
observable customer experience -- this notebook demonstrates the retrieval-augmented-generation
(RAG) system built to let a person investigate that question interactively, grounded in real,
cited evidence.

**Explicitly in scope:** claim-experience divergence investigation, open analytical questions,
modality-aware retrieval, Gemini grounding with citation validation, automated (non-human-label)
evaluation.

**Explicitly out of scope:** engagement prediction, manual/human-coder labelling, protected-
characteristic inference from images, any fallback generator or backup vector store.

Full architecture: `docs/multimodal_rag_architecture.md`. Full evaluation narrative:
`docs/multimodal_rag_evaluation.md`.""")

code("""import sys
from pathlib import Path

PROJECT_ROOT = Path().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
pd.set_option("display.max_colwidth", 100)

from src.rag import orchestrator, query_router, rank_fusion, retriever
from src.rag.chroma_store import get_client, get_all_metadata
from src.rag.schemas import COLLECTION_CLAIMS, COLLECTION_TEXT_EVIDENCE, COLLECTION_VISUAL_EVIDENCE, CHROMA_PERSIST_DIR

TABLES = PROJECT_ROOT / "outputs" / "tables"
client = get_client()
print(f"Connected to local persistent ChromaDB at {CHROMA_PERSIST_DIR.relative_to(PROJECT_ROOT)}")""")

# 2. Corpus composition
md("""## 2. Corpus composition

Built exclusively from existing, already-verified Day 1-3A processed datasets -- nothing here
is freshly scraped. See `docs/multimodal_rag_architecture.md` for the full source table and the
evidence-strength/verification gates applied.""")

code("""collection_summary = pd.read_csv(TABLES / "rag_collection_summary.csv")
display(collection_summary)

corpus_coverage = pd.read_csv(TABLES / "rag_corpus_coverage.csv")
display(corpus_coverage[corpus_coverage["dimension"] == "source_type"])""")

md("""**Documented gap, not silently dropped:** `data/processed/reference_images.csv` (584 rows)
has no downloaded local asset for any row, so none are CLIP-embedded into `tala_visual_evidence`
-- they remain a citation-only catalogue, not indexed visual evidence.""")

# 3. Chroma collections
md("""## 3. Chroma collections

Three persistent collections at `data/vector_db/chroma/`, cosine distance, deterministic IDs
(`claim::<id>`, `text::<source_type>::<raw_id>`, `visual::<modality>::<raw_id>`) so the index
builder is idempotent on rerun.""")

code("""for name in (COLLECTION_CLAIMS, COLLECTION_TEXT_EVIDENCE, COLLECTION_VISUAL_EVIDENCE):
    collection = client.get_collection(name)
    print(f"{name}: {collection.count()} rows")

modality_coverage = pd.read_csv(TABLES / "rag_modality_coverage.csv")
display(modality_coverage)""")

# 4. Modality routing
md("""## 4. Modality routing

Deterministic, inspectable, reuses the authoritative Day 3A visual-groundability rule
(`configs/fusion_rules.yaml`) -- no LLM secretly decides what gets searched.""")

code("""for category in ("materials", "packaging", "labour", "emissions", "manufacturing"):
    modalities = query_router.route_modalities_for_claim(category)
    print(f"{category:15s} -> {modalities}")""")

code("""for question in [
    "What evidence challenges TALA's durability claims?",
    "Which responsibility claims lack independent validation?",
    "How does TALA's creator partnership intent differ from competitors?",
]:
    intents = query_router.classify_intents(question)
    modalities = query_router.route_modalities_for_intents(intents)
    print(f"Q: {question}\\n  intents={intents}\\n  modalities={modalities}\\n")""")

routing_eval_md = """**Modality-routing accuracy** (`outputs/tables/rag_query_routing_evaluation.csv`):
every one of the 37 real claims is checked against the authoritative groundability config."""
md(routing_eval_md)

code("""routing_eval = pd.read_csv(TABLES / "rag_query_routing_evaluation.csv")
print(f"{routing_eval['routing_correct'].sum()}/{len(routing_eval)} claims routed correctly")
display(routing_eval.head(10))""")

# 5. Retrieval pipeline
md("""## 5. Retrieval pipeline

Text retrieval (sentence-transformers `all-MiniLM-L6-v2`) and visual retrieval (CLIP
`openai/clip-vit-base-patch32` text-tower query against CLIP image embeddings), plus direct
Day 3A claim-evidence links looked up by metadata (not similarity).""")

code("""text_results = retriever.retrieve_text_evidence(client, "durability of the fabric after washing", n_results=5)
for r in text_results:
    print(r["id"], round(r["similarity"], 3), r["metadata"]["source_type"])""")

code("""visual_results = retriever.retrieve_visual_evidence(client, "a person wearing activewear leggings", n_results=5)
for r in visual_results:
    print(r["id"], round(r["similarity"], 3), r["metadata"]["modality"], r["metadata"].get("video_role", ""))""")

md("""### Video evidence is genuinely temporal, not one thumbnail

Every video result carries video identity, a representative frame position (opening/interval/
quarter/half/three_quarter/closing), an exact timestamp, video role, and is drawn only from the
8 genuinely processed TALA videos (the other 61 discovered videos are metadata-only leads and
are never treated as video evidence).""")

code("""video_frames = [r for r in visual_results if r["metadata"]["modality"] == "video_frame"]
if video_frames:
    display(pd.DataFrame([r["metadata"] for r in video_frames])[["video_id", "opening_middle_closing_position", "timestamp_seconds", "video_role"]])
else:
    print("No video_frame results in this particular sample query -- see the text+image+video demo case below for a guaranteed example.")""")

# 6. Rank fusion
md("""## 6. Rank fusion

Reciprocal-rank fusion (k=60) across independently-retrieved lists, plus deterministic bonuses
for direct claim-evidence links, evidence strength, and source independence. NLI is never used
as the relevance/stance authority -- Day 3A already established it is auxiliary-only.""")

code("""result = orchestrator.investigate_claim("c_OC_0003_01", n_results=8, generate=False)
trace_rows = [{
    "id": c["id"], "collection": c["collection"], "final_rank": c.get("final_rank"),
    "final_score": c.get("final_score"), "rrf_contribution": c["trace"]["rank_fusion_contribution"],
    "direct_link_bonus": c["trace"].get("direct_link_bonus"), "exclusion_reason": c["trace"]["exclusion_reason"],
} for c in result["fused_candidates"]]
display(pd.DataFrame(trace_rows))""")

# 7. Gemini grounding
md("""## 7. Gemini grounding

`google-genai` is the only generation path. The prompt contains only the question, the selected
claim and its existing Day 3A fusion label (never re-decided by Gemini), the retrieved evidence
with explicit evidence IDs, and grounding instructions. If `GEMINI_API_KEY`/`GEMINI_MODEL` are
missing or the API call fails, generation stops -- there is no fallback.""")

code("""from src.rag.gemini_generator import GeminiConfigError, GeminiGenerationError
try:
    demo = orchestrator.investigate_claim("c_OC_0003_01", n_results=8, generate=True)
    print("MODEL USED:", demo["generation"]["_model_used"])
    print("\\nCONCISE ANSWER:\\n", demo["generation"]["concise_answer"])
    print("\\nCITATION VALIDATION PASSED:", demo["validation"]["passed"])
except (GeminiConfigError, GeminiGenerationError) as exc:
    print(f"Gemini call did not succeed in this run (shown honestly, no fallback substituted): {exc}")""")

# 8. Citation validation
md("""## 8. Citation validation

Every Gemini response is checked automatically: cited IDs must exist in the retrieved context,
the production fusion label must be preserved, self-reported evidence must never be called
independent, and the response must never claim to have inspected evidence beyond what was
supplied (e.g. a full video when only sampled frames were given).""")

code("""citation_eval = pd.read_csv(TABLES / "rag_citation_validation.csv")
display(citation_eval)""")

# 9. Automated evaluation
md("""## 9. Automated evaluation (no human-label evaluation)

Every metric below is derived from existing structural relationships (claim-evidence links,
claim categories, modality eligibility, stance, production fusion labels) -- no new manual-
label CSV was created, no coder_1/coder_2/adjudication workflow exists, and Gemini is never
used to grade its own answers.""")

code("""retrieval_eval = pd.read_csv(TABLES / "rag_retrieval_evaluation.csv")
display(retrieval_eval)""")

code("""latency = pd.read_csv(TABLES / "rag_latency_summary.csv")
display(latency)""")

# 10. Demonstration cases
md("""## 10. Demonstration cases

At least one text+image+video claim, one mixed-label claim explanation, and one insufficient-
evidence explanation.""")

code("""demo_cases = pd.read_csv(TABLES / "rag_demo_cases.csv")
display(demo_cases)""")

# 11. Limitations
md("""## 11. Limitations

- `reference_images.csv`'s 584 rows have no downloaded local asset and are excluded from visual
  evidence -- a real, documented corpus gap, not a Day 3B omission.
- Video evidence is genuinely temporal (sampled frames + motion/scene features) but is **not**
  full raw-video understanding -- the prompt and validator both enforce this distinction.
- Official reference/claim evidence **may be self-reported**; the system labels this explicitly.
- Absence of independent evidence does not prove a claim false -- `insufficient_evidence` claims
  are surfaced as evidence gaps, never as negative conclusions.
- Automated evaluation measures structural correctness (recall, routing accuracy, citation
  validity) -- it does not substitute for human semantic judgement of explanation quality, and
  human-label evaluation was deliberately excluded from this project's scope throughout.
- Live Gemini demo-case generation is subject to the API's free-tier rate/quota limits; a 429
  or 503 response is shown honestly (see `rag_demo_cases.csv`'s `api_outcome` column) rather
  than masked with a substitute answer.""")

# 12. Final GO/NO-GO
md("## 12. Final GO/NO-GO")

code("""go_no_go = pd.read_csv(TABLES / "rag_go_no_go.csv")
display(go_no_go)

print("=" * 70)
print("NOTEBOOK 05 COMPLETE -- Day 3B Multimodal RAG")
print("=" * 70)
for _, row in go_no_go.iterrows():
    print(f"  {row['gate']:<45} {row['status']}")""")

nb["cells"] = cells
with open("notebooks/05_multimodal_rag.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook written.")
