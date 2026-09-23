"""Page 6: Methodology & Limitations."""

from __future__ import annotations

import streamlit as st

from app import data_loader as dl

LIMITATIONS = [
    "Public data only -- no internal support tickets or post-purchase surveys.",
    "Automated labels without human adjudication.",
    "Creator sample dominated by YouTube.",
    "No Instagram creator evidence.",
    "Thin video/image evidence relative to text.",
    "Official evidence may be self-reported.",
    "No independent correctness benchmark.",
    "Sampled frames are not complete video understanding.",
    "Insufficient evidence does not mean false.",
    "Moat hypotheses are not proven.",
]

KEY_TERMS = [
    ("Aligned", "The evidence corpus corroborates the claim. This does not mean the claim is independently proven."),
    ("Mixed", "The evidence corpus contains both supporting and challenging material. This does not automatically mean greenwashing."),
    ("Insufficient evidence", "The corpus does not contain enough evidence to assess the claim either way. This does not mean the claim is false."),
    ("Production label", "The system's evidence classification for a claim. It is not a legal or factual verdict."),
    ("Self-reported", "The evidence comes from the brand itself (its own claims, reference documents, video or images)."),
    ("Independent evidence", "Evidence not published by the brand itself. Independence does not automatically establish correctness."),
    ("Weight-sensitive", "The classification can change under reasonable changes to evidence weights."),
    ("Presentation restriction", "How a finding may be communicated based on evidence strength and risk."),
]

DOWNLOADABLE_TABLES = [
    "analytical_synthesis_claim_master", "claim_label_summary", "mixed_claim_theme_summary",
    "creator_partnership_mix", "creator_content_intent_mix", "ai_governance_go_no_go",
    "ai_governance_risk_register", "ai_governance_implementation_roadmap",
]


def render() -> None:
    st.title("Methodology & Limitations")

    st.subheader("Business question")
    st.write(
        "Diagnose divergence between TALA's official quality/responsibility claims and customer "
        "experience by integrating official and customer text, product/catalog and permitted UGC "
        "images, video frames/temporal features/transcripts where available, and enterprise "
        "reference documents; secondarily, compare creator-partnership and official-platform "
        "strategy against Adanola, Girlfriend Collective and Oner Active, descriptively."
    )

    st.subheader("Data sources")
    st.write(
        "Official claims corpus, customer experience corpus (reviews/press), creator strategy "
        "corpus, product/catalog images, sampled video frames, and multimodal reference documents "
        "(impact/responsibility reports, certifications, material specs, care guidance, return "
        "policies) -- all collected from free/public sources per the project charter."
    )

    st.subheader("Pipelines")
    st.markdown(
        "- **Text:** embeddings, TF-IDF, sentiment and topic features (`src/text_features.py`).\n"
        "- **Image:** CLIP/ViT feature extraction and colour-palette analysis (`src/image_features.py`).\n"
        "- **Video:** frame sampling, motion/scene-change features, frame-sequence embeddings (`src/video_features.py`).\n"
        "- **Reference package:** structured extraction from official reports, certifications and policy pages.\n"
        "- **Fusion:** early/late/hybrid claim-evidence integration (`src/fusion_models.py`), directed at "
        "claim-experience evidence integration, not engagement prediction.\n"
        "- **RAG:** ChromaDB retrieval + rank fusion + Gemini generation with citation validation (`src/rag/`).\n"
        "- **Governance:** control matrix, risk register and GO/PARTIAL/NO-GO gating "
        "(`scripts/build_governance_framework.py`)."
    )

    st.subheader("Key terms used in this cockpit")
    for term, definition in KEY_TERMS:
        st.markdown(f"- **{term}:** {definition}")

    st.subheader("Core limitations")
    for item in LIMITATIONS:
        st.markdown(f"- {item}")

    st.subheader("Reference downloads")
    for name in DOWNLOADABLE_TABLES:
        try:
            df = dl.load_table(name)
        except dl.MissingTableError:
            continue
        st.download_button(f"Download {name}.csv", df.to_csv(index=False).encode("utf-8"), file_name=f"{name}.csv")
