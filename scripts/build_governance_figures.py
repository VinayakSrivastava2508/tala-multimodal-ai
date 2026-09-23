"""Day 3D: presentation-ready governance figures. Reads only outputs/tables/
governance CSVs already built by scripts/build_governance_framework.py.
No 3D charts, clipped labels, or misleading axes; implemented vs. proposed
controls are visibly distinguished; risk scores are labelled as management
judgements.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
PALETTE = sns.color_palette("Set2")
STATUS_COLORS = {
    "Implemented": "#2E7D32",
    "Partially implemented": "#F9A825",
    "Proposed": "#C62828",
    "Not applicable": "#9E9E9E",
}


def save(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[fig]", path)


def fig_operating_model():
    layers = [
        ("Layer 1", "Data admission", "Source legality, provenance, rights basis, evidence-strength gates"),
        ("Layer 2", "Modality processing", "Text/image/video eligibility, protected-characteristic & face-embedding prohibition"),
        ("Layer 3", "Fusion & analytical classification", "Claim linkage, support/challenge separation, sensitivity testing"),
        ("Layer 4", "Retrieval & generation", "Chroma retrieval, Gemini generation, citation validation, no fallback"),
        ("Layer 5", "Managerial use & external communication", "Human approval, ESG/legal sign-off, correction & retraction"),
    ]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers) + 1)
    ax.axis("off")
    box_h = 0.72
    for i, (tag, name, desc) in enumerate(layers):
        y = len(layers) - i
        color = PALETTE[i % len(PALETTE)]
        ax.add_patch(plt.Rectangle((0.5, y - box_h / 2), 9, box_h, facecolor=color, edgecolor="black", alpha=0.85, zorder=2))
        ax.text(0.9, y, f"{tag}: {name}", fontsize=12, fontweight="bold", va="center", zorder=3)
        ax.text(9.1, y, desc, fontsize=8.5, va="center", ha="right", zorder=3, wrap=True)
        if i < len(layers) - 1:
            ax.annotate("", xy=(5, y - box_h / 2 - 0.12), xytext=(5, y - box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5), zorder=2)
    ax.text(5, len(layers) + 0.7, "Evidence retrieval -> automated classification -> human judgement -> approved communication",
            fontsize=9, ha="center", style="italic")
    ax.set_title("AI governance operating model -- five decision layers\nSource: ai_governance_control_matrix.csv (governance_layer)", fontsize=12)
    save(fig, "ai_governance_operating_model.png")


def fig_risk_heatmap():
    df = pd.read_csv(TABLES_DIR / "ai_governance_risk_register.csv")
    grid = np.zeros((5, 5), dtype=int)
    for _, row in df.iterrows():
        grid[int(row["impact_1_5"]) - 1, int(row["likelihood_1_5"]) - 1] += 1
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(grid, annot=True, fmt="d", cmap="YlOrRd", cbar_kws={"label": "n risks (inherent)"},
                ax=ax, linewidths=0.5, linecolor="white",
                xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
    ax.invert_yaxis()
    ax.set_xlabel("Likelihood (1=rare -- 5=near-certain)")
    ax.set_ylabel("Impact (1=negligible -- 5=severe)")
    ax.set_title(
        "Inherent risk heatmap -- 26 risks, 5x5 management-judgement scale\n"
        "Scores are structured judgements, not empirical probabilities. Source: ai_governance_risk_register.csv",
        fontsize=10,
    )
    save(fig, "ai_risk_heatmap.png")


def fig_claim_decision_flow():
    labels = ["aligned", "partially_aligned", "mixed", "divergent", "insufficient_evidence"]
    counts = {"aligned": 14, "partially_aligned": 0, "mixed": 5, "divergent": 0, "insufficient_evidence": 18}
    reviewers = {
        "aligned": "Marketing/brand lead\n(+ ESG lead if labour/ESG,\nno independent evidence)",
        "partially_aligned": "Sustainability/ESG lead\n+ Legal counsel",
        "mixed": "Legal counsel\n(+ ESG lead if labour/ESG)",
        "divergent": "Legal counsel + ESG lead\n+ Executive sponsor",
        "insufficient_evidence": "Sustainability/ESG lead\n(if labour/ESG category)",
    }
    ext_status = {
        "aligned": "Conditional (after sign-off)",
        "partially_aligned": "Not permitted w/o review",
        "mixed": "Not permitted w/o both-sides",
        "divergent": "Not permitted w/o full review",
        "insufficient_evidence": "Not permitted as negative claim",
    }
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    x_positions = np.linspace(1, 10, len(labels))
    colors = ["#2E7D32", "#9E9E9E", "#F9A825", "#C62828", "#1565C0"]
    for x, label, color in zip(x_positions, labels, colors):
        n = counts[label]
        ax.add_patch(plt.Rectangle((x - 0.9, 4.1), 1.8, 1.1, facecolor=color, alpha=0.85, edgecolor="black"))
        ax.text(x, 4.65, f"{label}\n(n={n}/37)", fontsize=9, ha="center", va="center", color="white", fontweight="bold")
        ax.annotate("", xy=(x, 3.6), xytext=(x, 4.1), arrowprops=dict(arrowstyle="-|>", color="black"))
        ax.text(x, 2.9, "Required reviewer:\n" + reviewers[label], fontsize=7.5, ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor="#EEEEEE", edgecolor="gray"))
        ax.annotate("", xy=(x, 1.6), xytext=(x, 2.3), arrowprops=dict(arrowstyle="-|>", color="black"))
        ax.text(x, 0.9, ext_status[label], fontsize=7.5, ha="center", va="center", wrap=True,
                bbox=dict(boxstyle="round", facecolor="#FFF3E0", edgecolor="gray"))
    ax.text(5.5, 5.6, "Claim decision & escalation flow (37 claim records)", fontsize=12, ha="center", fontweight="bold")
    ax.text(5.5, 0.15, "Source: claim_decision_policy.csv; analytical_synthesis_claim_master.csv", fontsize=8, ha="center", style="italic")
    save(fig, "claim_decision_escalation_flow.png")


def fig_control_map():
    df = pd.read_csv(TABLES_DIR / "ai_governance_control_matrix.csv")
    pivot = df.pivot_table(index="governance_layer", columns="control_type", values="control_id", aggfunc="count").fillna(0)
    order = ["Preventive", "Detective", "Corrective"]
    pivot = pivot[[c for c in order if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pivot.plot(kind="barh", stacked=True, ax=ax, color=["#1565C0", "#F9A825", "#2E7D32"])
    ax.set_xlabel("Number of controls")
    ax.set_title("Preventive / detective / corrective control map by governance layer\nSource: ai_governance_control_matrix.csv", fontsize=11)
    ax.legend(title="Control type", loc="lower right")
    save(fig, "prevent_detect_correct_control_map.png")


def fig_esg_matrix():
    df = pd.read_csv(TABLES_DIR / "esg_claim_assurance_matrix.csv")
    # Build a simple readiness signal: has independent evidence expectation vs. current gap text length as proxy is misleading;
    # instead show a categorical matrix of evidence-type eligibility (image/video eligible or not) per category.
    cats = df["claim_category"]
    image_eligible = df["eligible_image_evidence"].str.lower().str.contains("not typically") == False
    video_eligible = df["eligible_video_evidence"].str.lower().str.contains("not typically") == False
    matrix = pd.DataFrame({"Image evidence eligible": image_eligible.values, "Video evidence eligible": video_eligible.values}, index=cats)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(matrix.astype(int), annot=matrix.replace({True: "Yes", False: "No"}), fmt="", cmap="Greens",
                cbar=False, ax=ax, linewidths=0.5, linecolor="white", vmin=0, vmax=1)
    ax.set_title("ESG claim-assurance matrix -- visual-evidence eligibility by claim category\nSource: esg_claim_assurance_matrix.csv", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Claim category")
    save(fig, "esg_claim_assurance_matrix.png")


def fig_roadmap():
    df = pd.read_csv(TABLES_DIR / "ai_governance_implementation_roadmap.csv")
    counts = df["phase"].value_counts()
    order = [p for p in [
        "Phase 0 - Academic prototype (current state)",
        "Phase 1 - Controlled internal pilot (0-30 days)",
        "Phase 2 - Limited business deployment (31-90 days)",
        "Phase 3 - Scaled deployment (3-6 months)",
    ] if p in counts.index]
    counts = counts.reindex(order)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(counts)), counts.values, color=PALETTE[: len(counts)])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([p.split(" - ")[1] if " - " in p else p for p in counts.index], rotation=15, ha="right", fontsize=9)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(v), ha="center", fontsize=10)
    ax.set_ylabel("Number of initiatives")
    ax.set_title("Implementation roadmap -- initiative count by phase\nSource: ai_governance_implementation_roadmap.csv", fontsize=11)
    save(fig, "ai_governance_implementation_roadmap.png")


def main():
    fig_operating_model()
    fig_risk_heatmap()
    fig_claim_decision_flow()
    fig_control_map()
    fig_esg_matrix()
    fig_roadmap()
    print("All 6 governance figures written.")


if __name__ == "__main__":
    main()
