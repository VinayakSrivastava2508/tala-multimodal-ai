"""Page 5: Governance & Roadmap."""

from __future__ import annotations

import streamlit as st

from app import data_loader as dl
from app.components import charts

TOP_RISK_SCENARIOS = [
    "defamation", "reputational", "sampl", "representation", "staleness",
    "creator misrepresentation", "greenwash",
]

REVIEW_POLICY_TEXT = (
    "Mixed findings require domain-owner review. Legal review is additionally required "
    "before external communication or where regulatory, reputational or defamation risk is material."
)


def render() -> None:
    st.title("Governance & Roadmap")

    st.subheader("11.1 Deployment readiness")
    go_no_go = dl.load_go_no_go()
    contexts = go_no_go["deployment_context"].unique().tolist()
    context_labels = {
        "academic_demonstration_readiness": "Academic demonstration",
        "controlled_internal_pilot_readiness": "Controlled internal pilot",
        "external_customer_facing_deployment_readiness": "External/customer-facing deployment",
    }
    cols = st.columns(len(contexts))
    for col, ctx in zip(cols, contexts):
        with col:
            st.markdown(f"**{context_labels.get(ctx, ctx)}**")
            go_n, partial_n, nogo_n = charts.gate_status_bar(go_no_go, ctx, context_labels.get(ctx, ctx))
            verdict = "GO" if nogo_n == 0 and partial_n == 0 else ("NO-GO" if nogo_n > 0 else "PARTIAL")
            st.metric("Overall verdict", verdict)
            st.caption(f"GO: {go_n} | PARTIAL: {partial_n} | NO-GO: {nogo_n}")
            blockers = go_no_go[(go_no_go["deployment_context"] == ctx) & (go_no_go["status"] != "GO")]
            if not blockers.empty:
                st.caption("Main blocking gaps: " + "; ".join(blockers["gap_or_next_step"].dropna().head(3).tolist()))

    st.divider()
    st.subheader("11.2 Risk heatmap")
    risks = dl.load_risk_register()
    charts.risk_heatmap(risks, "residual_likelihood_1_5", "residual_impact_1_5", "risk_id", "Residual risk: likelihood x impact")
    pattern = "|".join(TOP_RISK_SCENARIOS)
    top_risks = risks[risks["risk_scenario"].str.lower().str.contains(pattern, na=False) | risks["risk_domain"].str.lower().str.contains(pattern, na=False)]
    st.markdown("**Top residual risks**")
    st.dataframe(
        top_risks[["risk_id", "risk_domain", "risk_scenario", "residual_risk_level", "risk_owner"]],
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.subheader("11.3 Control maturity")
    controls = dl.load_control_matrix()
    status_counts = controls["control_status"].value_counts()
    cols = st.columns(len(status_counts))
    for col, (status, n) in zip(cols, status_counts.items()):
        col.metric(status, int(n))
    charts.grouped_bar(
        controls.groupby(["control_type", "control_status"]).size().reset_index(name="n"),
        "control_type", "n", "control_status", "Control maturity by type (Preventive / Detective / Corrective)",
    )

    st.divider()
    st.subheader("11.4 Claim decision policy")
    st.caption("Select a row to view its meaning, reviewer and permitted use.")
    policy = dl.load_claim_decision_policy()
    for _, p in policy.iterrows():
        # Only "mixed" is expanded by default -- it is the highest-risk,
        # most-consulted policy row; the rest stay collapsed so the page
        # doesn't overwhelm a first-time executive reader.
        with st.expander(p["decision_label"], expanded=(str(p["decision_label"]).strip().lower() == "mixed")):
            st.write(f"**Meaning:** {p['meaning']}")
            st.write(f"**What it does not mean:** {p['what_it_does_not_mean']}")
            st.write(f"**Permitted internal use:** {p['permitted_internal_use']}")
            st.write(f"**External communication status:** {p['external_communication_status']}")
            st.write(f"**Required reviewer:** {p['required_reviewer']}")
    st.info(REVIEW_POLICY_TEXT)

    st.divider()
    st.subheader("11.5 ESG priorities")
    gaps = dl.load_evidence_gap_by_category()
    labour = gaps[gaps["claim_category"].str.lower() == "labour"]
    if not labour.empty:
        st.write(f"Labour evidence gap: {int(labour.iloc[0]['total_claims'])} claim(s), dominant missing evidence type: {labour.iloc[0]['dominant_missing_evidence_type']}")
    esg = dl.load_esg_claim_assurance_matrix()
    if esg is not None:
        st.dataframe(
            esg[["claim_category", "current_evidence_gap", "recommended_action"]],
            use_container_width=True, hide_index=True,
        )
    st.caption("Evidence gaps documented above reflect corpus coverage limits, not a finding of greenwashing.")

    st.divider()
    st.subheader("11.6 Implementation roadmap")
    st.caption("Select a phase to view its owner, priority and completion criteria.")
    roadmap = dl.load_implementation_roadmap()
    display_cols = [c for c in ["initiative", "owner", "priority", "status", "completion_criterion"] if c in roadmap.columns]
    for phase in roadmap["phase"].unique():
        # Only the current phase is expanded by default -- everything else
        # stays collapsed rather than default-expanding every phase.
        with st.expander(phase, expanded=("current" in str(phase).lower())):
            phase_df = roadmap[roadmap["phase"] == phase]
            st.dataframe(phase_df[display_cols], use_container_width=True, hide_index=True)
