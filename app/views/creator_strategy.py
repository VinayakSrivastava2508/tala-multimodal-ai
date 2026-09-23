"""Page 3: Creator Strategy -- descriptive cross-brand comparison, not predictive."""

from __future__ import annotations

import streamlit as st

from app import data_loader as dl
from app.components import charts
from app.components.limitations import notice

BOUNDARY_NOTICE = (
    "Creator comparison is based on a targeted sample dominated by YouTube: approximately "
    "60 YouTube records, one TikTok record and no Instagram creator records. It supports a "
    "cross-brand YouTube creator-strategy comparison combined with an official-channel platform "
    "comparison; it is not a comprehensive cross-platform performance study."
)


def render() -> None:
    st.title("Creator Strategy")
    notice(BOUNDARY_NOTICE)

    mix = dl.load_creator_partnership_mix()
    intent_mix = dl.load_creator_content_intent_mix()
    matrix = dl.load_creator_partnership_intent_matrix()
    platform = dl.load_brand_platform_strategy_comparison()
    interp = dl.load_creator_strategy_interpretation()
    lessons = dl.load_competitor_lessons()
    moats = dl.load_moat_hypotheses()

    st.subheader("Brand sample sizes")
    sizes = mix.groupby("brand")["brand_denominator_n"].first().reset_index()
    cols = st.columns(len(sizes))
    for col, (_, r) in zip(cols, sizes.iterrows()):
        col.metric(r["brand"], f"n = {int(r['brand_denominator_n'])}")

    st.divider()
    st.subheader("9.1 Partnership mix")
    charts.stacked_percentage_bar(
        mix, "brand", "partnership_type", "pct_within_brand", "count",
        "Partnership-type mix by brand (100% stacked, % within brand)",
        caption="Hover for raw n. Denominators vary by brand (n=8 to n=24) -- see sample sizes above.",
    )
    st.dataframe(mix, use_container_width=True, hide_index=True)
    st.markdown(
        "- Oner Active has the highest observed organic share, but only eight records.\n"
        "- TALA has an organic plurality in a larger sample and the most varied observed partnership mix.\n"
        "- Adanola is paid-sponsorship concentrated within the observed sample.\n"
        "- Girlfriend Collective is affiliate concentrated within the observed sample."
    )

    st.subheader("9.2 Content-intent mix")
    charts.stacked_percentage_bar(
        intent_mix, "brand", "content_intent", "pct_within_brand", "count",
        "Content-intent mix by brand (% within brand)",
    )

    st.subheader("9.3 Partnership x intent")
    scope = st.radio("View", ["overall"] + sorted(matrix["brand"].dropna().unique().tolist()), horizontal=True)
    if scope == "overall":
        subset = matrix[matrix["scope"] == "overall"]
    else:
        subset = matrix[matrix["brand"] == scope]
    if subset.empty:
        st.info("No partnership x intent records for this selection.")
    else:
        charts.category_label_heatmap(
            subset, "content_intent", "partnership_type", "count",
            f"Partnership type x content intent ({scope})",
            caption="Cell values are creator-record counts, not performance scores. Association shown here is not causal.",
        )

    st.subheader("9.4 Official-platform comparison")
    st.caption("Evidence source: **Official-channel evidence** (brand-owned platform presence). Shown separately from creator-record evidence above -- denominators are not merged.")
    st.dataframe(platform, use_container_width=True, hide_index=True)

    st.subheader("9.5 Strategic interpretation")
    for _, r in interp.iterrows():
        with st.container(border=True):
            st.markdown(f"**{r['brand']}**")
            st.write(r.get("statement", "-"))
            st.caption(f"Evidence strength: {r.get('evidence_strength', '-')}")
            st.caption(f"Limitation: {r.get('limitation', '-')}")

    st.subheader("9.6 Competitor lessons and moat hypotheses")
    st.markdown("**Competitor lessons**")
    st.dataframe(
        lessons[["comparator_brand", "observed_competitor_practice", "potential_tala_action", "evidence_confidence", "caveat"]],
        use_container_width=True, hide_index=True,
    )
    st.markdown("**Moat hypotheses**")
    for _, m in moats.iterrows():
        with st.container(border=True):
            st.markdown(f"**{m['hypothesis_id']} — {m['brand']}** (status: {m['status']})")
            st.write(m["observed_pattern"])
            st.caption(f"Evidence confidence: {m.get('evidence_confidence', '-')}")
            st.caption(f"Counter-evidence: {m.get('counterevidence', '-')}")
    st.caption("No hypothesis above is labelled a proven moat; all are candidates pending further validation.")
