"""
MedCollab — Streamlit Demo Application

Premium dark-themed medical diagnosis dashboard with:
- Patient case input form (sidebar)
- Real-time agent activity visualization
- IBIS debate cards
- Causal chain graph rendering
- Final diagnosis report
"""

from __future__ import annotations
import json
import sys
import logging
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.patient import PatientCase, Symptom, LabResult
from src.graph.workflow import run_diagnosis
from src.utils.visualization import render_causal_chain

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MedCollab — Multi-Agent Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { background-color: #0e1117; }
    .stApp { font-family: 'Inter', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2rem;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }

    /* Agent Cards */
    .agent-card {
        background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .agent-card:hover {
        border-color: #e94560;
        box-shadow: 0 0 20px rgba(233, 69, 96, 0.15);
    }
    .agent-card h3 {
        color: #e94560;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    .agent-card p {
        color: rgba(255,255,255,0.85);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Evidence tag */
    .evidence-tag {
        display: inline-block;
        background: rgba(233, 69, 96, 0.15);
        color: #e94560;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 0.15rem;
    }

    /* Diagnosis badge */
    .diagnosis-badge {
        background: linear-gradient(135deg, #e94560, #c23152);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
    }

    /* Confidence meter */
    .confidence-bar {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e94560;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Follow-up Q&A */
    .qa-item {
        background: rgba(22, 33, 62, 0.8);
        border-left: 3px solid #e94560;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }

    /* Hide default streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────

def load_sample_cases():
    """Load sample cases from data directory."""
    data_path = Path(__file__).parent / "data" / "sample_cases.json"
    with open(data_path, "r") as f:
        return json.load(f)


def confidence_color(score: float) -> str:
    """Return color based on confidence score."""
    if score >= 0.8:
        return "#66BB6A"
    elif score >= 0.6:
        return "#FFA726"
    else:
        return "#FF6B6B"


# ── Sidebar: Patient Input ───────────────────────────────────

with st.sidebar:
    st.markdown("## 📋 Patient Case Input")

    input_mode = st.radio(
        "Input Mode",
        ["Sample Case", "Manual Entry"],
        horizontal=True,
    )

    if input_mode == "Sample Case":
        cases = load_sample_cases()
        case_labels = [f"Case {i}: {c['chief_complaint'][:40]}..." for i, c in enumerate(cases)]
        selected_idx = st.selectbox("Select Case", range(len(cases)), format_func=lambda i: case_labels[i])
        case_data = cases[selected_idx].copy()
        ground_truth = case_data.pop("ground_truth", "")

        st.markdown(f"**Age:** {case_data['age']}  |  **Sex:** {case_data['sex']}")
        st.markdown(f"**Complaint:** {case_data['chief_complaint']}")
        st.markdown(f"**Ground Truth:** `{ground_truth}`")

    else:
        st.markdown("---")
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["M", "F", "Other"])
        chief_complaint = st.text_area("Chief Complaint", "Describe presenting symptoms...")

        symptoms_text = st.text_area(
            "Symptoms (one per line: name, severity, duration)",
            "chest pain, severe, 2 hours\nshortness of breath, moderate, 2 hours",
        )
        medical_history = st.text_input("Medical History (comma-separated)", "")
        medications = st.text_input("Current Medications (comma-separated)", "")

        # Build case data from manual input
        symptoms = []
        for line in symptoms_text.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 1:
                symptoms.append({
                    "name": parts[0],
                    "severity": parts[1] if len(parts) > 1 else "moderate",
                    "duration": parts[2] if len(parts) > 2 else "unknown",
                })

        case_data = {
            "age": age,
            "sex": sex,
            "chief_complaint": chief_complaint,
            "symptoms": symptoms,
            "medical_history": [h.strip() for h in medical_history.split(",") if h.strip()],
            "current_medications": [m.strip() for m in medications.split(",") if m.strip()],
        }
        ground_truth = ""

    st.markdown("---")
    max_rounds = st.slider("Max Consensus Rounds", 1, 5, 3)

    run_button = st.button("🚀 Run Diagnosis", type="primary", use_container_width=True)


# ── Main Content ─────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 MedCollab</h1>
    <p>Causal-Driven Multi-Agent Collaboration for Clinical Diagnosis via IBIS-Structured Argumentation</p>
</div>
""", unsafe_allow_html=True)

if run_button:
    # Build PatientCase
    patient_case = PatientCase(**case_data)

    # Display patient summary
    with st.expander("📋 Patient Case Summary", expanded=True):
        st.code(patient_case.summary(), language="text")

    # Run pipeline with status updates
    with st.status("🏥 Running MedCollab Diagnostic Pipeline...", expanded=True) as status:
        st.write("🏥 Triage Agent: Classifying case complexity...")
        final_state = run_diagnosis(
            patient_case=patient_case,
            ground_truth=ground_truth,
            max_rounds=max_rounds,
        )
        status.update(label="✅ Diagnosis Complete!", state="complete")

    # Store results in session state
    st.session_state["results"] = final_state
    st.session_state["ground_truth"] = ground_truth

# Display results if available
if "results" in st.session_state:
    final_state = st.session_state["results"]
    ground_truth = st.session_state.get("ground_truth", "")

    triage = final_state.get("triage_result", {})
    positions = final_state.get("specialist_positions", [])
    causal_chain = final_state.get("causal_chain", {})
    consensus = final_state.get("consensus_result", {})
    follow_up_q = final_state.get("follow_up_questions", [])
    follow_up_a = final_state.get("follow_up_answers", [])

    # ── Metrics Row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{triage.get('complexity', 'N/A').upper()}</div>
            <div class="metric-label">Case Complexity</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(positions)}</div>
            <div class="metric-label">Specialists Consulted</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{consensus.get('consensus_score', 0):.0%}</div>
            <div class="metric-label">Consensus Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(follow_up_q)}</div>
            <div class="metric-label">Follow-up Questions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Final Diagnosis ──
    primary_diag = consensus.get("primary_diagnosis", "Pending")
    st.markdown(f"""
    <div class="diagnosis-badge">
        🎯 {primary_diag}
    </div>
    """, unsafe_allow_html=True)

    # Ground truth comparison
    if ground_truth:
        match = ground_truth.lower() in primary_diag.lower() or primary_diag.lower() in ground_truth.lower()
        icon = "✅" if match else "❌"
        col_gt1, col_gt2 = st.columns(2)
        with col_gt1:
            st.info(f"**Ground Truth:** {ground_truth}")
        with col_gt2:
            color = "green" if match else "red"
            st.markdown(f":{color}[{icon} {'Match' if match else 'Mismatch'}]")

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏥 Triage", "🩺 Specialist Debate", "🔗 Causal Chain",
        "🤔 Patient Interaction", "⚖️ Consensus"
    ])

    with tab1:
        st.markdown("### Triage Result")
        st.markdown(f"""
        <div class="agent-card">
            <h3>🏥 Triage Agent</h3>
            <p><strong>Complexity:</strong> {triage.get('complexity', 'N/A')}</p>
            <p><strong>Recruited Specialists:</strong> {', '.join(triage.get('recruited_specialists', []))}</p>
            <p><strong>Primary Concern:</strong> {triage.get('primary_concern', 'N/A')}</p>
            <p><strong>Reasoning:</strong> {triage.get('reasoning', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Specialist IBIS Positions")
        for i, pos in enumerate(positions):
            with st.expander(
                f"🩺 {pos.get('agent_name', 'Specialist')} — "
                f"{pos.get('position', 'N/A')} "
                f"({pos.get('confidence', 0):.0%} confidence)",
                expanded=i == 0,
            ):
                st.markdown(f"**Issue:** {pos.get('issue', 'N/A')}")
                st.markdown(f"**Position:** {pos.get('position', 'N/A')}")

                conf = pos.get("confidence", 0)
                color = confidence_color(conf)
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf*100}%; background: {color};"></div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Confidence: {conf:.0%}")

                if pos.get("arguments_for"):
                    st.markdown("**Arguments For:**")
                    for arg in pos["arguments_for"]:
                        if isinstance(arg, dict):
                            st.markdown(f"- {arg.get('claim', 'N/A')} (strength: {arg.get('strength', 'N/A')})")
                            if arg.get("evidence"):
                                evidence_html = " ".join(
                                    f'<span class="evidence-tag">{e}</span>'
                                    for e in arg["evidence"]
                                )
                                st.markdown(evidence_html, unsafe_allow_html=True)

                if pos.get("arguments_against"):
                    st.markdown("**Arguments Against (Self-Critique):**")
                    for arg in pos["arguments_against"]:
                        if isinstance(arg, dict):
                            st.markdown(f"- {arg.get('claim', 'N/A')}")

                if pos.get("differential_diagnoses"):
                    st.markdown(f"**Differentials:** {', '.join(pos['differential_diagnoses'])}")

                if pos.get("recommended_tests"):
                    st.markdown(f"**Recommended Tests:** {', '.join(pos['recommended_tests'])}")

    with tab3:
        st.markdown("### Hierarchical Disease Causal Chain (HDCC)")

        if causal_chain.get("nodes"):
            # Render graph
            img_bytes = render_causal_chain(causal_chain)
            if img_bytes:
                st.image(img_bytes, caption="HDCC Visualization")

            st.markdown(f"**Root Diagnosis:** {causal_chain.get('root_diagnosis', 'N/A')}")
            st.markdown(f"**Summary:** {causal_chain.get('summary', 'N/A')}")

            if causal_chain.get("comorbidities"):
                st.markdown(f"**Comorbidities:** {', '.join(causal_chain['comorbidities'])}")

            # Node details
            with st.expander("📊 Chain Details"):
                for node in causal_chain.get("nodes", []):
                    level_emoji = {"symptom": "🔴", "mechanism": "🟠", "disease": "🟢", "comorbidity": "🔵"}.get(
                        node.get("level", ""), "⚪"
                    )
                    st.markdown(
                        f"{level_emoji} **{node.get('label', 'N/A')}** "
                        f"({node.get('level', 'N/A')}) — {node.get('description', '')}"
                    )
        else:
            st.info("Causal chain not available for this run.")

    with tab4:
        st.markdown("### 🤔 Patient Interaction Agent (Novel Contribution)")
        st.caption("Dynamic follow-up querying to fill evidence gaps — this is the primary research extension.")

        if follow_up_q:
            for i, (q, a) in enumerate(zip(follow_up_q, follow_up_a)):
                st.markdown(f"""
                <div class="qa-item">
                    <strong>Q{i+1}:</strong> {q}<br>
                    <em style="color: #66BB6A;">A{i+1}:</em> {a}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No follow-up questions were needed — evidence was sufficient.")

    with tab5:
        st.markdown("### Consensus Voting Results")

        if consensus:
            st.markdown(f"**Consensus Reached:** {'✅ Yes' if consensus.get('consensus_reached') else '❌ No'}")
            st.markdown(f"**Consensus Score:** {consensus.get('consensus_score', 0):.0%}")
            st.markdown(f"**Reasoning:** {consensus.get('reasoning', 'N/A')}")

            # Vote distribution
            votes = consensus.get("vote_distribution", {})
            if votes:
                st.markdown("**Vote Distribution:**")
                for diag, weight in sorted(votes.items(), key=lambda x: -x[1]):
                    pct = weight * 100 if weight <= 1 else weight
                    st.progress(min(pct / 100, 1.0), text=f"{diag}: {pct:.0f}%")

            # Agent scores
            agent_scores = consensus.get("agent_scores", [])
            if agent_scores:
                st.markdown("**Agent Quality Scores:**")
                for score in agent_scores:
                    name = score.get("agent_name", "Unknown")
                    logic = score.get("logic_score", 0)
                    evidence = score.get("evidence_score", 0)
                    penalized = score.get("penalized", False)
                    penalty_icon = " ⚠️" if penalized else ""
                    st.markdown(
                        f"- **{name}{penalty_icon}** — "
                        f"Logic: {logic:.0%} | Evidence: {evidence:.0%} | "
                        f"Weight: {score.get('weight', 0):.0%}"
                    )
                    if penalized:
                        st.caption(f"  Penalty: {score.get('penalty_reason', 'N/A')}")

            # Recommendations
            recs = consensus.get("recommendations", [])
            if recs:
                st.markdown("**Recommendations:**")
                for r in recs:
                    st.markdown(f"- {r}")

    # ── Raw Data Expander ──
    with st.expander("🗂 Raw Pipeline Data"):
        st.json(final_state)

else:
    # Landing page
    st.markdown("""
    ### 👋 Welcome to MedCollab

    This is a **multi-agent medical diagnosis system** that emulates the hierarchical
    consultation workflow of modern hospitals.

    **How it works:**
    1. 🏥 **Triage Agent** classifies the case and recruits specialists
    2. 🩺 **Specialist Agents** debate using IBIS argumentation protocol
    3. 🔗 **Causal Chain Builder** models pathological progression
    4. 🤔 **Patient Interaction Agent** asks follow-up questions *(Novel)*
    5. ⚖️ **Consensus Agent** performs weighted voting for final diagnosis

    **To get started:** Select a patient case in the sidebar and click **Run Diagnosis**.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 🔬 IBIS Protocol
        Every agent must back their position
        with traceable evidence and self-critique.
        """)
    with col2:
        st.markdown("""
        #### 🔗 Causal Chains
        Symptoms → Mechanisms → Disease
        modeled as a directed graph.
        """)
    with col3:
        st.markdown("""
        #### 🤔 Interactive Diagnosis
        Novel follow-up querying fills
        evidence gaps dynamically.
        """)
