# 🏥 MedCollab — Multi-Agent Medical Diagnosis System

**Causal-Driven Multi-Agent Collaboration for Full-Cycle Clinical Diagnosis via IBIS-Structured Argumentation**

A LangGraph-powered multi-agent framework that extends the MedCollab paper with a **novel Patient Interaction Agent** for dynamic follow-up querying, making the diagnostic process interactive rather than static.

---

## 🏗️ Architecture

```
Patient Case (text + optional lab values)
        ↓
  [Triage Agent] — classifies complexity, recruits specialists
        ↓
  [Specialist Agents] — Cardiologist / Neurologist / GP etc.
  Each argues with IBIS positions (Issue → Position → Evidence)
        ↓
  [Causal Chain Builder] — links symptoms → disease progression (HDCC)
        ↓
  [Patient Interaction Agent] ← NOVEL CONTRIBUTION
  Asks follow-up questions, updates evidence dynamically
        ↓                    ↺ loops back if new evidence
  [Consensus Agent / GP] — weighted voting, penalizes inconsistency
        ↓                    ↺ loops back if no consensus
  Final Diagnosis Report + Causal Explanation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API key
```

**Supported LLM Providers** (change `LLM_PROVIDER` in `.env`):
- `openai` — GPT-4o (default)
- `ollama` — Local models (Mistral, Llama, etc.)
- `anthropic` — Claude
- `google` — Gemini
- `azure` — Azure OpenAI

### 3. Run via CLI

```bash
# Run with default case (STEMI)
python run.py

# Run specific case (0-4)
python run.py --case-index 2

# Run custom case
python run.py --case-file my_case.json

# Verbose logging
python run.py --verbose
```

### 4. Run Streamlit UI

```bash
streamlit run app.py
```

## 📁 Project Structure

```
medAgent/
├── app.py                     # Streamlit web UI
├── run.py                     # CLI entry point
├── requirements.txt
├── .env.example
├── data/
│   └── sample_cases.json      # 5 hand-crafted test cases
├── src/
│   ├── config.py              # LLM config, constants
│   ├── models/                # Pydantic data models
│   │   ├── patient.py         # PatientCase, Symptom, LabResult
│   │   ├── ibis.py            # IBISPosition, IBISArgument
│   │   ├── causal_chain.py    # CausalChainNode, HDCC
│   │   └── diagnosis.py       # DiagnosisReport, ConsensusResult
│   ├── agents/                # Agent implementations
│   │   ├── triage.py          # Triage Agent
│   │   ├── specialist.py      # Specialist Agent factory
│   │   ├── causal_builder.py  # Causal Chain Builder
│   │   ├── patient_agent.py   # Patient Interaction Agent ★
│   │   └── consensus.py       # GP/Consensus Agent
│   ├── graph/                 # LangGraph orchestration
│   │   ├── state.py           # AgentState TypedDict
│   │   └── workflow.py        # StateGraph definition
│   ├── prompts/               # LLM prompt templates
│   └── utils/
│       ├── llm.py             # Modular LLM factory
│       └── visualization.py   # HDCC graph rendering
└── eval/
    ├── loader.py              # MedQA dataset loader
    └── metrics.py             # Evaluation metrics
```

## 🔬 Key Components

### IBIS Argumentation Protocol
Every specialist must structure their diagnosis as:
- **Issue** → The diagnostic question
- **Position** → Their diagnosis
- **Arguments For** → Evidence-backed supporting claims
- **Arguments Against** → Honest self-critique
- **Confidence** → 0.0 to 1.0

### Hierarchical Disease Causal Chain (HDCC)
Links symptoms → mechanisms → diseases → comorbidities as a directed graph with explicit causal relationships.

### Patient Interaction Agent (Novel)
Analyzes evidence gaps in specialist positions and generates targeted follow-up questions. In automated evaluation mode, patient responses are simulated by the LLM.

## 📊 Sample Cases

| # | Case | Complexity | Ground Truth |
|---|------|-----------|-------------|
| 0 | Chest pain, diaphoresis, elevated troponin | Complex | STEMI |
| 1 | Headache, neck stiffness, fever | Moderate | Bacterial Meningitis |
| 2 | Fatigue, polydipsia, elevated glucose + creatinine | Complex | DKA + AKI |
| 3 | Chronic cough, hemoptysis, night sweats | Moderate | Pulmonary TB |
| 4 | Joint pain, butterfly rash, positive ANA | Moderate | SLE |

## 📝 License

Research project — academic use.
