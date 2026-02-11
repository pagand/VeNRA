import streamlit as st
import json
import pandas as pd
import os
import numpy as np
from pathlib import Path

# --- CONFIG & PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANDIDATES_PATH = PROJECT_ROOT / "data/training_candidates/candidate_train.jsonl"
AUDIT_CACHE_PATH = PROJECT_ROOT / "data/training_ready/audit_cache.jsonl"
DECISIONS_PATH = PROJECT_ROOT / "data/training_ready/human_audit_decisions.jsonl"

st.set_page_config(layout="wide", page_title="VeNRA Refinement Studio", page_icon="🧪")

def is_empty(val):
    """Helper to check if a value is None, NaN, or empty string."""
    if val is None: return True
    if isinstance(val, float) and np.isnan(val): return True
    if str(val).lower() == "nan": return True
    if str(val).strip() == "": return True
    return False

def get_best_val(row, keys, default="N/A"):
    """Returns the first non-empty value from a list of keys."""
    for k in keys:
        val = row.get(k)
        if not is_empty(val):
            return val
    return default

def load_data():
    """Load and synchronize data safely while auditor is running."""
    candidates = {}
    if CANDIDATES_PATH.exists():
        with open(CANDIDATES_PATH, "r") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                candidates[obj["id"]] = obj
            
    audits = []
    if AUDIT_CACHE_PATH.exists():
        with open(AUDIT_CACHE_PATH, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    audits.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    decided_ids = set()
    if DECISIONS_PATH.exists():
        with open(DECISIONS_PATH, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    decided_ids.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    continue
    
    if not audits:
        return candidates, [], {"Status": "Waiting for auditor..."}
    
    df_audit = pd.DataFrame(audits)
    
    # Normalize 'status' column if missing
    if "status" not in df_audit.columns:
        df_audit["status"] = None
    
    mask_review = (
        (df_audit["status"] == "review") | 
        (df_audit["status"].isna() & (df_audit.get("is_valid_sabotage", True) == False))
    )
    
    mask_pending = mask_review & (~df_audit["id"].isin(decided_ids))
    review_queue = df_audit[mask_pending].to_dict('records')
    
    verified_count = len(df_audit[
        (df_audit["status"] == "verified") | 
        (df_audit.get("is_valid_sabotage") == True)
    ])
    
    stats = {
        "Total Audited": len(df_audit),
        "Auto-Verified": verified_count,
        "In Review Queue": len(review_queue),
        "Human Decisions": len(decided_ids)
    }
    return candidates, review_queue, stats

# --- LOAD DATA ---
candidates, audit_queue, stats = load_data()

# --- STATE MANAGEMENT ---
if "index" not in st.session_state:
    st.session_state.index = 0

if st.session_state.index >= len(audit_queue) and len(audit_queue) > 0:
    st.session_state.index = 0

# --- SIDEBAR ---
st.sidebar.header("Data Pipeline")
for label, val in stats.items():
    st.sidebar.metric(label, val)
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.title("🧪 VeNRA Active Data Refinement Studio")

# --- UI LOGIC ---
if not audit_queue:
    st.success("🎉 Queue is empty! (Everything is either verified or reviewed)")
    if st.button("Check for new Auditor results"):
        st.rerun()

else:
    # Get current item
    current_audit = audit_queue[st.session_state.index]
    row_id = current_audit["id"]
    raw_data = candidates.get(row_id, {})
    
    if not raw_data:
        st.error(f"Critical Error: ID {row_id} missing from candidates file.")
        if st.button("Skip Broken Record"):
            st.session_state.index += 1
            st.rerun()
        st.stop()

    # --- TOP ROW: Contextual metadata ---
    reason = get_best_val(current_audit, ['validation_reason', 'rejection_reason'], "Manual review required.")
    st.warning(f"**Audit Rejection Reason:** {reason}")
    
    # Metadata and Navigation Bar
    n_col1, n_col2, n_col3 = st.columns([1, 2, 1])
    with n_col1:
        if st.button("⬅️ Previous", use_container_width=True, disabled=st.session_state.index == 0):
            st.session_state.index -= 1
            st.rerun()
    with n_col2:
        # Interactive Jump-to-Sample
        sub_c1, sub_c2, sub_c3 = st.columns([1, 0.6, 1])
        with sub_c1:
            st.markdown("<h3 style='text-align: right;'>Sample</h3>", unsafe_allow_html=True)
        with sub_c2:
            jump_val = st.number_input("Jump", min_value=1, max_value=len(audit_queue), value=st.session_state.index + 1, label_visibility="collapsed")
            if jump_val != st.session_state.index + 1:
                st.session_state.index = jump_val - 1
                st.rerun()
        with sub_c3:
            st.markdown(f"<h3 style='text-align: left;'>/ {len(audit_queue)}</h3>", unsafe_allow_html=True)
    with n_col3:
        if st.button("Next (Skip) ➡️", use_container_width=True, disabled=st.session_state.index >= len(audit_queue) - 1):
            st.session_state.index += 1
            st.rerun()

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        target_grp = get_best_val(current_audit, ['audit_target_group'], "Legacy Sabotage")
        st.write(f"**Audit Target:** `{target_grp}`")
        st.write(f"**Attack Type:** `{raw_data.get('sabotage_info', {}).get('type', 'N/A')}`")
    with m_col2:
        st.write(f"**Intended Lie:** :red[**{get_best_val(current_audit, ['injected_value'])}**]")
    with m_col3:
        st.write(f"**Teacher Verdict:** `{get_best_val(current_audit, ['teacher_label'])}`")
        conf = get_best_val(current_audit, ['teacher_confidence', 'confidence'])
        st.write(f"**Confidence:** {conf}")

    st.divider()

    # --- FORM START ---
    with st.form(key=f"form_{row_id}"):
        col_evidence, col_edit = st.columns([1, 1])

        with col_evidence:
            st.subheader("📝 Evidence (Read Only)")
            with st.container(height=700, border=True):
                st.markdown("**User Question:**")
                st.info(raw_data.get('inputs', {}).get('query', 'N/A'))
                st.markdown("**Text Context:**")
                chunks = raw_data.get('inputs', {}).get('context_chunks', [])
                for chunk in chunks:
                    if "|" in chunk: st.markdown(chunk)
                    else: st.text(chunk)
                st.markdown("**Logic Trace:**")
                st.code(raw_data.get('inputs', {}).get('trace_code', ''), language='python')

        with col_edit:
            st.subheader("🛠️ Refinement Zone (Editable)")
            edited_sentence = st.text_area("🎯 Edit Target Sentence", value=raw_data.get('target_sentence', ''), height=100)
            edited_trace = st.text_area("⚙️ Edit Logic Trace", value=raw_data.get('inputs', {}).get('trace_code', ''), height=150)
            edited_thinking = st.text_area("💭 Edit Internal Thinking", value=get_best_val(current_audit, ['teacher_thinking'], ""), height=200)
            edited_analysis = st.text_area("🔍 Edit Final Analysis", value=get_best_val(current_audit, ['teacher_analysis'], ""), height=120)

        st.divider()
        st.write("### Commit Decision")
        c1, c2, c3, c4 = st.columns(4)
        submit_unfounded = c1.form_submit_button("🔥 UNFOUNDED", type="primary", use_container_width=True)
        submit_supported = c2.form_submit_button("✅ SUPPORTED", use_container_width=True)
        submit_general = c3.form_submit_button("🌐 GENERAL", use_container_width=True)
        submit_discard = c4.form_submit_button("🗑️ DISCARD", use_container_width=True)

        if submit_unfounded: decision_label = "Unfounded"
        elif submit_supported: decision_label = "Supported"
        elif submit_general: decision_label = "General"
        elif submit_discard: decision_label = "Discard"
        else: decision_label = None

        if decision_label:
            decision = {
                "id": row_id,
                "final_label": decision_label,
                "final_sentence": edited_sentence,
                "final_trace": edited_trace,
                "final_thinking": edited_thinking,
                "final_analysis": edited_analysis,
                "metadata": {
                    "original_teacher_label": current_audit.get("teacher_label"),
                    "sabotage_type": raw_data.get('sabotage_info', {}).get('type', 'manual'),
                    "is_manual_fix": True,
                    "audit_target_group": target_grp
                }
            }
            with open(DECISIONS_PATH, "a") as f:
                f.write(json.dumps(decision) + "\n")
            st.session_state.index += 1
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("VeNRA HITL v3.4 | Studio Mode")