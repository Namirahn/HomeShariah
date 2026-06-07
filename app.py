import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import torch

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Shariah Compliance Portal",
    layout="wide"
)

# =========================================================
# 2. GLOBAL STYLING (MODERN PREMIUM THEME - NO EMOJIS)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Gradient Title */
.main-title {
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #1d4ed8, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

/* Metric Card */
.metric-card {
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.15);
    background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
}

/* Status Badges */
.status-flagged {
    background-color: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    border: 1px solid rgba(239, 68, 68, 0.3);
    display: inline-block;
}

.status-passed {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10b981;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    border: 1px solid rgba(16, 185, 129, 0.3);
    display: inline-block;
}

/* Result Banner */
.result-banner {
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 25px;
    font-size: 20px;
    font-weight: 800;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* Clause Container Structure */
.clause-container {
    padding: 20px;
    border-radius: 12px;
    background-color: rgba(128,128,128,0.03);
    border-left: 5px solid #ef4444;
    margin-bottom: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}

/* Login Card Container */
.login-box {
    max-width: 460px;
    margin: 30px auto 10px auto;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid rgba(128,128,128,0.15);
    background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

/* Fix Streamlit Form Padding inside Login Box */
div[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background-color: transparent !important;
}

/* Professional Form Button Custom Override */
.stButton > button {
    background: linear-gradient(90deg, #10b981, #059669) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    border: none !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #059669, #047857) !important;
    transform: scale(1.02) !important;
    color: white !important;
}

/* Outcome Alert Custom Cards */
.success-banner {
    background-color: rgba(16,185,129,0.1);
    border-left: 4px solid #10b981;
    padding: 12px;
    border-radius: 8px;
    color: #10b981;
    font-weight: 600;
    margin-top: 15px;
}
.error-banner {
    background-color: rgba(239,68,68,0.1);
    border-left: 4px solid #ef4444;
    padding: 12px;
    border-radius: 8px;
    color: #ef4444;
    font-weight: 600;
    margin-top: 15px;
}

.dataframe {
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. SESSION STATE INITIALIZATION
# =========================================================
if 'scan_history' not in st.session_state:
    st.session_state['scan_history'] = []

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None

if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None

# =========================================================
# MODULE A: USER LOGIN SECTION
# =========================================================
if not st.session_state.get('logged_in', False):
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-weight:800; letter-spacing:-0.5px; margin-top:0; padding-top:0;'>SYSTEM LOGIN</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; opacity:0.6; font-size:13px; margin-bottom:20px;'>Shariah Document Analysis Portal</p>", unsafe_allow_html=True)

    with st.form("login_form"):
        input_username = st.text_input("Username", placeholder="Enter your username...")
        input_password = st.text_input("Password", type="password", placeholder="Enter your password...")
        submit_login = st.form_submit_button("Sign In", use_container_width=True)

        if submit_login:
            if input_username == "admin" and input_password == "123456789":
                st.session_state['logged_in'] = True
                st.session_state['current_user'] = "admin"
                st.session_state['user_role'] = "Admin"
                st.markdown("<div class='success-banner'>Login Successful - Welcome Admin</div>", unsafe_allow_html=True)
                st.rerun()
            else:
                st.markdown("<div class='error-banner'>Invalid Username or Password</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# CORE PORTAL ENVIRONMENT
# =========================================================
else:
    # =========================================================
    # 4. LOAD MODEL & DATASET WITH DATA CLEANING
    # =========================================================
    @st.cache_resource
    def load_model():
        return SentenceTransformer('all-MiniLM-L6-v2')

    @st.cache_data
    def load_dataset():
        dataset_path = 'PUSTAKA CLAUSE  (2).csv'
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            
            # Remove empty strings from primary clause column
            if 'Clause' in df.columns:
                df = df.dropna(subset=['Clause'])
            
            # Reformat string conversions out of timestamp items in Clause_No
            if 'Clause_No' in df.columns:
                df['Clause_No'] = df['Clause_No'].apply(
                    lambda x: x.strftime('%d/%m/%y') if isinstance(x, datetime) else str(x)
                )
                
            # Automated classification override for layout misalignments
            if 'Label' in df.columns and 'Justification' in df.columns:
                def fix_label_anomaly(row):
                    justification = str(row['Justification']).lower()
                    current_label = row['Label']
                    if "compliant:" in justification or "patuh:" in justification:
                        return 1
                    return current_label
                df['Label'] = df.apply(fix_label_anomaly, axis=1)

            return df, "PUSTAKA CLAUSE  (2).csv"
        else:
            st.error(f"Dataset file '{dataset_path}' was not found.")
            return pd.DataFrame(), "No File"

    model = load_model()
    df_lib, file_name = load_dataset()

    # =========================================================
    # 5. KNOWLEDGE BASE EMBEDDINGS
    # =========================================================
    kb_embeddings = None

    if not df_lib.empty:
        kb_clauses = df_lib['Clause'].astype(str).tolist()
        kb_embeddings = model.encode(
            kb_clauses,
            convert_to_tensor=True
        )

    # =========================================================
    # 6. KEYWORD DETECTION MODULE
    # =========================================================
    keyword_dict = {
        "Riba": ["interest", "penalty", "late payment", "per annum", "% charge", "conventional loan"],
        "Gharar": ["uncertain", "discretion", "undefined", "subject to change"],
        "Maysir": ["guaranteed return", "no risk", "profit guaranteed"]
    }

    def keyword_score(sentence):
        sentence_lower = sentence.lower()
        score = 0
        detected_keywords = []
        categories = []

        for category, keywords in keyword_dict.items():
            for keyword in keywords:
                if keyword in sentence_lower:
                    score += 1
                    detected_keywords.append(keyword)
                    categories.append(category)

        return score, detected_keywords, list(set(categories))

    # =========================================================
    # 7. SIDEBAR (STRICT ENGLISH)
    # =========================================================
    with st.sidebar:
        st.markdown("<h2 style='font-weight:800; color:#10b981; margin-bottom:0;'>SHARIAH PORTAL</h2>", unsafe_allow_html=True)
        st.caption("Islamic Finance Document Analysis System")
        st.divider()
        
        st.markdown("### System Metadata")
        st.markdown(f"Reference Dataset: \n`{file_name}`")
        st.markdown(f"Current User: \n`{st.session_state['current_user']}`")
        st.markdown(f"Assigned Role: \n`{st.session_state['user_role']}`")
        st.divider()

        if st.button("Clear Audit History", use_container_width=True):
            st.session_state['scan_history'] = []
            st.rerun()
            
        if st.button("Sign Out", type="primary", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['current_user'] = None
            st.session_state['user_role'] = None
            st.rerun()

    # =========================================================
    # 8. MAIN TABS (STRICT ENGLISH)
    # =========================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard",
        "Reference Dataset",
        "Document Scanner",
        "Audit History",
        "About System"
    ])

    # =========================================================
    # TAB 1 — DASHBOARD
    # =========================================================
    with tab1:
        st.markdown("<div class='main-title'>SHARIAH COMPLIANCE PORTAL</div>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align:center; font-weight:400; opacity:0.7; margin-bottom:30px;'>Hybrid NLP-Based Islamic Finance Document Analysis</h5>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h1 style="color:#1d4ed8; margin:0;">{len(st.session_state["scan_history"])}</h1><p style="margin:0; font-weight:600; opacity:0.8;">Total Documents Analysed</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h1 style="color:#10b981; margin:0;">{len(df_lib)}</h1><p style="margin:0; font-weight:600; opacity:0.8;">Knowledge Base Entries</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h1 style="color:#f59e0b; margin:0;">Active</h1><p style="margin:0; font-weight:600; opacity:0.8;">AI Engine Status</p></div>', unsafe_allow_html=True)

    # =========================================================
    # TAB 2 — REFERENCE DATASET
    # =========================================================
    with tab2:
        st.subheader("Shariah Reference Dataset (Knowledge Base)")
        st.caption("Reference clauses used as the underlying ground truth data repository for executing semantic mapping scores.")
        st.write("")
        if not df_lib.empty:
            st.dataframe(df_lib, use_container_width=True)
        else:
            st.error("Error: Failed to fetch the master reference database template data.")

    # =========================================================
    # TAB 3 — DOCUMENT SCANNER
    # =========================================================
    with tab3:
        st.subheader("Hybrid NLP Document Scanner")
        st.caption("Upload Islamic home financing offer documents (PDF) for automated risk-clause detection.")
        st.write("")

        uploaded_file = st.file_uploader("Select financing agreement or documents :", type="pdf")

        if uploaded_file and not df_lib.empty and kb_embeddings is not None:
            st.write("")
            if st.button("Analysis Document", type="primary", use_container_width=True):
                with st.spinner("AI Engine parsing and evaluating document clauses..."):

                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    raw_text = " ".join([page.get_text().replace('\n', ' ') for page in doc])
                    raw_text = re.sub(r'\s+', ' ', raw_text)

                    sentences = []
                    # REMOVED GENERAL WORDS LIKE 'amount', 'tenure', 'rm' TO PREVENT FALSE NOISE BLOCKING
                    blacklist = ["www", "0.00", "0%", "visit"]

                    # Preprocessing Loop & Noise Filtering Sequence
                    for sentence in re.split(r'(?<=[.!?])\s+', raw_text):
                        sentence = sentence.strip()
                        if "___" in sentence or "date:" in sentence.lower():
                            continue

                        sentence_norm = sentence.lower()
                        # CRITICAL FIX 2: LOWERED THRESHOLD FROM 60 TO 20 CHARACTERS TO CATCH SHORT NON-COMPLIANT CLAUSES
                        if len(sentence_norm) > 20:
                            if not any(word in sentence_norm for word in blacklist):
                                if not re.match(r'^\d+(\.\d+)*\s+[A-Z\s]{5,15}$', sentence):
                                    sentences.append(sentence_norm)

                    if sentences:
                        kb_labels = df_lib['Label'].astype(int).tolist()
                        kb_bodies = df_lib['Standard_Body'].astype(str).tolist()
                        kb_standards = df_lib['Standard_No'].astype(str).tolist()
                        kb_clauses_no = df_lib['Clause_No'].astype(str).tolist()
                        kb_pages = df_lib['Page'].astype(str).tolist()
                        kb_reasons = df_lib['Justification'].astype(str).tolist()

                        input_embeddings = model.encode(sentences, convert_to_tensor=True)
                        results = []
                        total_flagged = 0
                        total_sentences = len(sentences)

                        for i, embedding in enumerate(input_embeddings):
                            similarity_scores = util.cos_sim(embedding, kb_embeddings)
                            best_index = torch.argmax(similarity_scores).item()
                            semantic_score = similarity_scores[0][best_index].item()

                            kw_score, kw_found, kw_category = keyword_score(sentences[i])
                            
                            # Standard Fusion Hybrid Scoring Calculation
                            final_score = (0.7 * semantic_score) + (0.3 * (kw_score > 0))
                            txt_lower = sentences[i]

                            # -----------------------------------------------------
                            # CRITICAL FIX 1: PUSH OVERRIDE RULES TO THE ENTRANCE GATE
                            # -----------------------------------------------------
                            is_override_triggered = False
                            forced_label = None

                            # Case A: Strict Non-Compliant Override (Interest is present, but profit is absent)
                            if "interest" in txt_lower and "profit" not in txt_lower:
                                is_override_triggered = True
                                forced_label = 0  # Force FLAGGED status

                            # Case B: Strict Compliant Safe Guards (Legitimate Islamic terms present)
                            elif "profit rate" in txt_lower or "interest/profit" in txt_lower or "etiqa" in txt_lower or "takaful" in txt_lower:
                                is_override_triggered = True
                                forced_label = 1  # Force PASSED status

                            # -----------------------------------------------------
                            # PIPELINE ROUTING DECISION MATRIX
                            # -----------------------------------------------------
                            if is_override_triggered:
                                label = forced_label
                                status = "PASSED" if label == 1 else "FLAGGED"
                            elif final_score >= 0.55:
                                label = kb_labels[best_index]
                                status = "PASSED" if label == 1 else "FLAGGED"
                            else:
                                status = "PASSED"  # Below threshold, falls back to noise or safe default

                            badge_class = "status-passed" if status == "PASSED" else "status-flagged"

                            # Only append structural log records if the active clause is isolated as FLAGGED
                            if status == "FLAGGED":
                                total_flagged += 1
                                results.append({
                                    "clause": sentences[i],
                                    "body": kb_bodies[best_index] if not df_lib.empty else "N/A",
                                    "standard": kb_standards[best_index] if not df_lib.empty else "N/A",
                                    "clause_no": kb_clauses_no[best_index] if not df_lib.empty else "N/A",
                                    "page": kb_pages[best_index] if not df_lib.empty else "N/A",
                                    "badge": f'<span class="{badge_class}">{status}</span>',
                                    "score": f"{final_score:.1%}",
                                    "justification": kb_reasons[best_index] if not df_lib.empty else "Non-compliant term detected via Keyword Override Module."
                                })

                        # =========================================================
                        # RE-ENGINEERED COMPLIANCE RATIO & VERDICT
                        # =========================================================
                        total_passed = total_sentences - total_flagged
                        compliance_score = (total_passed / total_sentences) * 100 if total_sentences > 0 else 100.0
                        verdict = "NON-COMPLIANT" if total_flagged > 0 else "COMPLIANT"
                        
                        st.session_state['scan_history'].append({
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Document Name": uploaded_file.name,
                            "Total Clauses": total_sentences,
                            "Flagged Clauses": total_flagged,
                            "Compliance Score": f"{compliance_score:.1f}%",
                            "Overall Verdict": verdict
                        })

                        # 1. Primary Document Status Banner
                        if total_flagged > 0:
                            st.markdown(f'<div class="result-banner" style="color: #ef4444; background-color: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.2);">OVERALL VERDICT: {verdict}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="result-banner" style="color: #10b981; background-color: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.2);">OVERALL VERDICT: {verdict}</div>', unsafe_allow_html=True)

                        # 2. Optimized Summary Breakdown Matrix
                        st.markdown("### Document Audit Summary")
                        m_col1, m_col2, m_col3 = st.columns(3)
                        
                        with m_col1:
                            st.markdown(f"""
                            <div style="padding:15px; border-radius:10px; background-color:rgba(128,128,128,0.05); border:1px solid rgba(128,128,128,0.2); text-align:center;">
                                <h2 style="margin:0; color:#1d4ed8;">{total_sentences}</h2>
                                <small style="font-weight:600; opacity:0.7; letter-spacing:0.5px;">TOTAL EVALUATED CLAUSES</small>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with m_col2:
                            bg_flagged = "rgba(239,68,68,0.1)" if total_flagged > 0 else "rgba(128,128,128,0.05)"
                            border_flagged = "rgba(239,68,68,0.3)" if total_flagged > 0 else "rgba(128,128,128,0.2)"
                            color_flagged = "#ef4444" if total_flagged > 0 else "#6b7280"
                            
                            st.markdown(f"""
                            <div style="padding:15px; border-radius:10px; background-color:{bg_flagged}; border:1px solid {border_flagged}; text-align:center;">
                                <h2 style="margin:0; color:{color_flagged};">{total_flagged}</h2>
                                <small style="font-weight:600; color:{color_flagged}; letter-spacing:0.5px;">NON-COMPLIANT (FLAGGED)</small>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with m_col3:
                            score_color = "#10b981" if compliance_score == 100 else "#f59e0b" if compliance_score > 80 else "#ef4444"
                            st.markdown(f"""
                            <div style="padding:15px; border-radius:10px; background-color:rgba(128,128,128,0.05); border:1px solid rgba(128,128,128,0.2); text-align:center;">
                                <h2 style="margin:0; color:{score_color};">{compliance_score:.1f}%</h2>
                                <small style="font-weight:600; opacity:0.7; letter-spacing:0.5px;">COMPLIANCE RATE</small>
                            </div>
                            """, unsafe_allow_html=True)

                        st.write("")
                        st.divider()

                        # 3. List of Granular Violations (If Any)
                        if total_flagged > 0:
                            st.markdown("### List of Shariah Infractions (Flagged Clauses):")
                            for res in results:
                                st.markdown(f"""
                                <div class="clause-container">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                        <strong>Reference: {res['body']} | {res['standard']} (Clause: {res['clause_no']}, Page: {res['page']})</strong>
                                        <div>{res['badge']} <span style="margin-left:8px; font-size:12px; font-weight:600; opacity:0.7;">Match Score: {res['score']}</span></div>
                                    </div>
                                    <p style="font-style: italic; margin-bottom: 10px; padding: 10px; background-color: rgba(0,0,0,0.02); border-radius: 6px;">"{res['clause']}"</p>
                                    <div style="margin-top: 5px; color: #ef4444;"><strong>Shariah Justification:</strong> {res['justification']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.success("Congratulations! This document has passed all automated Shariah compliance matrix boundaries.")
                    else:
                        st.warning("No valid clauses found after preprocessing text filters.")

# =========================================================
# TAB 4 — AUDIT HISTORY
# =========================================================
    with tab4:
        st.subheader("Session Audit History Logs")
        st.caption("Audit trail logs tracking document evaluation metadata within the active window session.")
        st.write("")
        if st.session_state['scan_history']:
            st.dataframe(pd.DataFrame(st.session_state['scan_history']), use_container_width=True)
        else:
            st.info("No documents evaluated within this current session yet.")

# =========================================================
# TAB 5 — ABOUT SYSTEM
# =========================================================
    with tab5:
        st.markdown("""
            <h2 style='font-weight:700;'>System Information & Methodology</h2>
            <p style='opacity:0.8;'>Islamic Home Financing Document Evaluation Portal</p>
        """, unsafe_allow_html=True)
        st.divider()

        col_intro, col_goals = st.columns(2)
        with col_intro:
            st.markdown("### Introduction / Overview")
            st.write("""
            This system is engineered to automatically evaluate and audit Islamic home financing documents 
            to detect Shariah non-compliance risks using a **Hybrid NLP** (Natural Language Processing) approach. 
            The software acts as a smart screening layer to drastically lower manual overhead errors during routine Shariah internal audits.
            """)
        with col_goals:
            st.markdown("### Objectives / Goals")
            st.markdown("""
            * **Automated Auditing:** Assists auditors and compliance officers in identifying compliant (Passed) or non-compliant (Flagged) contract clauses efficiently.
            * **Decision Support System:** Delivers rapid summary results along with structural academic justifications to serve as a fast-reference tool for Shariah boards.
            * **Regulatory Alignment:** Promotes regulatory accountability in the Islamic banking ecosystem in accordance with BNM (IFSA 2013), AAOIFI, and IFSB reference frameworks.
            """)

        st.divider()

        col_feats, col_tech = st.columns(2)
        with col_feats:
            st.markdown("### System Features")
            st.markdown("""
            1. **Document Upload & Parsing:** Ingests long-form financial agreement templates in PDF format and seamlessly handles automated clean clause segmentation using *PyMuPDF*.
            2. **Hybrid Clause Scanning:** Suppresses raw text background noise and processes individual target sentences using structural text embeddings tied to dictionary pattern matrices.
            3. **Risk Highlighting & Justification:** Labels structural non-compliance indicators (Riba, Gharar, Maysir) and displays clear academic compliance justifications.
            4. **Session Audit History:** Logs live temporary diagnostic runs as long as the dashboard context stays active to maintain a safe tracking trail.
            """)
        with col_tech:
            st.markdown("### Technology Used")
            st.info("""
            This web-based prototype application is built entirely on top of the **Streamlit Framework (Python)**, powered by a customized **Hybrid NLP Engine**:
            * **Semantic Deep-Learning Engine (70% Weight Allocation):** Driven by the core *Sentence-BERT (all-MiniLM-L6-v2)* model to capture contextual meaning and dense linguistic intent.
            * **Lexical Keyword Layer (30% Weight Allocation):** Employs rule-based syntax validation checking to guarantee critical alternative risk terms (e.g., 'Interest' or 'Penalty') are securely captured.
            """)

        st.divider()

        with st.expander("Show Technical Deep Dive (5-Layer Data Flow Pipeline)", expanded=False):
            st.write("""
            The analytical operational flow ingests raw financial text streams across 5 distinctive internal pipeline layers:
            1. **Text Extraction Layer:** Streams unstructured document data matrices via PyMuPDF.
            2. **Noise Filtering Layer:** Strips document layout formatting markers, blank lines (`___`), or stray header strings.
            3. **Tokenization Layer:** Divides bulky paragraph groups into logical independent target clauses.
            4. **Vectorization Layer:** Converts target character arrays into mathematical float tensors.
            5. **Fusion Scoring Layer:** Computes weighted classification averages and matches them directly to reference columns.
            """)

        st.markdown(f"""
            <div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid rgba(128,128,128,0.2);">
                <small>Copyright &copy; 2026 | Final Year Project (FYP) | Nur Amirah Natasha | S23B0204 </small>
            </div>
        """, unsafe_allow_html=True)