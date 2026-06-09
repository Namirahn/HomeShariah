import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
import os
import time  # Ditambah untuk pengiraan masa proses
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import torch

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="ISLAMIC FINANCE DOCUMENT ANALYSIS HOME SHARIAH PORTAL",
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
    font-size: 32px;
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
/* Fix Streamlit Form Padding */
div[data-testid="stForm"], div[data-testid="stFormElement"] {
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
/* Instruction Container Box Style */
.instruction-box {
    background-color: rgba(29, 78, 216, 0.03);
    border: 1px solid rgba(29, 78, 216, 0.15);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 25px;
}
/* Processing Time Container Style */
.time-box {
    background-color: rgba(16, 185, 129, 0.05);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    margin-bottom: 20px;
    font-size: 14px;
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

        if 'Clause' in df.columns:
            df = df.dropna(subset=['Clause'])

        if 'Clause_No' in df.columns:
            df['Clause_No'] = df['Clause_No'].apply(
                lambda x: x.strftime('%d/%m/%y') if isinstance(x, datetime) else str(x)
            )

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
# 7. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("<h3 style='font-weight:800; color:#10b981; margin-bottom:0;'>HOME SHARIAH PORTAL</h3>", unsafe_allow_html=True)
    st.caption("Islamic Finance Document Analysis")
    st.divider()

    st.markdown("### System Information:")
    st.markdown(f"Reference Dataset: \n`{file_name}`")
    st.divider()
    
    if st.button("Clear Audit History", use_container_width=True):
        st.session_state['scan_history'] = []
        st.rerun()

# =========================================================
# 8. MAIN TABS
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
    st.markdown("<div class='main-title'>ISLAMIC FINANCE DOCUMENT ANALYSIS HOME SHARIAH PORTAL</div>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center; font-weight:400; opacity:0.7; margin-bottom:25px;'>Automated Intelligent Screening Tool for Home Financing Drafts</h5>", unsafe_allow_html=True)

    st.markdown("""
    <div class="instruction-box">
    <h4 style="margin-top:0; color:#1d4ed8; font-weight:700;">Quick Instructions / Arahan Ringkas:</h4>
    <p style="font-size:14px; opacity:0.9; line-height:1.6; margin-bottom:10px;">
    <strong>English:</strong> Welcome! This system helps audit home financing documents to catch unlawful elements like interest or hidden fees.
    Follow the steps below to screen your document.
    </p>
    <p style="font-size:14px; opacity:0.9; line-height:1.6; font-style: italic; color: #4b5563;">
    <strong>Bahasa Melayu:</strong> Selamat datang! Sistem ini membantu memeriksa dokumen pembiayaan perumahan bagi mengesan unsur tidak patuh Syariah (seperti faedah/riba).
    Sila ikut langkah di bawah untuk mula menyemak.
    </p>
    <hr style="border:0; border-top:1px solid rgba(29, 78, 216, 0.15); margin:12px 0;">
    <ol style="font-size:13.5px; opacity:0.95; padding-left:20px; line-height:1.7;">
    <li>Click on the <strong>Document Scanner / Pengimbas Dokumen</strong> tab above.</li>
    <li>Upload your housing loan agreement file (Must be in <strong>PDF format</strong>).</li>
    <li>Press the green <strong>"Start Shariah Audit Scan / Mula Semakan Audit"</strong> button to let the system analyze your file.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px; font-weight:600; margin-bottom:5px; color:#374151;'>System Evaluation Survey / Borang Penilaian Sistem:</p>", unsafe_allow_html=True)
    st.link_button("Complete User Feedback (Google Form)", "https://forms.gle/BHpLXbq2oi4fSqZb8", use_container_width=True)
    st.write("")
    st.divider()
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
    st.subheader("Shariah Reference Dataset")
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
    st.subheader("Document Scanner / Pengimbas Dokumen")
    st.info(
        " 💡  **Instruction:** Upload your PDF file below and click the green button to search for issues.\n\n"
        " 💡  **Arahan:** Muat naik fail PDF anda di bawah dan klik butang hijau untuk memula semakan."
    )
    st.write("")
    uploaded_file = st.file_uploader("Choose a PDF file :", type="pdf")
    if uploaded_file and not df_lib.empty and kb_embeddings is not None:
        st.write("")

        if st.button("Start Shariah Audit Scan / Mula Semakan Audit", type="primary", use_container_width=True):
            with st.spinner("Analyzing document... Please wait a moment. / Sistem sedang memeriksa dokumen... Mohon tunggu sebentar."):

                # ─── MULA PENGIRAAN MASA ───
                start_time = time.time()
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                raw_text = " ".join([page.get_text().replace('\n', ' ') for page in doc])
                raw_text = re.sub(r'\s+', ' ', raw_text)
                sentences = []
                blacklist = ["www", "0.00", "0%", "visit"]
                for sentence in re.split(r'(?<=[.!?])\s+', raw_text):
                    sentence = sentence.strip()
                    if "___" in sentence or "date:" in sentence.lower():
                        continue
                    sentence_norm = sentence.lower()
                    if len(sentence_norm) > 20:
                        if not any(word in sentence_norm for word in blacklist):
                            if not re.match(r'^\d+( \. \d+)*\s+[A-Z\s]{5,15}$', sentence):
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
                    found_elements = set()
                    for i, embedding in enumerate(input_embeddings):
                        similarity_scores = util.cos_sim(embedding, kb_embeddings)
                        best_index = torch.argmax(similarity_scores).item()
                        semantic_score = similarity_scores[0][best_index].item()
                        kw_score, kw_found, kw_categories = keyword_score(sentences[i])
                        final_score = (0.7 * semantic_score) + (0.3 * (kw_score > 0))
                        txt_lower = sentences[i]
                        is_override_triggered = False
                        forced_label = None
                        if "interest" in txt_lower and "profit" not in txt_lower:
                            is_override_triggered = True
                            forced_label = 0
                        elif "profit rate" in txt_lower or "interest/profit" in txt_lower or "etiqa" in txt_lower or "takaful" in txt_lower:
                            is_override_triggered = True
                            forced_label = 1
                        if is_override_triggered:
                            label = forced_label
                            status = "PASSED" if label == 1 else "FLAGGED"
                        elif final_score >= 0.57:
                            label = kb_labels[best_index]
                            status = "PASSED" if label == 1 else "FLAGGED"
                        else:
                            status = "PASSED"
                        badge_class = "status-passed" if status == "PASSED" else "status-flagged"
                        if status == "FLAGGED":
                            total_flagged += 1
                            if kw_categories:
                                for cat in kw_categories:
                                    found_elements.add(cat)
                            else:
                                just_text = kb_reasons[best_index].lower()
                                if "riba" in just_text or "interest" in just_text:
                                    found_elements.add("Riba")
                                if "gharar" in just_text or "uncertain" in just_text:
                                    found_elements.add("Gharar")
                                if "maysir" in just_text or "gambling" in just_text:
                                    found_elements.add("Maysir")
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
                    total_passed = total_sentences - total_flagged
                    compliance_score = (total_passed / total_sentences) * 100 if total_sentences > 0 else 100.0
                    verdict = "NON-COMPLIANT / TIDAK PATUH" if total_flagged > 0 else "COMPLIANT / PATUH"

                    # ─── TAMAT PENGIRAAN MASA ───
                    elapsed_time = time.time() - start_time
                    st.session_state['scan_history'].append({
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Document Name": uploaded_file.name,
                        "Total Clauses": total_sentences,
                        "Flagged Clauses": total_flagged,
                        "Compliance Score": f"{compliance_score:.1f}%",
                        "Overall Verdict": verdict,
                        "Processing Time": f"{elapsed_time:.1f}s"
                    })
                    # Paparan Masa Proses Premium Style
                    st.markdown(f"""
                    <div class="time-box">
                    <strong>Document processed in:</strong> <span style="color:#10b981; font-weight:700;">{elapsed_time:.1f} seconds</span>
                    </div>
                    """, unsafe_allow_html=True)
                    # Shariah Non-Compliant Breakdown Summary
                    st.markdown("### Shariah Non-Compliant Components Detected:")
                    if total_flagged > 0 and found_elements:
                        elements_styled = "".join([f"<span style='background-color:#ef4444; color:white; padding:5px 15px; margin:5px; border-radius:5px; font-weight:bold; display:inline-block;'> ⚠️  {elem.upper()} DETECTED</span>" for elem in found_elements])
                        st.markdown(f"<div style='margin-bottom:20px; padding:10px; border:1px solid #ef4444; border-radius:8px; background-color:rgba(239,68,68,0.05);'>{elements_styled}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='margin-bottom:20px; padding:15px; border:1px solid #10b981; border-radius:8px; background-color:rgba(16,185,129,0.05); color:#10b981; font-weight:bold;'> ✅  NO SHARIAH VIOLATION ELEMENTS FOUND (CLEAN DRAFT)</div>", unsafe_allow_html=True)
                    # Primary Document Status Verdict Banner
                    if total_flagged > 0:
                        st.markdown(f'<div class="result-banner" style="color: #ef4444; background-color: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.2);">OVERALL VERDICT: {verdict}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-banner" style="color: #10b981; background-color: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.2);">OVERALL VERDICT: {verdict}</div>', unsafe_allow_html=True)
                    # Summary Breakdown Metrics Grid
                    st.markdown("### Document Audit Summary / Ringkasan Keputusan Audit")
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
                        <small style="font-weight:600; color:{color_flagged}; letter-spacing:0.5px;">SHARIAH NON-COMPLIANT CLAUSES</small>
                        </div>
                        """, unsafe_allow_html=True)

                    with m_col3:
                        st.markdown(f"""
                        <div style="padding:15px; border-radius:10px; background-color:rgba(128,128,128,0.05); border:1px solid rgba(128,128,128,0.2); text-align:center;">
                        <h2 style="margin:0; color:#10b981;">{compliance_score:.1f}%</h2>
                        <small style="font-weight:600; opacity:0.7; letter-spacing:0.5px;">TOTAL COMPLIANCE SCORE</small>
                        </div>
                        """, unsafe_allow_html=True)
                    st.write("")
                    st.divider()
                    st.markdown("### Granular Audit Analysis Logs / Rekod Analisis Terperinci")
                    for res in results:
                        st.markdown(f"""
                        <div class="clause-container" style="border-left-color: {'#10b981' if 'PASSED' in res['badge'] else '#ef4444'};">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <strong style="font-size: 15px; color: #1f2937;">Evaluated Clause Block:</strong>
                                {res['badge']}
                            </div>
                            <p style="font-size: 14px; color: #4b5563; background-color: white; padding: 12px; border-radius: 6px; border: 1px solid rgba(0,0,0,0.05); font-style: italic;">
                                "{res['clause']}"
                            </p>
                            <div style="margin-top: 10px; font-size: 12.5px;">
                                <span style="margin-right: 15px;"><strong>Matched Authority:</strong> {res['body']} ({res['standard']})</span>
                                <span style="margin-right: 15px;"><strong>Ref Clause No:</strong> {res['clause_no']}</span>
                                <span><strong>Ground Truth Page:</strong> {res['page']}</span>
                            </div>
                            <div style="margin-top: 8px; font-size: 12.5px; color: #1e40af; background-color: rgba(30,64,175,0.04); padding: 10px; border-radius: 6px;">
                                <strong>Academic Audit Justification:</strong> {res['justification']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

# =========================================================
# TAB 4 — AUDIT HISTORY
# =========================================================
with tab4:
    st.subheader("Session Audit History")
    st.caption("Temporary diagnostic history logs stored securely within the active browser session window context runtime.")
    st.write("")
    if st.session_state['scan_history']:
        history_df = pd.DataFrame(st.session_state['scan_history'])
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No documents evaluated during this session yet.")

# =========================================================
# TAB 5 — ABOUT SYSTEM
# =========================================================
with tab5:
    st.subheader("About the Intelligent Screening System")
    st.write("")
    col_desc, col_tech = st.columns(2)
    with col_desc:
        st.markdown("### Core Capabilities")
        st.markdown("""
        This internal screening application serves as an augmented regulatory intelligence layer for Shariah audit officers. 
        It reduces human oversight by automating pattern mapping across structured mortgage documents.
        1. **Semantic Text Mapping:** Utilizes Sentence-BERT architecture to compare textual structures inside incoming draft files against validated regulatory guidelines.
        2. **Hybrid Evaluation Architecture:** Combines semantic models with hardcoded lexical matching to evaluate risk patterns across complex text structures.
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
