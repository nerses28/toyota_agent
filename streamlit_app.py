import os
import io
import re
import traceback
import streamlit as st
from contextlib import redirect_stdout, redirect_stderr
import agent_core


# --------------------------- Helpers ---------------------------

CITATION_RE = re.compile(r"\[([^\[\]]+?):(\d+)\]")  # matches [file_path:page]
SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SQL_ANY_BLOCK_RE = re.compile(r"```\s*(SELECT[\s\S]*?)```", re.IGNORECASE)

def extract_citations(answer_text: str):
    cites = CITATION_RE.findall(answer_text or "")
    cites = [(fp.strip(), int(pg)) for fp, pg in cites]
    unique_paths = []
    seen = set()
    for fp, _ in cites:
        if fp not in seen:
            unique_paths.append(fp)
            seen.add(fp)
    return cites, unique_paths

def extract_sql(answer_text: str):
    m = SQL_BLOCK_RE.search(answer_text or "")
    if m:
        return m.group(1).strip()
    m = SQL_ANY_BLOCK_RE.search(answer_text or "")
    if m:
        return m.group(1).strip()
    return ""

def load_agent_with_logs(db_path: str, chroma_path: str, collection_name: str, system_prompt_path: str):
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            agent = agent_core.build_agent(
                db_path=db_path,
                chroma_dir=chroma_path,
                collection_name=collection_name,
                system_prompt_path=system_prompt_path,
            )
        except Exception:
            traceback.print_exc()
            agent = None
    logs = out.getvalue() + "\n" + err.getvalue()
    return agent, logs

def run_agent_with_logs(agent, question: str):
    out = io.StringIO()
    err = io.StringIO()
    result_text = ""
    with redirect_stdout(out), redirect_stderr(err):
        try:
            result_text = agent.run(question)
        except Exception:
            traceback.print_exc()
            result_text = ""
    logs = out.getvalue() + "\n" + err.getvalue()
    return result_text, logs


# --------------------------- UI ---------------------------

st.set_page_config(page_title="Toyota/Lexus Agent", layout="wide")
st.title("Toyota/Lexus Agent")

# Config inputs
colA, colB, colC, colD = st.columns(4)
with colA:
    db_path = st.text_input("SQLite DB path", value="toyota.db")
with colB:
    chroma_path = st.text_input("Chroma path", value="chroma")
with colC:
    collection_name = st.text_input("Chroma collection", value="docs")
with colD:
    system_prompt_path = st.text_input("System prompt (.txt)", value="system_prompt.txt")

st.divider()

user_q = st.text_input("Ask a question", value="", placeholder="e.g., What is the standard Toyota warranty for Europe?")

if "agent" not in st.session_state:
    agent, build_logs = load_agent_with_logs(db_path, chroma_path, collection_name, system_prompt_path)
    st.session_state.agent = agent
    st.session_state.build_logs = build_logs

# Buttons row
btn_run_col, btn_reload_col = st.columns(2)
with btn_run_col:
    run_clicked = st.button("Run", type="primary", use_container_width=True)
with btn_reload_col:
    reload_clicked = st.button("Reload agent", use_container_width=True)

# Handle reload
if reload_clicked:
    agent, build_logs = load_agent_with_logs(db_path, chroma_path, collection_name, system_prompt_path)
    st.session_state.agent = agent
    st.session_state.build_logs = build_logs

# Handle run
answer_text = ""
run_logs = ""
if run_clicked:
    if st.session_state.agent is None:
        st.error("Agent is not initialized. Check logs below.")
    elif not user_q.strip():
        st.warning("Enter a question.")
    else:
        answer_text, run_logs = run_agent_with_logs(st.session_state.agent, user_q)

# Answer window
st.subheader("Answer")
if answer_text:
    st.markdown(answer_text)
else:
    st.caption("No answer yet.")

st.divider()

# Two windows below: citations and SQL
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cited documents")
    if answer_text:
        cites, unique_paths = extract_citations(answer_text)
        if not unique_paths:
            st.caption("No citations found.")
        else:
            with st.expander("Show file:page references"):
                for fp, pg in cites:
                    st.write(f"{os.path.basename(fp)} : page {pg}")
    else:
        st.caption("No citations yet.")

with col2:
    st.subheader("SQL used")
    if answer_text:
        sql = extract_sql(answer_text)
        if sql:
            st.code(sql, language="sql")
        else:
            st.caption("No SQL block found.")
    else:
        st.caption("No SQL yet.")

st.divider()

# Debug window
st.subheader("Debug logs")
debug_text = ""
if "build_logs" in st.session_state and st.session_state.build_logs:
    debug_text += st.session_state.build_logs.strip() + "\n"
if run_logs:
    debug_text += run_logs.strip() + "\n"

st.text_area("Logs", value=debug_text.strip(), height=300)
