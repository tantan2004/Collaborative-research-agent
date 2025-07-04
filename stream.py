import streamlit as st
from datetime import datetime
from typing import Dict, Any

try:
    from run import build_graph, GraphState, run_node
except ImportError:
    st.error("Could not import required functions from run.py.")
    st.stop()

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
}
.status-box {
    padding: 1rem;
    border-radius: 5px;
    margin: 0.5rem 0;
}
.status-running {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
}
.status-complete {
    background-color: #d4edda;
    border-left: 4px solid #28a745;
}
.status-error {
    background-color: #f8d7da;
    border-left: 4px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)

# Session state
if 'graph' not in st.session_state:
    try:
        st.session_state.graph = build_graph()
    except Exception as e:
        st.error(f"Failed to build graph: {e}")
        st.stop()

if 'research_state' not in st.session_state:
    st.session_state.research_state = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

def display_status(status: str, message: str):
    color_class = {
        "running": "status-running",
        "complete": "status-complete",
        "error": "status-error"
    }.get(status, "status-box")
    st.markdown(f'<div class="status-box {color_class}">{message}</div>', unsafe_allow_html=True)

def run_full_research_cycle(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        graph = st.session_state.graph
        progress_bar = st.progress(0)
        status_text = st.empty()

        current_state = state.copy()
        for step in range(10):
            if current_state.get("decision") == "end":
                break

            progress_bar.progress((step + 1) / 10)

            for phase in ["researcher", "summarizer", "critic"]:
                if current_state.get("decision") == "end":
                    break
                status_text.text(f"{phase.capitalize()} in progress...")
                current_state = run_node(phase, current_state)

            if current_state.get("decision") == "human_feedback":
                return current_state

        status_text.text("Research Completed.")
        progress_bar.progress(1.0)
        return current_state
    except Exception as e:
        st.error(f"Error during research: {e}")
        return state

def handle_human_decision(state: Dict[str, Any], decision: str, manual_input: str = "") -> Dict[str, Any]:
    if decision == "manual" and manual_input:
        return {**state, "summary": manual_input, "decision": "end"}
    if decision in ["end", "reresearch", "resummarize"]:
        return {**state, "decision": decision}
    return {**state, "decision": "end"}

# ────────────── Sidebar ──────────────
with st.sidebar:
    st.header("Research Controls")
    query = st.text_input("Enter your research query:")
    
    if st.button("Start Research", type="primary"):
        if query.strip():
            st.session_state.research_state = {
                "query": query,
                "raw_content": "",
                "summary": "",
                "previous_summary": "",
                "decision": "",
                "loop_count": 0,
                "research_count": 0,
                "summarize_count": 0,
                "_critic_recommendation": ""
            }
            st.session_state.current_step = "auto_research"
            st.rerun()
        else:
            st.error("Please enter a query.")

    if st.button("Clear History"):
        st.session_state.research_history = []
        st.session_state.research_state = None
        st.session_state.current_step = None
        st.rerun()

    if st.session_state.research_history:
        st.subheader("Past Queries")
        for entry in reversed(st.session_state.research_history[-5:]):
            with st.expander(f"Query: {entry['query']}"):
                st.write(f"Completed: {entry['timestamp']}")

# ────────────── Main Section ──────────────
st.markdown('<h1 class="main-header">Research Agent</h1>', unsafe_allow_html=True)

state = st.session_state.research_state

if state is None:
    st.markdown("""
        ### Welcome!
        Use the sidebar to input a query and start researching.
        This agent will use AI to research, summarize, and refine results until satisfied.
    """)
else:
    st.subheader(f"Query: {state['query']}")

    if st.session_state.current_step == "auto_research":
        st.markdown("### Running Research Agent...")
        updated = run_full_research_cycle(state)
        st.session_state.research_state = updated

        if updated.get("decision") == "human_feedback":
            st.session_state.current_step = "human_feedback"
        else:
            st.session_state.current_step = None
            st.session_state.research_history.append({
                "query": updated["query"],
                "summary": updated["summary"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
        st.rerun()

    elif st.session_state.current_step == "human_feedback":
        st.subheader("Your Input Needed")
        st.markdown("### Current Summary")
        st.markdown(state["summary"])

        if state.get("_critic_recommendation"):
            st.info(f"AI recommends: **{state['_critic_recommendation']}**")
        update=None
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(" Accept Summary"):
                updated = handle_human_decision(state, "end")
        with col2:
            if st.button(" More Research"):
                updated = handle_human_decision(state, "reresearch")
        with col3:
            if st.button(" Resummarize"):
                updated = handle_human_decision(state, "resummarize")
                

        manual_input = st.text_area("Or write your own summary:", height=200)
        if st.button("Submit Manual Summary"):
            updated = handle_human_decision(state, "manual", manual_input)

        if updated:
            st.session_state.research_state = updated
            st.session_state.current_step = None
            st.session_state.research_history.append({
                "query": updated["query"],
                "summary": updated["summary"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.rerun()
    else:
        st.success(" Research Completed")
        st.markdown("### Final Summary")
        st.markdown(state["summary"])

        with st.expander("View Raw Research Data"):
            st.text(state["raw_content"][:2000])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Continue Research"):
                st.session_state.current_step = "auto_research"
                st.rerun()
        with col2:
            if st.button("Improve Summary"):
                st.session_state.current_step = "auto_research"
                st.rerun()
