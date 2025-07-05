import streamlit as st
import sys
import os
from typing import Dict, Any
import time

# Import your backend logic
try:
    from run import (
        GraphState, 
        researcher_node,
        summarizer_node,
        critic_node,
        get_web_search_tool,
        create_llm_with_retry,
        summarizer_model,
        critic_model,
        search_tool
    )
except ImportError as e:
    st.error(f"Error importing from run.py: {e}")
    st.stop()

# Page setup
st.set_page_config(
    page_title="Collaborative Research Assistant",
    layout="wide"
)

# CSS (theme-aware)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: var(--text-color);
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: rgba(200, 200, 200, 0.05);
        border: 1px solid var(--secondary-background-color);
        margin: 1rem 0;
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<h1 class="main-header">Collaborative Research Assistant</h1>', unsafe_allow_html=True)

# Description Section
st.markdown("""
### What is this tool?

**Collaborative Research Assistant** is an intelligent AI-powered tool designed to help you explore and understand any research topic efficiently.

It works using three key AI agents:
- **Researcher** – Searches and collects information from trusted web sources.
- **Summarizer** – Distills complex data into concise, readable summaries.
- **Critic** – Evaluates the summary for completeness, relevance, and clarity.

You stay in control by providing feedback, choosing to continue researching, improve the summary, or input your own version.

Perfect for literature reviews, brainstorming, and collaborative academic work.
""")

# Session state initialization
for key in ['current_state', 'research_started', 'research_complete', 'awaiting_feedback', 'processing']:
    if key not in st.session_state:
        st.session_state[key] = False if key != 'current_state' else None

# Core logic
def run_research_cycle(state: GraphState) -> GraphState:
    st.write("Researching...")
    state = researcher_node().invoke(state)
    st.write("Summarizing...")
    state = summarizer_node().invoke(state)
    st.write("Evaluating...")
    state = critic_node().invoke(state)
    return state

def start_research(query: str):
    st.session_state.current_state = {
        "query": query,
        "raw_content": "",
        "summary": "",
        "previous_summary": "",
        "decision": "",
        "loop_count": 0,
        "research_count": 0,
        "summarize_count": 0,
        "_critic_recommendation": "",
        "_research_approaches": []
    }
    st.session_state.research_started = True
    st.session_state.research_complete = False
    st.session_state.awaiting_feedback = False
    st.session_state.processing = True

def process_research_step():
    if not st.session_state.current_state:
        return

    try:
        with st.spinner("Processing research..."):
            state = run_research_cycle(st.session_state.current_state)
            st.session_state.current_state = state
            if state.get("decision") == "human_feedback":
                st.session_state.awaiting_feedback = True
            else:
                st.session_state.research_complete = True
                st.session_state.awaiting_feedback = False
    except Exception as e:
        st.error(f"Error during research: {str(e)}")
    finally:
        st.session_state.processing = False

def handle_feedback(choice: str, manual_summary: str = ""):
    if not st.session_state.current_state:
        return

    state = st.session_state.current_state

    if choice == "Accept":
        state["decision"] = "end"
        st.session_state.research_complete = True
        st.session_state.awaiting_feedback = False
    elif choice == "Research More":
        if state.get("research_count", 0) < 4:
            state["decision"] = "reresearch"
            st.session_state.processing = True
            st.session_state.awaiting_feedback = False
        else:
            st.warning("Maximum research attempts reached.")
            return
    elif choice == "Improve Summary":
        if state.get("summarize_count", 0) < 4:
            state["decision"] = "resummarize"
            st.session_state.processing = True
            st.session_state.awaiting_feedback = False
        else:
            st.warning("Maximum summarization attempts reached.")
            return
    elif choice == "Manual Input":
        if manual_summary.strip():
            state["summary"] = manual_summary
            state["decision"] = "end"
            st.session_state.research_complete = True
            st.session_state.awaiting_feedback = False
        else:
            st.error("Please provide a manual summary.")
            return

    state["loop_count"] += 1
    st.session_state.current_state = state

# Input query
st.header("Research Query")
query = st.text_input(
    "Enter your research topic:",
    placeholder="e.g., Machine Learning, Quantum Computing, Blockchain...",
    help="Enter the topic you'd like to explore"
)

# Button layout
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Start Research", type="primary", disabled=not query or st.session_state.processing):
        start_research(query)
        process_research_step()

with col2:
    if st.button("Continue", disabled=not st.session_state.awaiting_feedback or st.session_state.processing):
        if st.session_state.current_state and st.session_state.current_state.get("decision") in ["reresearch", "resummarize"]:
            process_research_step()

with col3:
    if st.button("Reset", type="secondary"):
        for key in ['current_state', 'research_started', 'research_complete', 'awaiting_feedback', 'processing']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Feedback handling
if st.session_state.awaiting_feedback and st.session_state.current_state:
    st.markdown("---")
    st.header("Feedback Required")

    colA, colB = st.columns([2, 1])
    with colA:
        feedback_choice = st.selectbox(
            "Choose an action:",
            ["Accept", "Research More", "Improve Summary", "Manual Input"],
            key="feedback_choice"
        )
    with colB:
        if st.button("Submit Feedback", type="primary"):
            if feedback_choice == "Manual Input":
                if 'manual_summary_input' in st.session_state and st.session_state.manual_summary_input:
                    handle_feedback(feedback_choice, st.session_state.manual_summary_input)
                    st.rerun()
                else:
                    st.error("Please provide a manual summary below.")
            else:
                handle_feedback(feedback_choice)
                if feedback_choice in ["Research More", "Improve Summary"]:
                    process_research_step()
                st.rerun()

    if feedback_choice == "Manual Input":
        st.text_area(
            "Enter your manual summary:",
            height=150,
            placeholder="Write your own summary here...",
            key="manual_summary_input"
        )

# Results section
if st.session_state.current_state:
    st.markdown("---")
    st.header("Research Results")
    state = st.session_state.current_state

    if state.get("summary"):
        st.subheader("Current Summary")
        st.markdown(f'<div class="summary-box">{state["summary"]}</div>', unsafe_allow_html=True)

    if state.get("raw_content"):
        with st.expander("Raw Research Content"):
            st.text_area("Raw Content", state["raw_content"], height=200, disabled=True)

    if state.get("previous_summary"):
        with st.expander("Previous Summary"):
            st.write(state["previous_summary"])

    if state.get("_research_approaches"):
        with st.expander("Research Approaches Used"):
            for i, approach in enumerate(state["_research_approaches"], 1):
                st.write(f"{i}. {approach}")

# Loading state
if st.session_state.processing:
    st.info("Processing research cycle...")

st.markdown("---")
