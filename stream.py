import streamlit as st
from datetime import datetime
from typing import Dict, Any

try:
    from run import build_graph, GraphState
except ImportError:
    st.error("Could not import required functions from run.py.")
    st.stop()

st.set_page_config(page_title="Research Agent", page_icon="🔍", layout="wide")

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

def run_research_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run one step of research using the graph"""
    try:
        # Create a temporary graph with human feedback disabled for auto steps
        temp_state = state.copy()
        
        # Run the graph for one iteration, but capture human feedback requests
        result = st.session_state.graph.invoke(temp_state, config={"recursion_limit": 5})
        
        return result
    except Exception as e:
        st.error(f"Error during research step: {e}")
        return {**state, "decision": "end"}

def handle_human_decision(state: Dict[str, Any], decision: str, manual_input: str = "") -> Dict[str, Any]:
    """Handle human decision and return updated state"""
    if decision == "manual" and manual_input.strip():
        return {
            **state, 
            "summary": manual_input.strip(), 
            "decision": "end",
            "loop_count": state.get("loop_count", 0) + 1
        }
    if decision in ["end", "reresearch", "resummarize"]:
        return {
            **state, 
            "decision": decision,
            "loop_count": state.get("loop_count", 0) + 1
        }
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
        st.session_state.research_state = None
        st.session_state.current_step = None
        st.session_state.research_history = []
        st.rerun()

    if st.session_state.research_history:
        st.subheader("Past Queries")
        for entry in reversed(st.session_state.research_history[-5:]):
            with st.expander(f"Query: {entry['query']}"):
                st.write(f"Completed: {entry['timestamp']}")

# ────────────── Main Area ──────────────
st.markdown('<h1 class="main-header">Research Agent</h1>', unsafe_allow_html=True)

state = st.session_state.research_state

if state is None:
    st.markdown("""
    ### Welcome!
    Enter a query in the sidebar to begin. The AI will research, summarize, and improve until you're satisfied.
    """)
else:
    st.subheader(f"Query: {state['query']}")
    
    # Show current stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Loop Count", state.get("loop_count", 0))
    with col2:
        st.metric("Research Count", state.get("research_count", 0))
    with col3:
        st.metric("Summarize Count", state.get("summarize_count", 0))
    with col4:
        st.metric("Current Decision", state.get("decision", "None"))

    # Auto research step
    if st.session_state.current_step == "auto_research":
        st.markdown("### Running Research Agent...")
        
        # Show what action will be taken
        decision = state.get("decision", "")
        if decision == "reresearch":
            st.info(" Conducting additional research...")
        elif decision == "resummarize":
            st.info(" Improving the summary...")
        else:
            st.info(" Starting initial research...")
            
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Running research cycle...")
        progress_bar.progress(0.5)
        
        # Run one complete cycle
        updated = run_research_step(state)
        
        progress_bar.progress(1.0)
        status_text.text("Research cycle completed.")
        
        st.session_state.research_state = updated

        # Check what to do next
        if updated.get("decision") == "human_feedback":
            st.session_state.current_step = "human_feedback"
            st.info(" Human input required - redirecting...")
        elif updated.get("decision") == "end":
            st.session_state.current_step = None
            st.session_state.research_history.append({
                "query": updated["query"],
                "summary": updated["summary"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success(" Research completed automatically!")
        else:
            # Continue with more cycles if needed
            st.session_state.current_step = "auto_research"
            
        st.rerun()

    # Human decision required
    elif st.session_state.current_step == "human_feedback":
        st.subheader(" Your Input Needed")
        
        # Show current summary
        st.markdown("### Current Summary")
        st.markdown(state["summary"])

        # Show AI recommendation
        if state.get("_critic_recommendation"):
            recommendation = state["_critic_recommendation"]
            if recommendation == "reresearch":
                st.info(" AI Recommendation: **More Research Needed** - The content may be insufficient or needs different information.")
            elif recommendation == "resummarize":
                st.info(" AI Recommendation: **Improve Summary** - The information is good but summary needs better structure or detail.")
            else:
                st.info(f" AI Recommendation: **{recommendation}**")

        # Show execution stats
        st.markdown("### Current Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Research attempts:** {state.get('research_count', 0)}/4")
            st.write(f"**Total loops:** {state.get('loop_count', 0)}")
        with col2:
            st.write(f"**Summarize attempts:** {state.get('summarize_count', 0)}/4")
            st.write(f"**Last decision:** {state.get('decision', 'None')}")

        # Show action buttons
        st.markdown("### Choose Your Action:")
        
        updated = None
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(" Accept Summary", type="primary"):
                updated = handle_human_decision(state, "end")
                
        with col2:
            research_disabled = state.get("research_count", 0) >= 4
            if st.button(" More Research", disabled=research_disabled):
                if not research_disabled:
                    updated = handle_human_decision(state, "reresearch")
                else:
                    st.error("Maximum research attempts reached!")
            if research_disabled:
                st.caption(" Max research attempts reached")
                
        with col3:
            summarize_disabled = state.get("summarize_count", 0) >= 4
            if st.button("Improve Summary", disabled=summarize_disabled):
                if not summarize_disabled:
                    updated = handle_human_decision(state, "resummarize")
                else:
                    st.error("Maximum summarize attempts reached!")
            if summarize_disabled:
                st.caption(" Max summarize attempts reached")

        st.markdown("### Or Write Your Own Summary:")
        manual_input = st.text_area("Enter your custom summary:", height=200, placeholder="Write your own summary here...")
        
        if st.button("Submit Manual Summary"):
            if manual_input.strip():
                updated = handle_human_decision(state, "manual", manual_input)
            else:
                st.warning("Please enter a summary.")

        if updated:
            st.session_state.research_state = updated
            decision = updated.get("decision")
            
            if decision in ["reresearch", "resummarize"]:
                st.session_state.current_step = "auto_research"
                if decision == "reresearch":
                    st.success("Will conduct additional research...")
                else:
                    st.success(" Will improve the summary...")
            else:
                st.session_state.current_step = None
                st.session_state.research_history.append({
                    "query": updated["query"],
                    "summary": updated["summary"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.success(" Research completed!")
            st.rerun()

    else:
        st.success(" Research Completed!")
        
        st.markdown("### Final Summary")
        st.markdown(state["summary"])

        with st.expander(" View Raw Research Content"):
            raw_content = state.get("raw_content", "No raw content available")
            st.text(raw_content[:2000] + "..." if len(raw_content) > 2000 else raw_content)

        with st.expander(" View Execution Statistics"):
            st.write(f"**Total loops:** {state.get('loop_count', 0)}")
            st.write(f"**Research attempts:** {state.get('research_count', 0)}")
            st.write(f"**Summarize attempts:** {state.get('summarize_count', 0)}")
            st.write(f"**Final decision:** {state.get('decision', 'N/A')}")

        # Action buttons
        st.markdown("### Want to Continue?")
        col1, col2 = st.columns(2)
        
        with col1:
            research_disabled = state.get("research_count", 0) >= 4
            if st.button(" Continue Research", type="secondary", disabled=research_disabled):
                if not research_disabled:
                    st.session_state.research_state = {
                        **state,
                        "decision": "reresearch"
                    }
                    st.session_state.current_step = "auto_research"
                    st.rerun()
            if research_disabled:
                st.caption("Max research attempts reached")
                
        with col2:
            summarize_disabled = state.get("summarize_count", 0) >= 4
            if st.button("Improve Summary", type="secondary", disabled=summarize_disabled):
                if not summarize_disabled:
                    st.session_state.research_state = {
                        **state,
                        "decision": "resummarize"
                    }
                    st.session_state.current_step = "auto_research"
                    st.rerun()
            if summarize_disabled:
                st.caption("Max summarize attempts reached")
