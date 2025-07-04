import streamlit as st
from typing import Dict, Any
from datetime import datetime

try:
    from run import build_graph, GraphState
except ImportError:
    st.error("Could not import run.py. Make sure it's in the same directory.")
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
    .stats-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .decision-buttons {
        display: flex;
        gap: 10px;
        margin: 1rem 0;
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

if 'research_state' not in st.session_state:
    st.session_state.research_state = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = None
if 'graph' not in st.session_state:
    try:
        st.session_state.graph = build_graph()
    except Exception as e:
        st.error(f"Failed to initialize research graph: {e}")
        st.stop()

def display_status(status: str, message: str):
    if status == "running":
        st.markdown(f'<div class="status-box status-running">{message}</div>', unsafe_allow_html=True)
    elif status == "complete":
        st.markdown(f'<div class="status-box status-complete">{message}</div>', unsafe_allow_html=True)
    elif status == "error":
        st.markdown(f'<div class="status-box status-error">{message}</div>', unsafe_allow_html=True)

def run_research_step(state: Dict[str, Any], step: str):
    try:
        if step == "researcher":
            from run import researcher_node
            node = researcher_node()
            return node.invoke(state)
        elif step == "summarizer":
            from run import summarizer_node
            node = summarizer_node()
            return node.invoke(state)
        elif step == "critic":
            from run import critic_node
            node = critic_node()
            return node.invoke(state)
        else:
            return state
    except Exception as e:
        st.error(f"Error in {step}: {e}")
        return state

def handle_human_decision(state: Dict[str, Any], decision: str, manual_input: str = ""):
    try:
        if decision == "end":
            return {**state, "decision": "end"}
        elif decision == "reresearch":
            if state.get('research_count', 0) >= 4:
                st.warning("Maximum research attempts reached.")
                return {**state, "decision": "end"}
            return {**state, "decision": "reresearch"}
        elif decision == "resummarize":
            if state.get('summarize_count', 0) >= 4:
                st.warning("Maximum summarize attempts reached.")
                return {**state, "decision": "end"}
            return {**state, "decision": "resummarize"}
        elif decision == "manual" and manual_input:
            return {
                **state,
                "summary": manual_input,
                "decision": "end"
            }
        else:
            return {**state, "decision": "end"}
    except Exception as e:
        st.error(f"Error handling decision: {e}")
        return {**state, "decision": "end"}

def run_full_research_cycle(state: Dict[str, Any]):
    try:
        graph = st.session_state.graph
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        current_state = state.copy()
        step_count = 0
        max_steps = 10
        
        while step_count < max_steps:
            step_count += 1
            progress_bar.progress(min(step_count / max_steps, 1.0))
            
            if current_state.get('decision') == 'end':
                break
            
            loop_count = current_state.get('loop_count', 0)
            research_count = current_state.get('research_count', 0)
            summarize_count = current_state.get('summarize_count', 0)
            
            if loop_count >= 10 or (research_count >= 4 and summarize_count >= 4):
                break
            
            if step_count == 1 or current_state.get('decision') == 'reresearch':
                status_text.text("Researching...")
                from run import researcher_node
                node = researcher_node()
                current_state = node.invoke(current_state)
            
            if current_state.get('decision') != 'end':
                status_text.text("Summarizing...")
                from run import summarizer_node
                node = summarizer_node()
                current_state = node.invoke(current_state)
            
            if current_state.get('decision') != 'end':
                status_text.text("Evaluating...")
                from run import critic_node
                node = critic_node()
                current_state = node.invoke(current_state)
            
            decision = current_state.get('decision', 'end')
            
            if decision == 'human_feedback':
                status_text.text("Waiting for your input...")
                progress_bar.empty()
                return current_state
            elif decision == 'end':
                break
            elif decision not in ['reresearch', 'resummarize']:
                break
        
        progress_bar.progress(1.0)
        status_text.text("Research completed!")
        
        return current_state
        
    except Exception as e:
        st.error(f"Research cycle error: {e}")
        return state

st.markdown('<h1 class="main-header">Research Agent</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Research Controls")
    
    query = st.text_input("Enter your research query:", 
                         placeholder="e.g., artificial intelligence trends 2024")
    
    if st.button("Start Research", type="primary", use_container_width=True):
        if query.strip():
            initial_state = {
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
            st.session_state.research_state = initial_state
            st.session_state.current_step = "auto_research"
            st.rerun()
        else:
            st.error("Please enter a research query")
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.research_history = []
        st.session_state.research_state = None
        st.session_state.current_step = None
        st.rerun()
    
    if st.session_state.research_history:
        st.header("Research History")
        for i, item in enumerate(reversed(st.session_state.research_history[-5:])):
            with st.expander(f"Query: {item['query'][:30]}..."):
                st.write(f"**Completed:** {item['timestamp']}")

if st.session_state.research_state is None:
    st.markdown("""
    ## Welcome to the Research Agent! 
    
    This intelligent research assistant uses AI to:
    - **Research** your topics using web search and knowledge bases
    - **Summarize** findings into clear, structured information
    - **Optimize** results through iterative improvement
    - **Collaborate** with you for the best outcomes
    
    ### How to use:
    1. Enter your research query in the sidebar
    2. Click "Start Research" to begin
    3. Review and guide the research process
    4. Get comprehensive, well-structured results
    
    **Example queries:**
    - "machine learning applications in healthcare"
    - "renewable energy trends 2024"
    - "blockchain technology explained"
    - "climate change impact on agriculture"
    """)

else:
    state = st.session_state.research_state
    
    st.subheader(f"Researching: {state['query']}")
    
    if st.session_state.current_step == "auto_research":
        st.markdown("### Running Automated Research...")
        
        with st.container():
            updated_state = run_full_research_cycle(state)
            st.session_state.research_state = updated_state
            
            if updated_state.get('decision') == 'human_feedback':
                st.session_state.current_step = "human_feedback"
            else:
                st.session_state.current_step = None
                
                st.session_state.research_history.append({
                    'query': state['query'],
                    'summary': updated_state['summary'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                })
            
            st.rerun()
    
    elif st.session_state.current_step == "human_feedback":
        st.markdown("---")
        st.subheader("Your Input Needed")
        
        if state.get('summary'):
            st.markdown("### Current Summary:")
            st.markdown(state['summary'])
        
        if state.get('_critic_recommendation'):
            st.info(f"**AI Recommendation:** {state['_critic_recommendation']}")
        
        st.markdown("### What would you like to do?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Accept Summary", use_container_width=True):
                updated_state = handle_human_decision(state, "end")
                st.session_state.research_state = updated_state
                st.session_state.current_step = None
                
                st.session_state.research_history.append({
                    'query': state['query'],
                    'summary': updated_state['summary'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.rerun()
        
        with col2:
            if st.button("Research More", use_container_width=True):
                if state.get('research_count', 0) < 4:
                    updated_state = handle_human_decision(state, "reresearch")
                    st.session_state.research_state = updated_state
                    st.session_state.current_step = "auto_research"
                    st.rerun()
                else:
                    st.error("Maximum research attempts reached")
        
        with col3:
            if st.button("Improve Summary", use_container_width=True):
                if state.get('summarize_count', 0) < 4:
                    updated_state = handle_human_decision(state, "resummarize")
                    st.session_state.research_state = updated_state
                    st.session_state.current_step = "auto_research"
                    st.rerun()
                else:
                    st.error("Maximum summarize attempts reached")
        
        st.markdown("### Or provide your own improvements:")
        manual_input = st.text_area("Enter your improved summary:", 
                                  height=200,
                                  placeholder="Provide your own summary or improvements...")
        
        if st.button("Use Manual Input", use_container_width=True):
            if manual_input.strip():
                updated_state = handle_human_decision(state, "manual", manual_input)
                st.session_state.research_state = updated_state
                st.session_state.current_step = None
                
                st.session_state.research_history.append({
                    'query': state['query'],
                    'summary': updated_state['summary'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.rerun()
            else:
                st.error("Please enter some text for manual input")
    
    else:
        display_status("complete", "Research completed!")
        
        st.markdown("### Final Summary")
        if state.get('summary'):
            st.markdown(state['summary'])
        else:
            st.warning("No summary available")
        
        if state.get('raw_content') and len(state.get('raw_content', '')) > 100:
            with st.expander("View Raw Research Data"):
                st.text(state['raw_content'][:2000] + "..." if len(state['raw_content']) > 2000 else state['raw_content'])
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Continue Research", use_container_width=True):
                st.session_state.current_step = "auto_research"
                st.rerun()
        
        with col2:
            if st.button("Improve Summary", use_container_width=True):
                st.session_state.current_step = "auto_research"
                st.rerun()

st.markdown("---")