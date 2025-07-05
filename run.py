from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict
from difflib import SequenceMatcher
import time

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

class GraphState(TypedDict):
    query: str
    raw_content: str
    summary: str
    previous_summary: str
    decision: str
    loop_count: int
    research_count: int
    summarize_count: int
    _critic_recommendation: str
    _research_approaches: list

def get_web_search_tool():
    try:
        if not TAVILY_API_KEY:
            return None
        search_tool = TavilySearchResults(
            max_results=3,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
        )
        test_result = search_tool.run("python")
        return search_tool
    except:
        return None

def create_llm_with_retry(model_name: str, temperature: float = 0.1, max_tokens: int = 1024, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            llm = ChatGroq(
                model=model_name,
                api_key=GROQ_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60,
                max_retries=2
            )
            test_response = llm.invoke("Say 'OK'")
            test_content = test_response.content if hasattr(test_response, 'content') else str(test_response)
            if test_content and len(test_content.strip()) > 0:
                return llm
        except:
            time.sleep(2)
    return None

summarizer_model = create_llm_with_retry("llama-3.3-70b-versatile", 0.3, 2048)
critic_model = create_llm_with_retry("llama-3.1-8b-instant", 0.1, 512)
search_tool = get_web_search_tool()

def is_similar(a, b, threshold=0.75):
    if not a or not b:
        return False
    ratio = SequenceMatcher(None, a.strip(), b.strip()).ratio()
    return ratio >= threshold

def researcher_node():
    def run_research(state: GraphState) -> GraphState:
        query = state["query"]
        research_count = state.get("research_count", 0) + 1
        approaches = state.get("_research_approaches", [])
        result = None
        strategies = [
            f"{query} overview explanation",
            f"{query} technical details and examples",
            f"{query} applications and case studies",
            f"{query} current trends and future prospects",
            f"{query} challenges and innovations"
        ]
        available = [s for s in strategies if s not in approaches] or strategies
        search_query = available[min(research_count - 1, len(available) - 1)]
        approaches.append(search_query)
        if search_tool:
            try:
                search_results = search_tool.run(search_query)
                if search_results and len(str(search_results)) > 50:
                    result = f"Web search results for '{query}':\n\n{str(search_results)[:2000]}"
            except:
                pass
        if not result or len(result) < 100:
            try:
                temp_variation = 0.2 + (research_count * 0.1)
                research_llm = create_llm_with_retry("llama-3.3-70b-versatile", temp_variation, 1500)
                prompt = f"""
You are a domain expert tasked with explaining the concept of "{query}" in a structured and insightful manner.

Focus areas:
- Clear and concise definition
- Fundamental principles or working mechanisms
- Real-world use cases and applications
- Industry relevance and current developments
- Future outlook or emerging trends

Avoid repeating previous attempts: {approaches}

Create a comprehensive, readable explanation (300–500 words).
"""
                response = research_llm.invoke(prompt)
                result = response.content if hasattr(response, "content") else str(response)
            except:
                result = f"Minimal information for '{query}'."
        return {
            **state,
            "raw_content": result,
            "research_count": research_count,
            "_research_approaches": approaches,
            "decision": "",
        }
    return RunnableLambda(run_research)

def summarizer_node():
    def summarize(state: GraphState) -> GraphState:
        content = state.get("raw_content", "")
        query = state.get("query", "")
        current_summary = state.get("summary", "")
        summarize_count = state.get("summarize_count", 0) + 1
        is_error_content = (
            content.startswith("Research error") or 
            content.startswith("Minimal information") or
            len(content.strip()) < 100
        )
        if summarize_count > 4:
            return {
                **state,
                "previous_summary": state.get("summary", ""),
                "summary": current_summary or "Maximum summarization attempts reached.",
                "summarize_count": summarize_count,
                "decision": "end",
            }
        if is_error_content:
            final_summary = f"Research on '{query}' was limited."
        else:
            temp_variation = 0.3 + (summarize_count * 0.1)
            summarizer = create_llm_with_retry("llama-3.3-70b-versatile", temp_variation, 2048)
            prompt = f"""
Summarize the following content related to "{query}" in an organized manner:

Include:
- Definition
- Key principles or mechanism
- Applications or case studies
- Importance or relevance
- Any trends, challenges, or future scope

Source:
{content[:2000]}
"""
            try:
                response = summarizer.invoke(prompt)
                final_summary = response.content.strip() if hasattr(response, "content") else str(response).strip()
            except:
                final_summary = "Summary generation failed."
        return {
            **state,
            "previous_summary": state.get("summary", ""),
            "summary": final_summary,
            "summarize_count": summarize_count,
            "decision": "",
        }
    return RunnableLambda(summarize)

def critic_node():
    def evaluate(state: GraphState) -> GraphState:
        summary = state.get("summary", "")
        prev = state.get("previous_summary", "")
        loop_count = state.get("loop_count", 0) + 1
        research_count = state.get("research_count", 0)
        summarize_count = state.get("summarize_count", 0)
        query = state.get("query", "")

        try:
            prompt = f"""
You are a research quality evaluator in a multi-agent system.

Evaluate the current summary based on the following:

- Does it fully answer the user's research query: "{query}"?
- Does it cover key components: definition, principles, examples, applications, importance?
- Is it better than or significantly different from the previous summary?
- Is it detailed, well-structured, and non-redundant?

Previous summary (if any):
{prev or 'None'}

Current summary:
{summary}

Progress:
- Loop Count: {loop_count}
- Research Attempts: {research_count}/4
- Summarize Attempts: {summarize_count}/4

Decision rules:
- Respond with **reresearch** if more or better information is needed.
- Respond with **resummarize** if the content is enough but summary can be improved.
- Respond with **end** if the summary is complete and well-structured.

Reply ONLY with one word: reresearch, resummarize, or end.
"""
            response = critic_model.invoke(prompt)
            llm_decision = response.content.strip().lower()

            if "reresearch" in llm_decision:
                decision = "reresearch"
            elif "resummarize" in llm_decision:
                decision = "resummarize"
            elif "end" in llm_decision:
                decision = "end"
            else:
                decision = "resummarize"

            if decision == "resummarize" and prev and is_similar(summary, prev):
                decision = "reresearch" if research_count < 4 else "end"

            if decision == "reresearch" and research_count >= 3 and is_similar(summary, prev):
                decision = "human_feedback"

            if decision == "end" and prev and is_similar(summary, prev) and loop_count >= 2:
                decision = "human_feedback"

            if loop_count >= 6 or (research_count >= 4 and summarize_count >= 4):
                decision = "human_feedback"

            if decision in ["end", "resummarize", "reresearch"]:
                decision = "human_feedback"

        except:
            decision = "human_feedback"

        print(f"[Critic Evaluation] Loop: {loop_count} | Research: {research_count}/4 | Summarize: {summarize_count}/4 | Decision: {decision}")

        return {
            **state,
            "decision": decision,
            "loop_count": loop_count,
            "_critic_recommendation": decision
        }

    return RunnableLambda(evaluate)

def human_feedback_node():
    def get_feedback(state: GraphState) -> GraphState:
        summary = state.get("summary", "")
        query = state.get("query", "")
        print(f"\nQuery: {query}\nSummary: {summary}")
        print("1. Accept\n2. Research\n3. Summarize\n4. Manual Input")
        choice = input("Choose (1-4): ").strip()
        if choice == "1":
            return {**state, "decision": "end", "loop_count": state.get("loop_count", 0) + 1}
        elif choice == "2":
            return {**state, "decision": "reresearch", "loop_count": state.get("loop_count", 0) + 1}
        elif choice == "3":
            return {**state, "decision": "resummarize", "loop_count": state.get("loop_count", 0) + 1}
        elif choice == "4":
            print("Enter manual summary (end with two empty lines):")
            lines, empty = [], 0
            while empty < 2:
                line = input()
                if not line.strip(): empty += 1
                else: empty = 0
                lines.append(line)
            summary = "\n".join([l for l in lines if l.strip()])
            return {**state, "summary": summary, "decision": "end", "loop_count": state.get("loop_count", 0) + 1}
        return {**state, "decision": "end", "loop_count": state.get("loop_count", 0) + 1}
    return RunnableLambda(get_feedback)

def route_decision(state: GraphState) -> str:
    d = state.get("decision", "end")
    if d == "reresearch" and state.get("research_count", 0) < 4:
        return "researcher"
    elif d == "resummarize" and state.get("summarize_count", 0) < 4:
        return "summarizer"
    elif d == "human_feedback":
        return "human_feedback"
    return END

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("researcher", researcher_node())
    builder.add_node("summarizer", summarizer_node())
    builder.add_node("critic", critic_node())
    builder.add_node("human_feedback", human_feedback_node())
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "summarizer")
    builder.add_edge("summarizer", "critic")
    builder.add_conditional_edges("critic", route_decision)
    builder.add_conditional_edges("human_feedback", route_decision)
    return builder.compile()

def main():
    graph = build_graph()
    query = input("Enter your research query: ").strip()
    if not query:
        return
    initial_state = {
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
    final_state = graph.invoke(initial_state, config={"recursion_limit": 30})
    print(f"\nFinal Summary:\n{final_state.get('summary', 'No summary')}")
    print(f"\nTotal Loops: {final_state.get('loop_count', 0)}")

if __name__ == "__main__":
    main()
