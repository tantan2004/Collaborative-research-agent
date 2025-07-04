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

if not TAVILY_API_KEY:
    print("WARNING: TAVILY_API_KEY not found - web search will be limited")
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
        
        try:
            test_result = search_tool.run("python")
            return search_tool
        except Exception as e:
            return None
        
    except Exception as e:
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
            else:
                raise Exception("LLM test returned empty response")
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2)
    
    return None

try:
    summarizer_model = create_llm_with_retry("llama-3.3-70b-versatile", 0.3, 2048)
    critic_model = create_llm_with_retry("llama-3.1-8b-instant", 0.1, 512)
    search_tool = get_web_search_tool()
    
except Exception as e:
    print(f"Initialization failed: {e}")
    raise

def researcher_node():
    def run_research(input_state: GraphState) -> GraphState:
        try:
            query = input_state["query"]
            research_count = input_state.get("research_count", 0) + 1
            
            if research_count > 3:
                result = f"Research attempt limit reached for query: {query}."
            else:
                result = None
                
                if search_tool:
                    try:
                        search_query = f"{query} basics explanation"
                        search_results = search_tool.run(search_query)
                        
                        if search_results and len(str(search_results)) > 50:
                            result = f"Web search results for '{query}':\n\n{str(search_results)[:2000]}"
                    except Exception as e:
                        pass
                
                if not result or len(result) < 100:
                    try:
                        research_llm = create_llm_with_retry("llama-3.3-70b-versatile", 0.2, 1500)
                        
                        fallback_prompt = f"""Please explain {query} in simple terms. Include:

1. What it is (definition)
2. How it works (basic principles)
3. Why it matters (applications/importance)
4. Current status or developments

Keep the explanation clear and informative, around 300-500 words.

Topic: {query}"""

                        fallback_result = research_llm.invoke(fallback_prompt)
                        fallback_content = fallback_result.content if hasattr(fallback_result, "content") else str(fallback_result)
                        
                        if fallback_content and len(fallback_content.strip()) > 100:
                            result = f"Knowledge base information for '{query}':\n\n{fallback_content}"
                        else:
                            result = f"Limited information available for '{query}'. This may be a very specialized or emerging topic."
                            
                    except Exception as e:
                        result = f"Research encountered difficulties for '{query}'. Error: {str(e)}"
                
                if not result or len(result) < 50:
                    result = f"Minimal information available for '{query}'. This could indicate a very specialized topic or technical issues."
            
            return {
                **input_state, 
                "raw_content": result,
                "research_count": research_count,
                "decision": "",
            }
            
        except Exception as e:
            return {
                **input_state,
                "raw_content": f"Research error for '{input_state.get('query', 'unknown')}': {str(e)}",
                "research_count": input_state.get("research_count", 0) + 1,
                "decision": "end"
            }
    
    return RunnableLambda(run_research)

def summarizer_node():
    def summarize(input_state: GraphState) -> GraphState:
        try:
            content = input_state.get("raw_content", "")
            query = input_state.get("query", "")
            current_summary = input_state.get("summary", "")
            summarize_count = input_state.get("summarize_count", 0) + 1
            
            if summarize_count > 3:
                final_summary = current_summary if current_summary else "Maximum summarization attempts reached."
            else:
                is_error_content = (
                    content.startswith("Research error") or 
                    content.startswith("Research encountered") or
                    content.startswith("Minimal information") or
                    len(content.strip()) < 100
                )
                
                if is_error_content:
                    final_summary = f"Research on '{query}' was limited. The topic may require specialized sources or more specific search terms. Available information indicates this is related to {query.lower()} but detailed coverage was not possible."
                else:
                    if summarize_count == 1:
                        prompt = f"""Based on the following information about {query}, create a clear summary with 4-6 bullet points:

{content[:2000]}

Format your response as bullet points covering:
• Definition/what it is
• Key characteristics or principles
• Applications or importance
• Current relevance

Topic: {query}
Create a well-structured summary:"""

                    else:
                        prompt = f"""Improve this summary about {query}:

Current Summary:
{current_summary}

Source Material:
{content[:1500]}

Create an enhanced summary with 4-6 clear bullet points that better explains {query}."""

                    try:
                        max_attempts = 3
                        final_summary = None
                        
                        for attempt in range(max_attempts):
                            try:
                                response = summarizer_model.invoke(prompt)
                                candidate_summary = response.content.strip() if hasattr(response, "content") else str(response).strip()
                                
                                if candidate_summary and len(candidate_summary) > 50:
                                    final_summary = candidate_summary
                                    break
                                    
                            except Exception as attempt_error:
                                if attempt == max_attempts - 1:
                                    raise attempt_error
                                time.sleep(1)
                        
                        if not final_summary:
                            sentences = content.replace('\n', ' ').split('. ')[:3]
                            basic_content = '. '.join(sentences)
                            final_summary = f"Summary of {query}: {basic_content[:400]}..."
                        
                        if len(final_summary) < 50:
                            final_summary = f"Brief summary for '{query}': {final_summary}\n\nNote: Summary is limited due to source material constraints."
                            
                    except Exception as e:
                        if current_summary:
                            final_summary = current_summary
                        else:
                            words = content.split()[:100]
                            emergency_summary = ' '.join(words)
                            final_summary = f"Basic information about {query}: {emergency_summary}..."
            
            return {
                **input_state,
                "previous_summary": input_state.get("summary", ""),
                "summary": final_summary,
                "summarize_count": summarize_count,
                "decision": "",
            }
            
        except Exception as e:
            return {
                **input_state,
                "summary": f"Summarization failed for '{input_state.get('query', 'unknown')}': Critical error occurred",
                "summarize_count": input_state.get("summarize_count", 0) + 1,
                "decision": "end"
            }
    
    return RunnableLambda(summarize)

def critic_node():
    def evaluate(input_state: GraphState) -> GraphState:
        try:
            summary = input_state.get("summary", "")
            prev_summary = input_state.get("previous_summary", "")
            loop_count = input_state.get("loop_count", 0) + 1
            research_count = input_state.get("research_count", 0)
            summarize_count = input_state.get("summarize_count", 0)
            query = input_state.get("query", "")
            raw_content = input_state.get("raw_content", "")
            
            # Use LLM for intelligent decision making
            critic_prompt = f"""You are a research quality critic. Analyze the following research summary and decide what action to take next.

QUERY: {query}
CURRENT SUMMARY:
{summary}

PREVIOUS SUMMARY: {prev_summary if prev_summary else "None"}

RAW CONTENT PREVIEW: {raw_content[:300]}...

STATISTICS:
- Loop: {loop_count}
- Research attempts: {research_count}
- Summarize attempts: {summarize_count}
- Summary length: {len(summary)} chars
- Word count: {len(summary.split())} words

EVALUATION CRITERIA:
1. Does the summary adequately answer the query?
2. Is the information comprehensive and well-structured?
3. Are there clear bullet points or structured information?
4. Does it cover key aspects (definition, principles, applications, importance)?
5. Is the content detailed enough (minimum 100+ words, 4+ bullet points)?

DECISION OPTIONS:
- "end": Summary is comprehensive and high quality
- "resummarize": Summary needs improvement, restructuring, or more detail
- "reresearch": Content is insufficient, need more source material

LIMITS: Max 4 research attempts, Max 4 summarize attempts

Based on your analysis, respond with ONLY ONE WORD: end, resummarize, or reresearch"""

            try:
                critic_response = critic_model.invoke(critic_prompt)
                llm_decision = critic_response.content.strip().lower() if hasattr(critic_response, "content") else str(critic_response).strip().lower()
                
                if "reresearch" in llm_decision:
                    original_decision = "reresearch"
                elif "resummarize" in llm_decision:
                    original_decision = "resummarize"
                elif "end" in llm_decision:
                    original_decision = "end"
                else:
                    original_decision = "resummarize"
                
                if original_decision == "resummarize" and prev_summary and is_similar(summary, prev_summary):
                    original_decision = "reresearch" if research_count < 4 else "end"
                if loop_count < 10 and (original_decision == "reresearch" or original_decision == "resummarize"):
                  decision = original_decision
                else:
                   decision = "human_feedback"
                    
            except Exception as e:
                print(f"LLM critic evaluation failed: {e}")
                failure_indicators = ["failed", "error", "insufficient", "limited"]
                has_failure = any(indicator.lower() in summary.lower() for indicator in failure_indicators)
                is_too_short = len(summary.strip()) < 80
                
                if has_failure and research_count < 4:
                    original_decision = "reresearch"
                elif is_too_short and summarize_count < 4:
                    original_decision = "resummarize"
                else:
                    original_decision = "end"
                    
                decision = "human_feedback"
            
            return {
                **input_state,
                "decision": decision,
                "loop_count": loop_count,
                "_critic_recommendation": original_decision
            }
            
        except Exception as e:
            print(f"Critic node error: {e}")
            return {
                **input_state,
                "decision": "end",
                "loop_count": input_state.get("loop_count", 0) + 1
            }
    
    return RunnableLambda(evaluate)

def is_similar(a, b, threshold=0.75):
    """Check if two strings are similar above a threshold using sequence matching."""
    if not a or not b:
        return False
    
    ratio = SequenceMatcher(None, a.strip(), b.strip()).ratio()
    return ratio >= threshold

def human_feedback_node():
    """Node for human intervention to review and decide on next action."""
    def get_human_feedback(input_state: GraphState) -> GraphState:
        try:
            summary = input_state.get("summary", "")
            query = input_state.get("query", "")
            research_count = input_state.get("research_count", 0)
            summarize_count = input_state.get("summarize_count", 0)
            loop_count = input_state.get("loop_count", 0)
            
            # Get the critic's original recommendation
            critic_recommendation = input_state.get('_critic_recommendation', 'Not available')
            
            print("\n" + "="*50)
            print("HUMAN REVIEW REQUIRED")
            print("="*50)
            print(f"\nQuery: {query}")
            print(f"\nCurrent Summary:")
            print("-" * 30)
            print(summary)
            print("-" * 30)
            
            print(f"\nExecution Stats:")
            print(f"• Loops completed: {loop_count}")
            print(f"• Research attempts: {research_count}")
            print(f"• Summarize attempts: {summarize_count}")
            
            print(f"\nCritic's Recommendation: {critic_recommendation}")
            
            print(f"\nAvailable Actions:")
            print("1. ACCEPT - Summary is satisfactory, end the process")
            print("2. RESEARCH - Need more/different information")
            print("3. SUMMARIZE - Current info is good but summary needs improvement")
            print("4. MANUAL - Provide your own summary/improvements")
            
            while True:
                try:
                    choice = input("\nEnter your choice (1-4): ").strip()
                    
                    if choice == "1" or choice.lower() in ["accept", "a"]:
                        decision = "end"
                        print("Summary accepted. Ending process.")
                        break
                        
                    elif choice == "2" or choice.lower() in ["research", "r"]:
                        if research_count >= 4:
                            print("Maximum research attempts reached. Try summarize or manual instead.")
                            continue
                        decision = "reresearch"
                        print("Will conduct additional research.")
                        break
                        
                    elif choice == "3" or choice.lower() in ["summarize", "s"]:
                        if summarize_count >= 4:
                            print("Maximum summarize attempts reached. Try research or manual instead.")
                            continue
                        decision = "resummarize"
                        print("Will improve the summary.")
                        break
                        
                    elif choice == "4" or choice.lower() in ["manual", "m"]:
                        print("\nProvide your improvements or new summary:")
                        print("(Press Enter twice when done)")
                        
                        manual_input = []
                        empty_lines = 0
                        while empty_lines < 2:
                            line = input()
                            if line.strip() == "":
                                empty_lines += 1
                            else:
                                empty_lines = 0
                            manual_input.append(line)
                        
                        while manual_input and manual_input[-1].strip() == "":
                            manual_input.pop()
                            
                        if manual_input:
                            manual_summary = "\n".join(manual_input)
                            decision = "end"
                            print("Manual input accepted.")
                            
                            return {
                                **input_state,
                                "summary": manual_summary,
                                "decision": decision,
                                "loop_count": loop_count + 1
                            }
                        else:
                            print("No input provided. Please choose again.")
                            continue
                            
                    else:
                        print("Invalid choice. Please enter 1, 2, 3, or 4.")
                        continue
                        
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user.")
                    decision = "end"
                    break
                except Exception as e:
                    print(f"Input error: {e}. Please try again.")
                    continue
            
            return {
                **input_state,
                "decision": decision,
                "loop_count": loop_count + 1
            }
            
        except Exception as e:
            print(f"Human feedback error: {e}")
            return {
                **input_state,
                "decision": "end",
                "loop_count": input_state.get("loop_count", 0) + 1
            }
    
    return RunnableLambda(get_human_feedback)

def route_decision(state: GraphState) -> str:
    decision = state.get("decision", "end")
    loop_count = state.get("loop_count", 0)
    research_count = state.get("research_count", 0)
    summarize_count = state.get("summarize_count", 0)
    
    if loop_count >= 10:
        return END
    
    if research_count >= 4 and summarize_count >= 4:
        return END
    
    if decision == "end":
        return END
    elif decision == "reresearch" and research_count < 4:
        return "researcher"
    elif decision == "resummarize" and summarize_count < 4:
        return "summarizer"
    elif decision == "human_feedback":
        return "human_feedback"
    else:
        return END

def build_graph():
    try:
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

        graph = builder.compile()
        return graph
        
    except Exception as e:
        print(f"Graph building error: {e}")
        raise

def main():
    try:
        print("RESEARCH AGENT")
        print("="*40)
        
        graph = build_graph()
        
        query = input("Enter your research query: ").strip()
        if not query:
            print("No query provided. Exiting.")
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
            "_critic_recommendation": ""
        }
        
        print(f"\nResearching: '{query}'")
        print("="*40)
        
        try:
            final_state = graph.invoke(initial_state, config={"recursion_limit": 25})
                
        except Exception as e:
            print(f"\nExecution error: {e}")
            return
        
        print("\nRESEARCH COMPLETE")
        print("="*40)
        
        print(f"\nQuery: {final_state.get('query', 'N/A')}")
        
        print(f"\nSummary:")
        print("-" * 30)
        summary = final_state.get('summary', 'No summary available')
        print(summary)
        
        print(f"\nStats:")
        print(f"Loops: {final_state.get('loop_count', 0)}")
        print(f"Research: {final_state.get('research_count', 0)}")
        print(f"Summarize: {final_state.get('summarize_count', 0)}")
        print(f"Decision: {final_state.get('decision', 'N/A')}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()