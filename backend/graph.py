from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from .agents import (
    AgentState, SupervisorAgent, create_profile_analyzer, 
    create_job_fit_analyzer, create_content_enhancer, create_career_counselor
)

def build_graph(llm):
    """
    Builds the multi-agent graph with a manual supervisor, explicit routing,
    and a robust tool-handling loop.
    """
    # Create our agent runnables and supervisor
    profile_analyzer_runnable = create_profile_analyzer(llm)
    job_fit_analyzer_runnable = create_job_fit_analyzer(llm)
    content_enhancer_runnable = create_content_enhancer(llm)
    career_counselor_runnable = create_career_counselor(llm)
    supervisor = SupervisorAgent(llm)
    
    # --- Graph Definition ---
    workflow = StateGraph(AgentState)

    # Add the nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("Profile Analyzer", (profile_analyzer_runnable))
    workflow.add_node("Job Fit Analyzer", (job_fit_analyzer_runnable))
    workflow.add_node("Content Enhancer", (content_enhancer_runnable))
    workflow.add_node("Career Counselor", (career_counselor_runnable))

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "Profile Analyzer": "Profile Analyzer",
            "Job Fit Analyzer": "Job Fit Analyzer",
            "Content Enhancer": "Content Enhancer",
            "Career Counselor": "Career Counselor",
            "__end__": END,
        },
    )

    workflow.add_edge("Profile Analyzer", END)
    workflow.add_edge("Content Enhancer", END)
    workflow.add_edge("Job Fit Analyzer", END)
    workflow.add_edge("Career Counselor", END)

    # Compile the graph
    conn = sqlite3.connect('checkpoints.sqlite', check_same_thread=False)
    mem = SqliteSaver(conn=conn)
    graph = workflow.compile(checkpointer=mem)
    
    return graph