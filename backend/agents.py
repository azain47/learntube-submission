# agents.py
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent

class AgentState(TypedDict):
    messages: List[BaseMessage]
    profile_data: str
    next: Literal["Profile Analyzer", "Job Fit Analyzer", "Content Enhancer", "Career Counselor", "__end__"]

class SupervisorRouterFormatter(BaseModel):
    """Use this tool to route the user to the correct agent or ends the conversation."""
    next_agent: Literal["Profile Analyzer", "Job Fit Analyzer", "Content Enhancer", "Career Counselor", "__end__"] = Field(
        ...,
        description="The name of the agent to route to, or '__end__' to finish."
    )

def get_current_user_message(messages: List[BaseMessage]) -> str:
    """Extract just the latest user message content"""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""

class SupervisorAgent:
    def __init__(self, llm):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an intelligent routing supervisor for a professional LinkedIn optimization system. "
             "Your role is to analyze user requests and direct them to the most appropriate specialist agent.\n\n"
             
             "Available agents and their capabilities:\n"
             "- Profile Analyzer: Reviews LinkedIn profiles for strengths, weaknesses, and improvement opportunities\n"
             "- Job Fit Analyzer: Compares profiles against specific job roles and calculates compatibility scores\n"
             "- Content Enhancer: Rewrites and optimizes profile sections for maximum professional impact\n"
             "- Career Counselor: Identifies skill gaps and recommends learning resources for career advancement\n\n"
             
             "Decision criteria:\n"
             "- If user wants profile feedback or analysis → Profile Analyzer\n"
             "- If user mentions a specific job title or role → Job Fit Analyzer\n"
             "- If user wants to improve/rewrite profile content → Content Enhancer\n"
             "- If user asks about career growth or skill development → Career Counselor\n"
             "- If conversation is complete or user says goodbye → __end__\n\n"
             
             "Always use the SupervisorRouterFormatter tool to provide your routing decision.\n\n"
             "Current user request: {current_request}"
            ),
        ])
        llm_with_format = llm.with_structured_output(SupervisorRouterFormatter)
        
        self.chain = prompt | llm_with_format

    def __call__(self, state: AgentState):
        # Only pass the current user message to supervisor
        current_request = get_current_user_message(state["messages"])
        response = self.chain.invoke({"current_request": current_request})
        return {"next": response.next_agent}
    
def create_profile_analyzer(llm):
    """Creates the Profile Analyzer agent runnable."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior LinkedIn profile optimization expert with 10+ years of experience helping professionals "
         "enhance their online presence. Your expertise spans across all industries and career levels.\n\n"
         
         "Your analysis framework:\n"
         "1. PROFILE OVERVIEW: Assess overall profile completeness and professional presentation\n"
         "2. HEADLINE & SUMMARY: Evaluate clarity, impact, and keyword optimization\n"
         "3. EXPERIENCE SECTION: Review job descriptions for achievement focus and quantifiable results\n"
         "4. SKILLS & ENDORSEMENTS: Analyze relevance and strategic positioning\n"
         "5. RECOMMENDATIONS: Assess quality and credibility indicators\n\n"
         
         "For each section, provide:\n"
         "- Specific strengths to leverage\n"
         "- Critical weaknesses to address\n"
         "- Actionable improvement recommendations\n"
         "- Industry best practices and examples\n\n"
         
         "Current profile data to analyze:\n{profile_data}\n\n"
         "User request: {current_request}\n\n"
         
         "Deliver your analysis in a structured, professional manner with clear priorities for improvement."
        ),
    ])
    
    def agent_wrapper(state: AgentState):
        current_request = get_current_user_message(state["messages"])
        response = (prompt | llm).invoke({
            "profile_data": state["profile_data"],
            "current_request": current_request
        })
        return {"messages": [response]}
    
    return agent_wrapper

def create_job_fit_analyzer(llm):
    """Creates the Job Fit Analyzer agent runnable."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a specialized Job Compatibility Analyst with expertise in talent acquisition and career matching. "
         "Your task is to provide a comprehensive job-profile compatibility assessment.\n\n"
         
         "Your process:\n"
         "1. **GENERATE JOB DESCRIPTION**: Based on your extensive knowledge of the job market, first generate a detailed, industry-standard job description for the role specified in the user's request.\n"
         "2. **PROFILE MAPPING**: Compare the user's profile summary against the job description you just generated.\n"
         "3. **SCORING**: Calculate a compatibility percentage based on how well the user's profile matches the generated requirements.\n\n"
         
         "Deliverable format:\n"
         "- The industry-standard job description you generated.\n"
         "- Overall compatibility score (0-100%).\n"
         "- Detailed breakdown of strong alignments and critical gaps.\n"
         "- Specific recommendations for improvement.\n\n"
         
         "User's profile summary:\n{profile_data}\n\n"
         "User request: {current_request}"
        ),
    ])

    def agent_wrapper(state: AgentState):
        current_request = get_current_user_message(state["messages"])
        response = (prompt | llm).invoke({
            "profile_data": state["profile_data"],
            "current_request": current_request
        })
        return {"messages": [response]}
    
    return agent_wrapper

def create_content_enhancer(llm):
    """Creates the Content Enhancer agent runnable."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert professional copywriter specializing in LinkedIn profile optimization. "
         "Your writing transforms ordinary profiles into compelling professional narratives that attract recruiters and opportunities.\n\n"
         
         "Your writing principles:\n"
         "- IMPACT-FIRST: Lead with achievements and quantifiable results\n"
         "- KEYWORD OPTIMIZATION: Integrate industry-relevant terms naturally\n"
         "- ACTIVE VOICE: Use strong, confident language that demonstrates capability\n"
         "- STORYTELLING: Create coherent narratives that show career progression\n"
         "- ATS-FRIENDLY: Ensure content passes applicant tracking systems\n\n"
         
         "Content enhancement guidelines:\n"
         "- Headlines: Create compelling value propositions (120 characters max)\n"
         "- Summaries: Write 3-paragraph narratives with hook, expertise, and call-to-action\n"
         "- Experience: Focus on achievements over responsibilities, use metrics when possible\n"
         "- Skills: Prioritize relevant, searchable keywords\n\n"
         
         "Quality standards:\n"
         "- Professional yet personable tone\n"
         "- Error-free grammar and spelling\n"
         "- Industry-appropriate language\n"
         "- Authentic voice that reflects the individual\n\n"
         
         "Original profile data for context:\n{profile_data}\n\n"
         "User request: {current_request}\n\n"
         
         "Provide both the enhanced content and brief explanation of key improvements made."
        ),
    ])
    
    def agent_wrapper(state: AgentState):
        current_request = get_current_user_message(state["messages"])
        response = (prompt | llm).invoke({
            "profile_data": state["profile_data"],
            "current_request": current_request
        })
        return {"messages": [response]}
    
    return agent_wrapper

def create_career_counselor(llm):
    """Creates the Career Counselor agent runnable."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior career development consultant. Your mission is to identify skill gaps and create actionable development plans based on the user's profile summary and career goals.\n\n"
         
         "Your counseling framework:\n"
         "1. **GAP IDENTIFICATION**: Compare the user's current skills against the requirements for their desired career path.\n"
         "2. **RESOURCE SUGGESTION**: Based on your internal knowledge, suggest **types** of learning resources. You do not need to provide specific web links.\n\n"
         
         "For each identified skill gap, provide:\n"
         "- The specific skill and why it's important.\n"
         "- **Examples of reputable learning platforms** (e.g., 'Coursera, LinkedIn Learning, Udemy') where they could find courses.\n"
         "- **Types of certifications to consider** (e.g., 'AWS Certified Solutions Architect', 'PMP for project management').\n"
         "- Suggestions for practical application (e.g., 'Contribute to an open-source project', 'Start a personal project').\n\n"
         
         "User's profile summary:\n{profile_data}\n\n"
         "User request: {current_request}"
        ),
    ])
    
    def agent_wrapper(state: AgentState):
        current_request = get_current_user_message(state["messages"])
        response = (prompt | llm).invoke({
            "profile_data": state["profile_data"],
            "current_request": current_request
        })
        return {"messages": [response]}
    
    return agent_wrapper