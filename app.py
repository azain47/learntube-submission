import os
import uuid
import asyncio
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from backend.graph import build_graph
from backend.utils import create_profile_summary, scrape_linkedin_profile
from langchain_community.tools import DuckDuckGoSearchResults

st.set_page_config(page_title="LinkedIn Profile Optimizer", layout="wide", page_icon="🤖")
st.title("LinkedIn Profile Optimizer")

# --- Helper Functions ---

def get_llm(provider, api_key):
    if provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(model="gpt-4.1", temperature=0.3, streaming=True)
    elif provider == "Groq":
        os.environ["GROQ_API_KEY"] = api_key
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, streaming=True)
    return None

# --- Session State Initialization ---

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    llm_provider = st.selectbox("Choose LLM Provider", ["OpenAI","Groq"])
    api_key = st.text_input(f"Enter {llm_provider} API Key", type="password")

    st.header("Profile Data")
    profile_url = st.text_input("Enter LinkedIn URL")
    if st.button("Scrape & Analyze Profile"):
        if profile_url:
            llm = get_llm(llm_provider, api_key)
            if not llm:
                st.error("LLM not initialized. Check API key.")
            try:
                with st.spinner("Scraping LinkedIn profile..."):
                    # Run async function in event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    profile_data = loop.run_until_complete(scrape_linkedin_profile(profile_url))
                    loop.close()
                with st.spinner("Step 2/2: Creating profile summary to save tokens..."):
                        summary = create_profile_summary(llm, str(profile_data))
                        st.session_state.profile_summary = summary 
                        
                st.session_state.profile_data = profile_data
                st.session_state.messages = []  # Reset chat on new data
                st.success("Profile data loaded!")
                st.rerun()  # Refresh the app to show the new state
            except Exception as e:
                st.error(f"Scraping error: {str(e)}")
        else:
            st.warning("Please enter a LinkedIn URL")

if not api_key:
    st.info("Please enter your API key in the sidebar to begin.")
    st.stop()

# Initialize graph if not already done
if st.session_state.graph is None:
    llm = get_llm(llm_provider, api_key)
    # this tool doesnt work, issue in langchain.
    search = DuckDuckGoSearchResults(return_direct=True, num_results=1)
    tools = [search]
    st.session_state.graph = build_graph(llm)

def display_messages():
    """Display all messages in session state"""
    for msg in st.session_state.messages:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.write(msg.content)

# --- Chat Interface ---

if "profile_data" in st.session_state:
    # Initial analysis if chat is empty
    if not st.session_state.messages:
        config = {"configurable": {"thread_id": st.session_state.session_id}}        
        # Add initial message to session state first
        initial_prompt = HumanMessage(content="Analyze my profile by using the Profile Analyzer tool.")
        st.session_state.messages.append(initial_prompt)
        
        with st.chat_message("user"):
            st.write(initial_prompt.content)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            final_ai_response = ""
            
            with st.spinner("Analyzing profile..."):
                for chunk in st.session_state.graph.stream(
                    {"messages": [initial_prompt], "profile_data": st.session_state.profile_summary},
                    config=config
                ):
                    for agent_name, state in chunk.items():
                        if agent_name != "supervisor" and isinstance(state["messages"][-1], AIMessage):
                            # Get the latest content (not cumulative)
                            latest_content = state["messages"][-1].content
                            if latest_content and latest_content not in final_ai_response:
                                final_ai_response = latest_content
                                message_placeholder.write(final_ai_response)
            
            # Save the complete AI response to session state
            if final_ai_response:
                st.session_state.messages.append(AIMessage(content=final_ai_response))
    else:
        # Display existing conversation
        display_messages()

    # Handle new user input
    if prompt := st.chat_input("Ask a follow-up question..."):
        # Add user message and display it
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.write(prompt)
        
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            final_ai_response = ""
            
            with st.spinner("Thinking..."):
                for chunk in st.session_state.graph.stream(
                    {"messages": st.session_state.messages, "profile_data": st.session_state.profile_summary}, 
                    config=config
                ):
                    for agent_name, state in chunk.items():
                        if agent_name != "supervisor" and isinstance(state["messages"][-1], AIMessage):
                            latest_content = state["messages"][-1].content
                            if latest_content:
                                final_ai_response = latest_content
                                message_placeholder.write(final_ai_response)
                
                if final_ai_response:
                    st.session_state.messages.append(AIMessage(content=final_ai_response))
else:
    st.info("Please load profile data using the sidebar to start the conversation.")