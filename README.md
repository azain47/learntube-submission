# LearnTube LinkedIn Profile Optimizer – Take Home Assessment

## Objective

As per the requirements, the assessment has been completed, and is ready to help users optimize their LinkedIn profiles, analyze job fit, and receive personalized career guidance. The system processes LinkedIn profile data, suggests improvements, and generates job-specific recommendations, all through a conversational chat interface with memory capabilities.

---

## Features

- **Interactive Chat Interface**  
  - Users input their LinkedIn profile URL.
  - The system scrapes the profile (using Apify’s LinkedIn Scraper) and provides feedback via a chat interface.
  - The assistant guides users through profile optimization, job fit analysis, and career recommendations.

- **Profile Optimization, Job Fit Analysis & Career Guidance**  
  - **Profile Analysis:** Evaluates LinkedIn sections (About, Experience, Skills, etc.), identifies gaps, and suggests improvements.
  - **Job Fit Analysis:** Users specify target job roles; the system generates an industry-standard job description, compares it to the user’s profile, calculates a match score, and recommends improvements.
  - **Content Enhancement:** Rewrites profile sections for better alignment with industry standards and job requirements.
  - **Career Counseling & Skill Gap Analysis:** Identifies missing skills for target roles and suggests learning resources or career paths.

- **Memory System for Personalized Experience**  
  - Maintains session-based and persistent memory using LangGraph’s checkpointers.
  - Retains context across multiple user queries for a seamless experience.

---

## Architecture & Approach

- **Frontend:**  
  - Built with [Streamlit](https://streamlit.io/) for rapid prototyping and interactive UI.
  - Sidebar for configuration (LLM provider, API keys, LinkedIn URL input).
  - Main area for chat-based interaction and feedback.

- **Backend:**  
  - **Multi-Agent System:**  
    - Implemented using [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain).
    - Agents include:  
      - **Supervisor Agent:** Routes user requests to the appropriate specialist.
      - **Profile Analyzer:** Evaluates profile strengths and weaknesses.
      - **Job Fit Analyzer:** Compares profiles with job requirements.
      - **Content Enhancer:** Improves profile content.
      - **Career Counselor:** Provides career guidance and skill gap analysis.
  - **Profile Scraping:**  
    - Uses Apify’s LinkedIn Scraper API (free credits available).
  - **Memory:**  
    - Session and persistent memory via LangGraph’s SQLite checkpointer.

- **Prompt Engineering:**  
  - Carefully crafted prompts for each agent to ensure high-quality, actionable AI responses.

---

## Setup & Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/azain47/learntube-submission.git
    cd learntube
    ```

2. **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**  
   It is recommended to use [uv](https://github.com/astral-sh/uv) for faster and more reliable dependency installation:
    ```bash
    pip install uv
    uv pip install -r requirements.txt
    ```
   Or, you can use pip directly:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Environment Variables**
    - Create a `.env` file in the project root:
      ```env
      APIFY_API_TOKEN=your_apify_api_token
      ```

---

## Running the Application

1. **Start the Streamlit App**
    ```bash
    streamlit run app.py
    ```
    Or if using `uv`:
    ```bash
    uv run streamlit run app.py
    ```

2. **Usage**
    - Open the app in your browser.
    - Use the sidebar to select an LLM provider, enter your API key, and input a LinkedIn profile URL.
    - Interact with the chat interface for analysis, job fit evaluation, content enhancement, and career advice.

---

## File Structure

```
learntube/
├── app.py               # Main Streamlit application (UI)
├── backend/
│   ├── agents.py        # Agent implementations (Profile Analyzer, Job Fit, etc.)
│   ├── graph.py         # LangGraph workflow definition
│   └── utils.py         # Utility functions (scraping, summarization)
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Approach, Challenges & Solutions

- **Approach:**  
  - Modularized agents for each core functionality.
  - Used LangGraph for agent orchestration and memory.
  - Streamlit for rapid UI development and chat interface.

- **Challenges:**  
  - **Token Management:** The most challenging aspect was managing token usage, due to LangGraph's unpredictable agent routing and prompt submission. This sometimes led to higher-than-expected token consumption, requiring careful prompt design and summary steps to minimize usage.
  - Integrating multiple LLM providers and handling API key management.
  - Ensuring robust error handling for external API calls (e.g., Apify).
  - Designing prompts that yield actionable, structured feedback.

- **Solutions:**  
  - Used environment variables and sidebar inputs for flexible configuration.
  - Implemented try/except blocks and user feedback for error cases.
  - Iteratively refined prompts and agent logic for clarity and relevance.
  - Added a profile summary step to reduce token usage before sending data to the agents.

---

## Hosting

**Note:**  
The application is not currently hosted, as I do not have API credits left for the required services. Please run locally with your own API keys.

---

## Submission Checklist

- [x] Hosted Streamlit application (can be run locally)
- [x] Complete source code with clear structure and comments
- [x] `requirements.txt` for dependencies
- [x] This README with setup, usage, and architecture details
- [x] Documented approach, challenges, and solutions

