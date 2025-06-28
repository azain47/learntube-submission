import os
import httpx
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
APIFY_SCRAPER_ACTOR = "dev_fusion~linkedin-profile-scraper"
APIFY_API_URL = f"https://api.apify.com/v2/acts/{APIFY_SCRAPER_ACTOR}/run-sync-get-dataset-items"

import re
from typing import Dict, Any, Optional

async def scrape_linkedin_profile(linkedin_url: str) -> Dict[str, Any]:
    """
    Calls Apify's LinkedIn Scraper API and returns validated profile data.
    Handles errors and validates LinkedIn URL format.
    """
    # Validate LinkedIn URL format
    if not re.match(r'^https?://(www\.)?linkedin\.com/in/[a-zA-Z0-9-]+/?$', linkedin_url):
        raise ValueError("Invalid LinkedIn profile URL format")

    if not APIFY_API_TOKEN:
        raise RuntimeError("APIFY_API_TOKEN environment variable not set.")
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {APIFY_API_TOKEN}'
    }
    
    payload = {
        "profileUrls": [linkedin_url],
        "maxRetries": 3,
        "timeoutSecs": 90
    }
    
    timeout = httpx.Timeout(120.0) 
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(APIFY_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            # Validate response structure
            if not isinstance(data, list) or len(data) == 0:
                raise ValueError("Unexpected response format from Apify API")
                
            profile = data[0]

            return profile
        except httpx.HTTPStatusError as e:
            error_msg = f"Apify API error: {e.response.status_code} {e.response.text}"
            raise RuntimeError(error_msg) from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error: {str(e)}") from e
          
def create_profile_summary(llm, profile_data: str) -> str:
    """
    Uses an LLM to create a dense, structured summary of the raw profile data.
    This summary is then used in all subsequent prompts to save tokens.

    Args:
        llm: The language model instance to use for summarization.
        profile_data: The raw JSON string of the scraped LinkedIn profile.

    Returns:
        A string containing the structured summary of the profile.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert LinkedIn Profile Summarizer with extensive experience in talent assessment and "
         "professional profile analysis. Your role is to create comprehensive, structured summaries that "
         "capture the essence of a professional's background, skills, and career trajectory.\n\n"
         
         "PROFILE SUMMARIZATION FRAMEWORK:\n\n"
         
         "1. PROFESSIONAL IDENTITY\n"
         "   - Extract: fullName, headline, jobTitle\n"
         "   - Create a compelling professional identity statement\n"
         "   - Include current role and key value proposition\n\n"
         
         "2. CONTACT & LOCATION\n"
         "   - Summarize: addressWithCountry, email, mobileNumber, linkedinUrl\n"
         "   - Present professional contact information clearly\n"
         "   - Note location and availability for remote/relocation\n\n"
         
         "3. CAREER OVERVIEW\n"
         "   - Analyze: experiences, currentJobDuration, currentJobDurationInYrs\n"
         "   - Summarize career progression and key roles\n"
         "   - Highlight leadership positions and career growth\n"
         "   - Calculate total experience and current role tenure\n\n"
         
         "4. CURRENT COMPANY CONTEXT\n"
         "   - Extract: companyName, companyIndustry, companySize, companyFoundedIn\n"
         "   - Provide context about current employer\n"
         "   - Include industry, company scale, and market position\n\n"
         
         "5. EDUCATIONAL BACKGROUND\n"
         "   - Summarize: educations\n"
         "   - List degrees, institutions, and relevant academic achievements\n"
         "   - Include any specialized training or coursework\n\n"
         
         "6. CORE COMPETENCIES\n"
         "   - Analyze: skills, topSkillsByEndorsements\n"
         "   - Categorize skills by type (technical, leadership, domain expertise)\n"
         "   - Prioritize most endorsed and relevant skills\n"
         "   - Include skill strength indicators\n\n"
         
         "7. CREDENTIALS & CERTIFICATIONS\n"
         "   - Extract: licenseAndCertificates, courses, testScores\n"
         "   - List professional certifications and their relevance\n"
         "   - Include completion dates and issuing organizations\n\n"
         
         "8. ACHIEVEMENTS & RECOGNITION\n"
         "   - Compile: honorsAndAwards, highlights, recommendations\n"
         "   - Summarize notable achievements and recognition\n"
         "   - Include peer recommendations and endorsements\n\n"
         
         "9. ADDITIONAL ACTIVITIES\n"
         "   - Review: projects, publications, patents, volunteerAndAwards\n"
         "   - Highlight relevant side projects and contributions\n"
         "   - Include volunteer work and community involvement\n\n"
         
         "10. NETWORK & INFLUENCE\n"
         "    - Analyze: connections, followers, openConnection\n"
         "    - Assess professional network size and engagement\n"
         "    - Note thought leadership indicators\n\n"
         
         "SUMMARY OUTPUT STRUCTURE:\n\n"
         "**EXECUTIVE SUMMARY** (2-3 sentences)\n"
         "- Professional identity and current role\n"
         "- Key value proposition and expertise areas\n"
         "- Career level and industry focus\n\n"
         
         "**PROFESSIONAL BACKGROUND**\n"
         "- Career progression summary\n"
         "- Current role and company context\n"
         "- Total experience and specialization areas\n\n"
         
         "**KEY STRENGTHS**\n"
         "- Top 5-7 core competencies\n"
         "- Technical and soft skills highlights\n"
         "- Industry-specific expertise\n\n"
         
         "**CREDENTIALS**\n"
         "- Educational background\n"
         "- Professional certifications\n"
         "- Notable achievements and awards\n\n"
         
         "**PROFILE STRENGTH INDICATORS**\n"
         "- Network size and engagement level\n"
         "- Profile completeness score\n"
         "- Professional activity and thought leadership\n\n"
         
         "QUALITY STANDARDS:\n"
         "- Use professional, objective language\n"
         "- Focus on quantifiable achievements when available\n"
         "- Highlight unique value propositions\n"
         "- Maintain consistency in tone and structure\n"
         "- Ensure summary is ATS-friendly and keyword-rich\n"
         "- Keep each section concise but comprehensive\n\n"
         
         "SPECIAL HANDLING INSTRUCTIONS:\n"
         "- If any field is empty or null, skip gracefully without mentioning gaps\n"
         "- Prioritize most recent and relevant information\n"
         "- Use industry-standard terminology and abbreviations\n"
         "- Maintain professional confidentiality (don't include personal details unnecessarily)\n"
         "- Focus on career-relevant information over personal interests\n\n"
         
         "Raw profile data to summarize:\n{profile_data}\n\n"
         
         "Create a comprehensive, well-structured summary that captures this professional's "
         "career story, core competencies, and unique value proposition."
        ),
    ])
    
    summarization_chain = prompt | llm
    
    # Invoke the chain to get the AIMessage object containing the summary
    response_message = summarization_chain.invoke({"profile_data": profile_data})
    
    # Return the content of the AIMessage
    return response_message.content