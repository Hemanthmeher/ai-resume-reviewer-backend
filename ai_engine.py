import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Global variable to store resume in memory for the session
RESUME_MEMORY = {"text": None}

def store_resume(resume_text):
    """Store resume in memory for future questions"""
    RESUME_MEMORY["text"] = resume_text

def get_stored_resume():
    """Get stored resume from memory"""
    return RESUME_MEMORY["text"]

def analyze_resume(resume_text, question):
    """
    Analyze resume comprehensively based on user question.
    Now with memory - resume persists across questions.
    """
    try:
        # If new resume provided, store it
        if resume_text and resume_text.strip():
            store_resume(resume_text)
        
        # Use stored resume if no new resume provided
        current_resume = resume_text if resume_text else get_stored_resume()

        # Check if API key exists
        if not GROQ_API_KEY:
            return {
                "answer": "⚠️ GROQ_API_KEY not found. Please check your .env file."
            }

        # Keywords to detect FULL ANALYSIS request
        full_analysis_keywords = [
            'full analysis', 'complete analysis', 'full review', 'complete review',
            'analyze my resume', 'review my resume', 'check my resume',
            'evaluate my resume', 'assess my resume', 'examine my resume',
            'overall analysis', 'detailed analysis', 'comprehensive analysis'
        ]
        
        # Keywords to detect PROJECT/SKILL recommendation requests
        recommendation_keywords = [
            'recommend', 'suggestion', 'project', 'skill to learn', 
            'skill to focus', 'what should i learn', 'improve my skills'
        ]
        
        question_lower = question.lower().strip()
        
        # Extract number from question if user specifies count
        import re
        number_match = re.search(r'\b(\d+)\b', question)
        requested_count = int(number_match.group(1)) if number_match else None
        
        # Check what type of request this is
        wants_full_analysis = any(keyword in question_lower for keyword in full_analysis_keywords)
        wants_recommendations = any(keyword in question_lower for keyword in recommendation_keywords)

        # ============================================
        # CASE 1: NO RESUME - Just chat/general advice
        # ============================================
        if not current_resume or not current_resume.strip():
            prompt = f"""You are a helpful AI assistant.

USER QUESTION: {question}

Respond ONLY with valid JSON in this exact format:
{{
  "answer": "Your helpful response to their question in 2-4 sentences"
}}

IMPORTANT: Return ONLY valid JSON, nothing else."""

            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 500
            }

            response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                return {"answer": f"API Error: {response.status_code}. Please try again."}

            result = response.json()
            if "choices" not in result or len(result["choices"]) == 0:
                return {"answer": "API returned invalid response. Please try again."}

            content = result["choices"][0]["message"]["content"].strip()
            
            # Clean markdown
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            try:
                parsed = json.loads(content)
                return {"answer": parsed.get("answer", content)}
            except:
                return {"answer": content if len(content) < 500 else "I couldn't understand that. Please try again."}

        # ============================================
        # CASE 2: RESUME EXISTS + FULL ANALYSIS REQUEST
        # ============================================
        if wants_full_analysis:
            if requested_count:
                count_instruction = f"USER REQUESTED EXACTLY {requested_count} ITEMS. Provide EXACTLY {requested_count} items in the relevant arrays based on what they asked for."
            else:
                count_instruction = "Provide 4-6 items in each array."
            
            prompt = f"""You are an expert ATS system and senior technical recruiter.

RESUME:
{current_resume}

USER REQUEST: {question}

{count_instruction}

CRITICAL INSTRUCTIONS:
1. Understand their CURRENT skills, experience level, and career goals from the resume
2. Identify GAPS in their resume (missing skills for their field)
3. Match projects and skills to THEIR resume, not generic advice
4. Each array item should be ONE clear point, not a paragraph

Respond ONLY with valid JSON in this exact format:
{{
  "answer": "Professional 3-4 sentence overall assessment of resume quality and readiness",
  "ats_score": 75,
  "strengths": ["Clear point 1", "Clear point 2", "Clear point 3", "Clear point 4"],
  "weaknesses": ["Clear point 1", "Clear point 2", "Clear point 3", "Clear point 4"],
  "improvement_suggestions": ["Clear point 1", "Clear point 2", "Clear point 3", "Clear point 4", "Clear point 5"],
  "recommended_projects": ["Project 1", "Project 2", "Project 3"],
  "skills_to_focus": ["Skill 1", "Skill 2", "Skill 3", "Skill 4", "Skill 5"]
}}

IMPORTANT: Return ONLY valid JSON, nothing else."""

        # ============================================
        # CASE 3: RESUME EXISTS + RECOMMENDATION REQUEST
        # ============================================
        elif wants_recommendations:
            if requested_count:
                if 'project' in question_lower:
                    count_instruction = f"Provide EXACTLY {requested_count} projects in recommended_projects array."
                elif 'skill' in question_lower:
                    count_instruction = f"Provide EXACTLY {requested_count} skills in skills_to_focus array."
                else:
                    count_instruction = f"Provide EXACTLY {requested_count} items in both arrays."
            else:
                count_instruction = "Provide 3 projects and 3-5 skills."
            
            prompt = f"""You are an expert career advisor.

RESUME:
{current_resume}

USER REQUEST: {question}

{count_instruction}

IMPORTANT INSTRUCTIONS:
1. Analyze the resume to understand their CURRENT skills and experience level
2. Read the user's question carefully to understand WHAT THEY WANT
3. Recommend ONLY projects and skills that:
   - Match their current skill level (not too easy, not too hard)
   - Are RELEVANT to what they asked for
   - Build upon skills they already have in their resume
   - Fill gaps in their resume for their target career path
4. Each item should be ONE clear point, not a paragraph

Respond ONLY with valid JSON in this exact format:
{{
  "answer": "Detailed 3-4 sentence answer addressing their specific question, explain WHY you're recommending these based on their resume",
  "recommended_projects": ["Project 1", "Project 2", "Project 3"],
  "skills_to_focus": ["Skill 1", "Skill 2", "Skill 3"]
}}

BE SPECIFIC - Don't give generic advice. Match to THEIR resume and THEIR question.
IMPORTANT: Return ONLY valid JSON, nothing else."""

        # ============================================
        # CASE 4: RESUME EXISTS + SPECIFIC QUESTION
        # ============================================
        else:
            # Check if user is asking for numbered items
            if requested_count and any(word in question_lower for word in ['positive', 'negative', 'strength', 'weakness', 'good', 'bad', 'improvement', 'suggestion', 'thing', 'idea']):
                prompt = f"""You are an expert resume analyst.

RESUME:
{current_resume}

USER QUESTION: {question}

THE USER ASKED FOR EXACTLY {requested_count} ITEMS.

CRITICAL: Format your answer as a NUMBERED LIST with EXACTLY {requested_count} bullet points.

Respond ONLY with valid JSON in this exact format:
{{
  "answer": "1. First point here\\n2. Second point here\\n3. Third point here\\n... (continue until {requested_count})"
}}

RULES:
- Each point should be ONE clear sentence
- Number each point (1., 2., 3., etc.)
- Use \\n between points for new lines
- Provide EXACTLY {requested_count} numbered points
- Reference actual content from their resume

IMPORTANT: Return ONLY valid JSON, nothing else."""
            else:
                prompt = f"""You are an expert resume analyst.

Answer this specific question about the resume:

RESUME:
{current_resume}

USER QUESTION: {question}

Respond ONLY with valid JSON in this exact format:
{{
  "answer": "Detailed 3-5 sentence answer to their specific question, referencing actual content from their resume"
}}

Be specific and reference actual details from the resume. IMPORTANT: Return ONLY valid JSON, nothing else."""

        # ============================================
        # CALL GROQ API
        # ============================================
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2000
        }

        print(f"🔄 Calling Groq API... (Count: {requested_count}, Full Analysis: {wants_full_analysis}, Recommendations: {wants_recommendations})")
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code != 200:
            error_msg = f"Groq API Error {response.status_code}: {response.text}"
            print(f"❌ {error_msg}")
            return {"answer": f"API Error: {response.status_code}. Please check your GROQ_API_KEY."}

        result = response.json()

        if "choices" not in result or len(result["choices"]) == 0:
            print("❌ No choices in response")
            return {"answer": "API returned invalid response. Please try again."}

        content = result["choices"][0]["message"]["content"]
        print(f"📝 Raw Content: {content[:200]}...")
        
        # Clean the content
        content = content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        content = content.strip()
        print(f"🧹 Cleaned Content: {content[:200]}...")

        # Try to parse JSON
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"📄 Full Content: {content}")
            return {"answer": content if len(content) < 500 else "I received a response but couldn't parse it. Please try again."}

        # Ensure default fields based on request type
        if wants_full_analysis:
            defaults = {
                "answer": "",
                "ats_score": None,
                "strengths": [],
                "weaknesses": [],
                "improvement_suggestions": [],
                "recommended_projects": [],
                "skills_to_focus": []
            }
        elif wants_recommendations:
            defaults = {
                "answer": "",
                "recommended_projects": [],
                "skills_to_focus": []
            }
        else:
            defaults = {"answer": ""}
        
        for key, value in defaults.items():
            if key not in parsed:
                parsed[key] = value

        print("✅ Successfully parsed JSON response")
        return parsed

    except requests.exceptions.Timeout:
        print("❌ Request timeout")
        return {"answer": "Request timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {str(e)}")
        return {"answer": "Network error occurred. Please try again."}
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"answer": "An unexpected error occurred. Please try again."}