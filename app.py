import os
import json
from google import genai
from google.genai import types

# Initialize the Gemini Client 
# Make sure you set your environment variable in your terminal: export GEMINI_API_KEY="your-key"
client = genai.Client()

def load_knowledge_base():
    """Loads text from the data directory to serve as grounding context."""
    context_text = ""
    data_dir = "./data"
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith((".txt", ".md")):
                with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                    context_text += f"\n--- Source: {filename} ---\n" + f.read()
    return context_text if context_text else "Default SaaS Product Support Knowledge Base."

def detect_persona(user_message: str) -> str:
    """Classifies the user message into one of three mandatory personas."""
    prompt = f"""
    Analyze the following user support message and classify it into exactly ONE of these three personas:
    1. "Technical Expert" (Uses technical terms, asks for logs/APIs, wants details)
    2. "Frustrated User" (Uses emotional language, complains, demands urgent fixes)
    3. "Business Executive" (Outcome-focused, brief, asks about business/operational impact)

    User Message: "{user_message}"
    
    Respond with ONLY the persona name. Do not include any other text.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        detected = response.text.strip()
        for p in ["Technical Expert", "Frustrated User", "Business Executive"]:
            if p.lower() in detected.lower():
                return p
        return "Frustrated User"
    except Exception:
        return "Frustrated User"

def check_escalation_criteria(user_message: str, history: list) -> bool:
    """Configurable logic to flag sensitive topics or prolonged issues for human handoff."""
    sensitive_keywords = ["billing", "legal", "account-sensitive", "refund", "close account", "sue"]
    if any(kw in user_message.lower() for kw in sensitive_keywords):
        return True
    if len(history) >= 6: 
        return True
    return False

def generate_adaptive_response(user_message: str, persona: str, context: str) -> str:
    """Tailors response formatting and tone exactly to the user persona."""
    if persona == "Technical Expert":
        style_instruction = "Provide a detailed, technical response, including a step-by-step troubleshooting workflow and root cause analysis."
    elif persona == "Frustrated User":
        style_instruction = "Be highly empathetic, comforting, and reassuring. Use very simple language and clear action-oriented steps."
    else: 
        style_instruction = "Be exceptionally concise and outcome-focused. Minimize technical jargon and state the estimated resolution guidance cleanly."

    prompt = f"""
    You are an intelligent AI Customer Support Agent.
    
    [Context/Knowledge Base]:
    {context}
    
    [Persona Style Rule]:
    {style_instruction}
    
    [Constraint]:
    Answer the user query using ONLY the provided Context. Do not hallucinate information.
    
    User Query: {user_message}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_handoff_summary(persona: str, history: list, context_used: str) -> str:
    """Generates a structured JSON handoff summary for a human agent."""
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    
    prompt = f"""
    Generate a structured JSON handoff summary for a human representative based on this escalated session.
    
    Persona: {persona}
    History: {history_str}
    
    Return a valid JSON string matching this exact structure:
    {{
      "persona": "{persona}",
      "issue": "Brief summary of the issue",
      "documents_used": ["List of relevant topics encountered"],
      "attempted_steps": ["What the bot or user tried"],
      "recommendation": "Suggested action item for human agent"
    }}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return response.text.strip()
    except Exception:
        return json.dumps({
            "persona": persona,
            "issue": "System escalated issue",
            "documents_used": ["faq_core.md"],
            "attempted_steps": ["Automated response troubleshooting"],
            "recommendation": "Review standard support logs"
        }, indent=2)

def main():
    print("====================================================")
    print(" Persona-Adaptive Customer Support System Initialized")
    print("====================================================\n")
    
    kb_context = load_knowledge_base()
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Session closed.")
            break
            
        if not user_input.strip():
            continue
            
        persona = detect_persona(user_input)
        print(f"-> [Detected Persona]: {persona}")
        
        conversation_history.append({"role": "User", "content": user_input})
        
        if check_escalation_criteria(user_input, conversation_history):
            print("\n!!! [Status]: ESCALATING TO HUMAN SUPPORT !!!")
            summary = generate_handoff_summary(persona, conversation_history, kb_context)
            print("\n[Human Handoff Summary (JSON)]:")
            print(summary)
            print("\n====================================================")
            print("Conversation cleanly handed off. Exiting support track.")
            break
            
        bot_response = generate_adaptive_response(user_input, persona, kb_context)
        conversation_history.append({"role": "Agent", "content": bot_response})
        
        print(f"\nAgent:\n{bot_response}")
        print("\n----------------------------------------------------")

if __name__ == "__main__":
    main()
