# Persona-Aware Customer Support Agent

An intelligent, persona-adaptive customer support automation system built with Python, utilizing Gemini for classification and adaptive generation, combined with custom rule-based human escalation tracking.

## Tech Stack
* **Language**: Python 3.11+
* **LLM Core**: Google Gemini API (`google-genai`)
* **Knowledge Base**: Text and Markdown flat-file vector context
* **Interface**: Interactive Command-Line Interface (CLI)

## Architecture Workflow
1. **User Input** -> Read message string via interactive loop.
2. **Persona Detection** -> Zero-shot classification into *Technical Expert*, *Frustrated User*, or *Business Executive*.
3. **Escalation Verification** -> Evaluates rule-based triggers (e.g., account-sensitive keywords or session length defaults).
4. **Context Synthesis** -> Ingests flat-file data context securely to ground response patterns.
5. **Adaptive Response** -> Tailors linguistic formatting (detailed code walkthroughs vs. brief operational milestones) based on classification metrics.
6. **Human Handoff** -> Outputs a structured JSON object capturing user details upon escalation.

## Execution Guide
1. Install dependencies:
   ```bash
   pip install google-genai
