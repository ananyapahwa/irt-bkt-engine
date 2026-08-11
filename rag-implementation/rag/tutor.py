"""
tutor.py — LLM tutor prompt assembly and invocation via Ollama.
"""

from typing import List, Dict, Optional
import requests
from .models import TutoringContext, TutoringResponse
from .config import MAX_TUTOR_RESPONSE_WORDS, OLLAMA_HOST, OLLAMA_MODEL

def _build_prompt(context: TutoringContext) -> str:
    retrieved_text = ""
    for r_chunk in context.retrieved_chunks:
        retrieved_text += f"\n--- [Source: {r_chunk.chunk.source}, Part {r_chunk.chunk.chunk_index+1}] ---\n"
        retrieved_text += f"{r_chunk.chunk.text}\n"

    # Convert theta to descriptive ability level for the LLM prompt
    if context.theta < -1.0:
        ability = "low (struggling)"
    elif context.theta > 1.0:
        ability = "high (advanced)"
    else:
        ability = "average"
        
    mastery_pct = round(context.mastery_probability * 100, 1)

    return f"""You are the AI tutor inside Synapse, an adaptive learning platform for Class 9 Science students (NCERT curriculum, ages 13-15). You employ the Socratic method to guide students to the correct answer rather than just giving it to them.

CONTEXT FOR THIS TURN
- Concept: {context.concept_name} (id: {context.concept_id})
- Student mastery estimate: {mastery_pct}% (BKT probability)
- Student ability estimate (theta): {context.theta:.2f} ({ability} ability)
- Student just answered: {context.student_answer}
- Correct answer: {context.correct_answer}
- Misconception tag: {context.misconception_tag if context.misconception_tag else 'None'}
- Turn number: {context.turn_number}

Retrieved textbook context (grounding — use only this, do not add outside facts):
{retrieved_text}

YOUR JOB, IN ORDER
1. Acknowledge the student's attempt. If a misconception tag is provided, briefly identify the flaw in their reasoning based on their selected answer without being harsh.
2. DO NOT just explain the concept or give away why the correct answer is correct. 
3. Use the retrieved textbook context to formulate a single, focused guiding question or hint that prompts the student to rethink their approach. 
4. Use vocabulary appropriate for a 14-year-old — short sentences, concrete analogies from everyday life, no jargon without immediately defining it.
5. Adjust depth to theta: since the student has {ability} ability, if theta is low, make the hint more concrete and step-by-step; if theta is high, keep the hint brief and conceptual.

RULES
- NEVER give the direct answer or fully solve the problem for them.
- Never fabricate facts not present in the retrieved context.
- Keep the whole response under ~{MAX_TUTOR_RESPONSE_WORDS} words.
- Warm, encouraging tone — this is a student who just got something wrong, act as a supportive guide.
"""

def generate_tutoring_response(context: TutoringContext) -> TutoringResponse:
    """Generate a tutoring response using the Gemini API."""
    chunk_ids = [f"{rc.chunk.concept_id}_{rc.chunk.chunk_index}" for rc in context.retrieved_chunks]
    
    if not OLLAMA_HOST:
        # Fallback for when Ollama is missing
        return TutoringResponse(
            response_text="[Ollama Config missing] AI Tutor suggests you review the retrieved context.",
            retrieved_chunk_ids=chunk_ids,
            turn_number=context.turn_number
        )
        
    prompt = _build_prompt(context)
    
    try:
        # Call local Ollama API
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        result_json = response.json()
        response_text = result_json.get("response", "").strip()
    except Exception as e:
        # Graceful fallback if the LLM call fails (Ollama not running, etc.)
        print(f"WARNING: Ollama API call failed: {e}")
        
        # Build a fallback response from the retrieved chunks
        fallback_parts = []
        for rc in context.retrieved_chunks:
            fallback_parts.append(rc.chunk.text)
        
        if fallback_parts:
            # Create a more Socratic-style fallback instead of just dumping the textbook
            response_text = (
                f"Let's review {context.concept_name} together. "
                f"The textbook mentions that: '{fallback_parts[0][:150]}...' "
                f"Based on this rule, how would you approach the problem differently? "
                f"Think about what happens to the other variables when one changes."
            )
        else:
            response_text = (
                f"Let's review {context.concept_name} together. "
                f"The correct answer was: {context.correct_answer}. "
                f"Try thinking about what makes this answer right."
            )
    
    return TutoringResponse(
        response_text=response_text,
        retrieved_chunk_ids=chunk_ids,
        turn_number=context.turn_number
    )

