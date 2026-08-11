"""
tutor.py — LLM tutor prompt assembly and invocation via Gemini.
"""

import google.generativeai as genai
from .models import TutoringContext, TutoringResponse
from .config import GEMINI_API_KEY, GEMINI_MODEL, MAX_TUTOR_RESPONSE_WORDS

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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

    return f"""You are the AI tutor inside Synapse, an adaptive learning platform for Class 9 Science students (NCERT curriculum, ages 13-15).

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
1. If the answer was wrong (which it likely is if they are getting tutored), name what specifically went wrong in one sentence — tie it to the misconception tag if one is given. Do not just say "that's incorrect."
2. Re-teach {context.concept_name} using ONLY the retrieved textbook context above. If the context doesn't fully cover something you need, say the textbook doesn't cover it rather than inventing an explanation.
3. Use vocabulary appropriate for a 14-year-old — short sentences, concrete analogies from everyday life, no jargon without immediately defining it.
4. Adjust depth to theta: since the student has {ability} ability, if theta is low, use a worked example and go slower; if theta is high but they still slipped, keep it brief and precise.
5. End with ONE guiding question or hint that nudges the student toward the right idea — never give away the answer to a question they haven't attempted yet.

RULES
- Never fabricate facts not present in the retrieved context.
- Never solve the next practice question for them outright.
- Keep the whole response under ~{MAX_TUTOR_RESPONSE_WORDS} words.
- Warm, encouraging tone — this is a student who just got something wrong, not a peer reviewer.
"""

def generate_tutoring_response(context: TutoringContext) -> TutoringResponse:
    """Generate a tutoring response using the Gemini API."""
    chunk_ids = [f"{rc.chunk.concept_id}_{rc.chunk.chunk_index}" for rc in context.retrieved_chunks]
    
    if not GEMINI_API_KEY:
        # Fallback for when API key is missing
        return TutoringResponse(
            response_text="[LLM API Key missing] AI Tutor suggests you review the retrieved context to understand the misconception.",
            retrieved_chunk_ids=chunk_ids,
            turn_number=context.turn_number
        )
        
    prompt = _build_prompt(context)
    
    try:
        # Use genai model
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        
        # We can also use safety settings if needed, but defaults are fine
        response = model.generate_content(prompt)
        response_text = response.text.strip()
    except Exception as e:
        # Graceful fallback if the LLM call fails (bad key, rate limit, blocked content, etc.)
        print(f"WARNING: Gemini API call failed: {e}")
        
        # Build a fallback response from the retrieved chunks
        fallback_parts = []
        for rc in context.retrieved_chunks:
            fallback_parts.append(rc.chunk.text)
        
        if fallback_parts:
            response_text = (
                f"I noticed you're having trouble with {context.concept_name}. "
                f"Here's what the textbook says:\n\n"
                + "\n\n".join(fallback_parts[:2])
                + "\n\nTry reviewing this and see if it helps clarify the concept!"
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

