import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# Also add irt-engine to path so we can import from irt module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'irt-engine'))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any

from examples.mock_orchestrator import BKTOrchestrator, IRTItemBank, IRTStudentProfile, ConceptIRTInfo, PracticeQuestion
from irt.theta import estimate_theta, AnswerRecord, QuestionIRTParameters
from irt.mastery_initializer import initialize_mastery, ConceptAttempt
from irt.bloom_mapper import difficulty_for

app = FastAPI(title="BKT & IRT Interactive Quiz Engine")

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Load questions from JSON
QUESTIONS_FILE = os.path.join(os.path.dirname(__file__), "questions.json")
try:
    with open(QUESTIONS_FILE, "r") as f:
        questions_data = json.load(f)
except Exception as e:
    print(f"Error loading {QUESTIONS_FILE}: {e}")
    questions_data = []

# Build item bank
items = {}
for q in questions_data:
    try:
        b = difficulty_for(q.get("bloom", "understand"))
    except:
        b = 0.0 # fallback
    items[q["id"]] = (1.0, b) # default discrimination a=1.0

item_bank = IRTItemBank(items=items)

@app.get("/api/quiz")
def get_quiz():
    return questions_data

class AnswerSubmit(BaseModel):
    question_id: str
    is_correct: bool
    response_time_ms: int

class QuizSubmitRequest(BaseModel):
    student_id: str = "student_live"
    diagnostic_answers: List[AnswerSubmit]
    practice_answers: List[AnswerSubmit]

@app.post("/api/quiz/submit")
def submit_quiz(req: QuizSubmitRequest):
    if not req.diagnostic_answers:
        raise HTTPException(status_code=400, detail="No diagnostic answers provided")
        
    # 1. Prepare IRT parameters and records for Theta estimation
    irt_params = []
    for a_record in req.diagnostic_answers:
        a, b = item_bank.get(a_record.question_id)
        irt_params.append(QuestionIRTParameters(
            question_id=a_record.question_id,
            discrimination=a,
            difficulty=b
        ))
        
    responses = [AnswerRecord(a.question_id, a.is_correct) for a in req.diagnostic_answers]
    
    # 2. Estimate Theta
    theta_result = estimate_theta(responses, irt_params)
    
    # 3. Initialize Mastery for all concepts seen in diagnostic
    # We need to build concept_attempts
    concept_attempts = []
    q_dict = {q["id"]: q for q in questions_data}
    
    for a in req.diagnostic_answers:
        q_info = q_dict.get(a.question_id)
        if not q_info: continue
        concept_attempts.append(ConceptAttempt(
            concept_id=q_info["concept_id"],
            question_id=a.question_id,
            is_correct=a.is_correct,
            bloom_level=q_info.get("bloom", "understand")
        ))
        
    mastery_res = initialize_mastery(req.student_id, theta_result, concept_attempts)
    
    # 4. Prepare BKT Orchestrator Inputs
    student_profile = IRTStudentProfile(
        student_id=req.student_id,
        theta=theta_result.theta,
        theta_se=theta_result.standard_error or 0.5,
        converged=theta_result.converged
    )
    
    concept_info = {}
    for cid, mastery in mastery_res.concept_masteries.items():
        # Get all b-values for items in this concept that were in diagnostic
        b_vals = []
        for a in req.diagnostic_answers:
            if q_dict[a.question_id]["concept_id"] == cid:
                b_vals.append(item_bank.get(a.question_id)[1])
                
        concept_info[cid] = ConceptIRTInfo(
            concept_id=cid,
            item_difficulties=b_vals,
            n_attempted_diagnostic=mastery.n_attempted,
            n_correct_diagnostic=mastery.n_correct
        )
        
    # 5. Run BKT Orchestrator on practice answers
    orchestrator = BKTOrchestrator(
        student=student_profile,
        item_bank=item_bank,
        concept_info=concept_info
    )
    
    initial_p_l0 = {cid: state.p_l for cid, state in orchestrator.states.items()}
    history = []
    
    for i, p_ans in enumerate(req.practice_answers, 1):
        # We need concept_id for PracticeQuestion
        q_info = q_dict.get(p_ans.question_id)
        if not q_info: continue
        
        pq = PracticeQuestion(
            question_id=p_ans.question_id,
            concept_id=q_info["concept_id"],
            response_time_ms=p_ans.response_time_ms,
            is_correct=p_ans.is_correct
        )
        
        result = orchestrator.process_answer(pq)
        
        history.append({
            "step": i,
            "question_id": pq.question_id,
            "concept_id": pq.concept_id,
            "is_correct": pq.is_correct,
            "p_l": result.new_state.p_l,
            "message": "Update processed"
        })
        
    summary = orchestrator.mastery_summary()
    
    # Build suggestions based on final mastery
    suggestions = []
    for cid, stats in summary.items():
        score = stats["p_l"]
        if score > 0.90:
            suggestions.append(f"Concept {cid}: Mastery achieved! Great job.")
        elif score < 0.60:
            suggestions.append(f"Concept {cid}: Needs review. We suggest RAG remediation.")
        else:
            suggestions.append(f"Concept {cid}: On track, keep practicing.")
            
    if not suggestions:
        suggestions = ["Not enough data for concepts. Try answering more questions!"]

    return {
        "theta": theta_result.theta,
        "initial_masteries": initial_p_l0,
        "history": history,
        "summary": summary,
        "suggestions": suggestions
    }

from fastapi.responses import RedirectResponse
@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")
