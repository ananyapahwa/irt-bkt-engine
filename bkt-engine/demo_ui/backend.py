import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# Also add irt-engine to path so we can import from irt module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'irt-engine'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'rag-implementation'))


from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any

from examples.mock_orchestrator import BKTOrchestrator, IRTItemBank, IRTStudentProfile, ConceptIRTInfo, PracticeQuestion
from irt.theta import estimate_theta, AnswerRecord, QuestionIRTParameters
from irt.mastery_initializer import initialize_mastery, ConceptAttempt
from irt.bloom_mapper import difficulty_for

from integration.orchestrator import RAGOrchestrator
from data.seed_questions import get_tagged_questions

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

# Load concepts for name lookup
CONCEPTS_FILE = os.path.join(os.path.dirname(__file__), "concepts.json")
concept_name_map = {}  # concept_id -> concept_name
try:
    with open(CONCEPTS_FILE, "r") as f:
        concepts_data = json.load(f)
    for c in concepts_data:
        concept_name_map[c["concept_id"]] = c["concept_name"]
except Exception as e:
    print(f"Error loading {CONCEPTS_FILE}: {e}")

# Update questions with misconception tags
try:
    tagged_questions = get_tagged_questions()
    if tagged_questions:
        questions_data = tagged_questions
except Exception as e:
    print(f"Error loading tagged questions: {e}")

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
    selected_option: str = ""  # The option key the student selected (e.g., "A", "B")

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
    bkt_orchestrator = BKTOrchestrator(
        student=student_profile,
        item_bank=item_bank,
        concept_info=concept_info
    )
    
    # Wrap in RAGOrchestrator
    orchestrator = RAGOrchestrator(
        bkt_orchestrator=bkt_orchestrator,
        session_id=req.student_id,
        concept_name_map=concept_name_map
    )
    
    initial_p_l0 = {cid: state.p_l for cid, state in bkt_orchestrator.states.items()}
    history = []
    tutoring_interventions = []
    
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
        
        # Get correct answer and resolve the selected option's misconception tag
        correct_answer = q_info.get("correct_answer", "")
        misconception_tag = None
        if not p_ans.is_correct and "misconception_tags" in q_info:
            tags_map = q_info["misconception_tags"]
            # Use selected_option to get the exact misconception tag for this wrong answer
            if p_ans.selected_option and p_ans.selected_option in tags_map:
                misconception_tag = tags_map[p_ans.selected_option]
            elif tags_map:
                # Fallback: pick the first available tag
                misconception_tag = next(iter(tags_map.values()))
        
        try:
            result = orchestrator.process_answer(pq, correct_answer=correct_answer, misconception_tag=misconception_tag)
        except Exception as e:
            print(f"RAG processing error for {pq.question_id}: {e}")
            # Fall back to BKT-only processing
            bkt_result = bkt_orchestrator.process_answer(pq)
            from dataclasses import dataclass as _dc
            @_dc
            class _FallbackResult:
                bkt_result: object
                tutoring_turn: object = None
            result = _FallbackResult(bkt_result=bkt_result)
        
        history.append({
            "step": i,
            "question_id": pq.question_id,
            "concept_id": pq.concept_id,
            "concept_name": concept_name_map.get(pq.concept_id, pq.concept_id),
            "is_correct": pq.is_correct,
            "p_l": result.bkt_result.new_state.p_l,
            "message": "Update processed"
        })
        
        if result.tutoring_turn:
            tutoring_interventions.append({
                "concept_id": result.tutoring_turn.concept_id,
                "concept_name": concept_name_map.get(result.tutoring_turn.concept_id, result.tutoring_turn.concept_id),
                "student_answer": result.tutoring_turn.student_answer,
                "tutor_response": result.tutoring_turn.tutor_response,
                "turn_number": result.tutoring_turn.turn_number,
                "misconception_tag": result.tutoring_turn.misconception_tag,
                "mastery_at_turn": result.tutoring_turn.mastery_at_turn
            })
            
    summary = bkt_orchestrator.mastery_summary()
    
    # Run Prerequisite Tracing for Concepts < 0.6
    from bkt.graph_tracker import KnowledgeGraph
    try:
        kg = KnowledgeGraph(os.path.join(os.path.dirname(__file__), "edges.json"))
        tracing_results = {}
        
        # Build mastery dict for tracing (using final p_l from summary)
        mastery_dict = {cid: stats["p_l"] for cid, stats in summary.items()}
        
        for cid, stats in summary.items():
            if stats["p_l"] < 0.60:
                failing_prereqs = kg.find_failing_prerequisites(cid, mastery_dict, threshold=0.6)
                if failing_prereqs:
                    tracing_results[cid] = failing_prereqs
    except Exception as e:
        print(f"Graph tracing error: {e}")
        tracing_results = {}
    
    # Build suggestions based on final mastery
    suggestions = []
    for cid, stats in summary.items():
        cname = concept_name_map.get(cid, cid)
        score = stats["p_l"]
        if score > 0.90:
            suggestions.append(f"{cname} ({cid}): Mastery achieved! Great job.")
        elif score < 0.60:
            trace = tracing_results.get(cid)
            if trace:
                prereq_names = [f"{concept_name_map.get(t['concept_id'], t['concept_id'])}" for t in trace]
                suggestions.append(f"{cname} ({cid}): Needs review. Failing prerequisites: {', '.join(prereq_names)}. We suggest reviewing them first.")
            else:
                suggestions.append(f"{cname} ({cid}): Needs review. AI Tutor remediation recommended.")
        else:
            suggestions.append(f"{cname} ({cid}): On track, keep practicing.")
            
    if not suggestions:
        suggestions = ["Not enough data for concepts. Try answering more questions!"]

    return {
        "theta": theta_result.theta,
        "initial_masteries": initial_p_l0,
        "history": history,
        "summary": summary,
        "suggestions": suggestions,
        "tracing_results": tracing_results,
        "tutoring_interventions": tutoring_interventions,
        "concept_names": concept_name_map
    }

from fastapi.responses import RedirectResponse
@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")
