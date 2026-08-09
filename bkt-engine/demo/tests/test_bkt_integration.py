import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from demo.personas import build_personas
from demo.events import classify_event
from examples.mock_orchestrator import BKTOrchestrator

def test_persona_a_rag_remediation():
    personas = build_personas()
    persona_a = next(p for p in personas if p.id == "student_A")
    
    orchestrator = BKTOrchestrator(
        student=persona_a.student,
        item_bank=persona_a.item_bank,
        concept_info=persona_a.concept_info
    )
    
    events = []
    for q in persona_a.session:
        res = orchestrator.process_answer(q)
        events.append(classify_event(res))
        
    # Should trigger RAG remediation due to drops in P(L)
    assert any(e.kind == "RAG_REMEDIATION" for e in events)

def test_persona_b_guess_flagged():
    personas = build_personas()
    persona_b = next(p for p in personas if p.id == "student_B")
    
    orchestrator = BKTOrchestrator(
        student=persona_b.student,
        item_bank=persona_b.item_bank,
        concept_info=persona_b.concept_info
    )
    
    events = []
    for q in persona_b.session:
        res = orchestrator.process_answer(q)
        events.append(classify_event(res))
        
    assert any(e.kind == "GUESS_FLAGGED" for e in events)

def test_persona_c_mastery_achieved():
    personas = build_personas()
    persona_c = next(p for p in personas if p.id == "student_C")
    
    orchestrator = BKTOrchestrator(
        student=persona_c.student,
        item_bank=persona_c.item_bank,
        concept_info=persona_c.concept_info
    )
    
    events = []
    for q in persona_c.session:
        res = orchestrator.process_answer(q)
        events.append(classify_event(res))
        
    assert any(e.kind == "MASTERY_ACHIEVED" for e in events)
