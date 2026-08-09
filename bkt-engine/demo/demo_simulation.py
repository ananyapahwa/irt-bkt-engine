import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo.personas import build_personas
from demo.events import classify_event
from demo.renderer import render_table
from examples.mock_orchestrator import BKTOrchestrator

def run_demo():
    personas = build_personas()
    
    for persona in personas:
        print(f"\n{'='*90}")
        print(f"Persona: {persona.name}")
        print(f"Description: {persona.description}")
        print(f"Initial θ: {persona.student.theta:.2f}")
        print(f"{'='*90}")
        
        orchestrator = BKTOrchestrator(
            student=persona.student,
            item_bank=persona.item_bank,
            concept_info=persona.concept_info
        )
        
        print("\n[Initial State Seeded from IRT Diagnostic]")
        for cid, state in orchestrator.states.items():
            print(f"  Concept: {cid} | P(L0) = {state.p_l:.4f}")
        
        print("\n[Simulating Session]")
        headers = ["Step", "Question", "b (diff)", "RT (ms)", "Result", "P(L)", "Event"]
        rows = []
        
        for i, q in enumerate(persona.session, 1):
            result = orchestrator.process_answer(q)
            event = classify_event(result)
            
            # get returns (a, b), we want b
            b_val = persona.item_bank.get(q.question_id)[1]
            correct_str = "✅ Correct" if q.is_correct else "❌ Incorrect"
            
            rows.append([
                i,
                q.question_id,
                f"{b_val:.1f}",
                q.response_time_ms,
                correct_str,
                f"{result.new_state.p_l:.4f}",
                event.message
            ])
            
        render_table(headers, rows)
        
        summary = orchestrator.mastery_summary()
        print("\n[Final Status]")
        for cid, info in summary.items():
            status = "Mastered" if info["is_mastered"] else "Not Mastered"
            print(f"  {cid}: {status} (P(L) = {info['p_l']:.4f})")
            
    print(f"\n{'='*90}")
    print("Demo complete.")

if __name__ == "__main__":
    run_demo()
