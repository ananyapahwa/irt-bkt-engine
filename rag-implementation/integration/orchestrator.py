"""
orchestrator.py — The integration seam between BKT, IRT, and RAG.
"""

import sys
import os

# Ensure bkt and irt engines are importable by adding project root
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)
    
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import from BKT
from bkt.engine import BKTUpdateResult
from examples.mock_orchestrator import BKTOrchestrator, PracticeQuestion

# Import from RAG
from rag.config import RAG_MASTERY_THRESHOLD, TOP_K_CHUNKS
from rag.models import TutoringContext, TutoringTurn
from rag.vector_store import retrieve
from rag.tutor import generate_tutoring_response
from .session import TutoringSession

@dataclass
class SessionResult:
    """The combined output of processing an answer: BKT update + optional Tutoring."""
    bkt_result: BKTUpdateResult
    tutoring_turn: Optional[TutoringTurn] = None

class RAGOrchestrator:
    """
    Wraps the BKTOrchestrator to inject RAG tutoring when a concept is weak.
    It reads BKT and IRT state, but never modifies them.
    """
    def __init__(self, bkt_orchestrator: BKTOrchestrator, session_id: str, concept_name_map: Dict[str, str] = None):
        self.bkt = bkt_orchestrator
        self.session = TutoringSession(session_id)
        self.concept_name_map = concept_name_map or {}
        
    def process_answer(
        self, 
        question: PracticeQuestion, 
        correct_answer: str, 
        misconception_tag: Optional[str] = None,
        student_selected_text: Optional[str] = None
    ) -> SessionResult:
        """
        Process an answer through BKT, and if mastery drops below threshold, 
        trigger the AI Tutor.
        """
        concept_id = question.concept_id
        
        # 1. Run the standard BKT update (no changes to BKT math)
        bkt_result = self.bkt.process_answer(question)
        new_mastery = bkt_result.new_state.p_l
        
        tutoring_turn = None
        
        # 2. Check if we should trigger tutoring
        if new_mastery < RAG_MASTERY_THRESHOLD:
            # Check turn limit to avoid infinite loops
            if not self.session.is_turn_limit_reached(concept_id):
                turn_number = self.session.get_turn_number(concept_id)
                
                # Fetch concept name from the map (fallback to ID)
                concept_name = self.concept_name_map.get(concept_id, concept_id)
                
                # Retrieve relevant chunks from vector store
                # We can optionally pass the misconception tag or question text as a semantic query,
                # but for this demo, filtering by concept_id and getting top_k is sufficient.
                query_str = misconception_tag if misconception_tag else None
                retrieved_chunks = retrieve(concept_id=concept_id, query=query_str, top_k=TOP_K_CHUNKS)
                
                actual_answer = student_selected_text if student_selected_text else ("[Incorrect Option]" if not question.is_correct else "[Correct Option]")
                
                # Build context
                context = TutoringContext(
                    concept_id=concept_id,
                    concept_name=concept_name,
                    student_answer=actual_answer,
                    correct_answer=correct_answer,
                    misconception_tag=misconception_tag,
                    mastery_probability=new_mastery,
                    theta=self.bkt.student.theta,
                    retrieved_chunks=retrieved_chunks,
                    turn_number=turn_number
                )
                
                # Generate LLM response
                tutor_response = generate_tutoring_response(context)
                
                # Create audit record
                tutoring_turn = TutoringTurn(
                    session_id=self.session.session_id,
                    concept_id=concept_id,
                    student_answer=actual_answer,
                    misconception_tag=misconception_tag,
                    retrieved_chunk_ids=tutor_response.retrieved_chunk_ids,
                    theta=self.bkt.student.theta,
                    mastery_at_turn=new_mastery,
                    tutor_response=tutor_response.response_text,
                    turn_number=tutor_response.turn_number
                )
                
                # Save to session history
                self.session.add_turn(tutoring_turn)
                
        else:
            # If mastery is high, reset the conversation history so next time they stumble it starts fresh
            self.session.reset_concept(concept_id)
            
        return SessionResult(
            bkt_result=bkt_result,
            tutoring_turn=tutoring_turn
        )
