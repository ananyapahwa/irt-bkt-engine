"""
seed_questions.py — Misconception-tagged questions for the RAG demo.
"""

import os
import json

def get_tagged_questions():
    """Load questions.json from demo_ui and tag them with misconceptions."""
    questions_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
        "bkt-engine", "demo_ui", "questions.json"
    )
    
    with open(questions_file, "r") as f:
        questions = json.load(f)
        
    tagged_questions = []
    
    # We will statically assign tags to some known questions for the demo
    tags_map = {
        "q_1": {
            "A": "believes_current_gradually_decreases",
            "C": "thinks_current_direction_reverses_when_broken",
            "D": "thinks_current_jumps_air_gaps"
        },
        "q_2": {
            "A": "confuses_conventional_with_electron_flow",
            "B": "thinks_electrons_flow_positive_to_negative",
            "D": "unaware_of_conventional_current_definition"
        },
        "q_3": {
            "A": "reversed_ammeter_and_voltmeter_rules",
            "C": "thinks_voltmeter_goes_in_series",
            "D": "thinks_ammeter_goes_in_parallel"
        },
        "q_4": {
            "A": "thinks_bulb_is_ohmic",
            "C": "thinks_wire_is_nonohmic",
            "D": "thinks_all_resistance_changes_with_current"
        },
        "q_5": {
            "B": "misinterprets_VI_slope_as_conductance",
            "C": "confuses_resistance_with_voltage",
            "D": "thinks_slope_unrelated_to_resistance"
        }
    }
    
    for q in questions:
        q_copy = dict(q)
        q_copy["misconception_tags"] = tags_map.get(q["id"], {})
        tagged_questions.append(q_copy)
        
    return tagged_questions
