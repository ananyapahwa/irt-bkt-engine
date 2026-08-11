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
        },
        "q_6": {
            "A": "multiplied_voltage_and_current",
            "C": "divided_current_by_voltage",
            "D": "ignored_current_value"
        },
        "q_7": {
            "A": "forgot_to_square_current_and_didnt_convert_time",
            "B": "forgot_to_convert_minutes_to_seconds",
            "D": "forgot_to_square_current"
        },
        "q_8": {
            "B": "math_error_in_kwh_conversion",
            "C": "incorrect_metric_prefix_conversion",
            "D": "forgot_to_multiply_by_days"
        },
        "q_9": {
            "A": "ignored_other_series_resistors_used_only_six",
            "C": "used_largest_resistor_only",
            "D": "confused_parallel_and_series_current_rules"
        },
        "q_10": {
            "A": "used_series_formula_for_parallel",
            "B": "averaged_resistances",
            "D": "multiplied_resistances_without_dividing"
        },
        "q_11": {
            "A": "forgot_area_changes_when_stretched",
            "B": "thought_longer_wire_has_less_resistance",
            "D": "divided_instead_of_multiplying_by_four"
        },
        "q_12": {
            "A": "thinks_parallel_lowers_voltage",
            "C": "thinks_parallel_reduces_total_current",
            "D": "thinks_its_just_for_wiring_convenience"
        },
        "q_13": {
            "A": "thinks_fuse_is_variable_resistor",
            "C": "thinks_broken_fuse_conducts",
            "D": "thinks_fuse_regulates_current"
        },
        "q_14": {
            "A": "confuses_earth_with_neutral",
            "C": "thinks_earth_saves_energy",
            "D": "misunderstands_basic_circuit_roles"
        },
        "q_15": {
            "A": "believes_in_magnetic_monopoles",
            "C": "thinks_cutting_destroys_magnetism",
            "D": "thinks_poles_are_unequal"
        }
    }
    
    for q in questions:
        q_copy = dict(q)
        q_copy["misconception_tags"] = tags_map.get(q["id"], {})
        tagged_questions.append(q_copy)
        
    return tagged_questions
