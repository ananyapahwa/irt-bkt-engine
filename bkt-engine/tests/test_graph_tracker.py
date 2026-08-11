import os
import tempfile
import json
import pytest
from bkt.graph_tracker import KnowledgeGraph

@pytest.fixture
def temp_edges_file():
    edges_data = [
        {"from_concept_id": "C1", "to_concept_id": "C2", "relationship_type": "PREREQUISITE_OF"},
        {"from_concept_id": "C2", "to_concept_id": "C3", "relationship_type": "PREREQUISITE_OF"},
        {"from_concept_id": "C4", "to_concept_id": "C3", "relationship_type": "PREREQUISITE_OF"}
    ]
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        json.dump(edges_data, f)
        temp_name = f.name
    
    yield temp_name
    os.remove(temp_name)

def test_get_prerequisites(temp_edges_file):
    kg = KnowledgeGraph(temp_edges_file)
    assert set(kg.get_prerequisites("C3")) == {"C2", "C4"}
    assert set(kg.get_prerequisites("C2")) == {"C1"}
    assert set(kg.get_prerequisites("C1")) == set()

def test_find_failing_prerequisites(temp_edges_file):
    kg = KnowledgeGraph(temp_edges_file)
    
    # Simulate a student who fails C3, and their mastery is:
    # C1: 0.9 (mastered)
    # C2: 0.4 (failing)
    # C4: 0.8 (mastered)
    mastery_dict = {
        "C1": 0.9,
        "C2": 0.4,
        "C3": 0.3,
        "C4": 0.8
    }
    
    failing = kg.find_failing_prerequisites("C3", mastery_dict, threshold=0.6)
    
    assert len(failing) == 1
    assert failing[0]['concept_id'] == "C2"
    assert failing[0]['mastery'] == 0.4
    assert failing[0]['failed_for'] == "C3"

def test_find_failing_prerequisites_indirect(temp_edges_file):
    kg = KnowledgeGraph(temp_edges_file)
    
    # Simulate a student where C2 is okay, but C1 is failing.
    mastery_dict = {
        "C1": 0.2, # Indirect prerequisite, failing
        "C2": 0.7, # Direct prerequisite, passing
        "C3": 0.3, # Target
        "C4": 0.8
    }
    
    failing = kg.find_failing_prerequisites("C3", mastery_dict, threshold=0.6)
    
    assert len(failing) == 1
    assert failing[0]['concept_id'] == "C1"
    assert failing[0]['mastery'] == 0.2
    assert failing[0]['failed_for'] == "C2"
