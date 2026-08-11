import json
from typing import Dict, List, Optional, Set

class KnowledgeGraph:
    def __init__(self, edges_file: str):
        """
        Initialize the Knowledge Graph with prerequisite edges.
        edges_file should be the path to edges.json
        """
        self.adjacency_list: Dict[str, List[str]] = {}
        
        with open(edges_file, 'r') as f:
            edges = json.load(f)
            
        for edge in edges:
            if edge.get('relationship_type') == 'PREREQUISITE_OF':
                from_id = edge['from_concept_id']
                to_id = edge['to_concept_id']
                
                if to_id not in self.adjacency_list:
                    self.adjacency_list[to_id] = []
                self.adjacency_list[to_id].append(from_id)
                
    def get_prerequisites(self, concept_id: str) -> List[str]:
        """Returns direct prerequisites of a concept"""
        return self.adjacency_list.get(concept_id, [])

    def find_failing_prerequisites(self, target_concept_id: str, student_mastery_dict: Dict[str, float], threshold: float = 0.6) -> List[Dict]:
        """
        Traverses the graph backwards from target_concept_id to find any prerequisites 
        (direct or indirect) where the student's mastery is strictly below the threshold.
        
        student_mastery_dict: Mapping from concept_id to mastery probability [0.0, 1.0].
        threshold: The threshold below which a concept is considered "failing".
        
        Returns a list of dictionaries with 'concept_id' and 'mastery'.
        """
        failing_prereqs = []
        visited: Set[str] = set()
        
        # We use a queue for BFS traversal of prerequisites
        queue = [target_concept_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            # Check prerequisites of the current concept
            prereqs = self.get_prerequisites(current)
            for prereq in prereqs:
                if prereq not in visited and prereq not in queue:
                    # Check mastery level
                    mastery = student_mastery_dict.get(prereq)
                    
                    if mastery is not None and mastery < threshold:
                        failing_prereqs.append({
                            'concept_id': prereq,
                            'mastery': mastery,
                            'failed_for': current
                        })
                    
                    # Continue traversing backwards regardless of if this one failed
                    queue.append(prereq)
                    
        return failing_prereqs
