import pandas as pd
import json

# Read Questions
df = pd.read_excel("./irt-engine/data/Synapse_Quiz_30Q.xlsx", header=1)
# Drop completely empty rows just in case
df = df.dropna(how='all')

questions = []
for index, row in df.iterrows():
    if pd.isna(row.get('Q#')):
        continue
        
    q = {
        "id": f"q_{int(row['Q#'])}",
        "chapter": row['Chapter'],
        "concept_id": row['concept_id'],
        "bloom": row['Bloom'],
        "difficulty": row.get('Difficulty', 0),
        "type": row['Type'],
        "text": row['Question Stem'],
        "options": {
            "A": row['Option A'],
            "B": row['Option B'],
            "C": row['Option C'],
            "D": row['Option D']
        },
        "correct_answer": row['Correct Answer'],
        "correct_reasoning": row.get('Correct Reasoning', '')
    }
    questions.append(q)

with open("./bkt-engine/demo_ui/questions.json", "w") as f:
    json.dump(questions, f, indent=2)

# Read Concepts
df_concepts = pd.read_excel("./irt-engine/data/Synapse_KG_Concepts.xlsx", header=0)
concepts = []
for index, row in df_concepts.iterrows():
    if pd.isna(row.get('concept_id')):
        continue
    concepts.append(row.to_dict())

with open("./bkt-engine/demo_ui/concepts.json", "w") as f:
    json.dump(concepts, f, indent=2)

print(f"Parsed {len(questions)} questions and {len(concepts)} concepts successfully.")
