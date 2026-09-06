import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")
dev_doc_bin = DocBin()

# These must be DIFFERENT examples than the ones in train.spacy
dev_data = [
    ("I need a new ticket for the broken printer in HR", 
     {"intent": "CREATE_INCIDENT", "entities": [(34, 41, "CATEGORY")]}),
    ("Is INC9998887 resolved yet?", 
     {"intent": "GET_INCIDENT", "entities": [(3, 13, "INCIDENT_NUMBER")]}),
    # ... add about 20% of your total data here ...
]

for text, annotations in dev_data:
    doc = nlp.make_doc(text)
    
    # Add Entities
    ents = []
    for start, end, label in annotations.get("entities", []):
        span = doc.char_span(start, end, label=label)
        if span:
            ents.append(span)
    doc.ents = ents
    
    # Add Intents
    doc.cats = {
        "CREATE_INCIDENT": 1.0 if annotations.get("intent") == "CREATE_INCIDENT" else 0.0,
        "GET_INCIDENT": 1.0 if annotations.get("intent") == "GET_INCIDENT" else 0.0,
        "UPDATE_INCIDENT": 1.0 if annotations.get("intent") == "UPDATE_INCIDENT" else 0.0
    }
    
    dev_doc_bin.add(doc)

# Save the dev file
dev_doc_bin.to_disk("./dev.spacy")
print("Validation data saved to dev.spacy")