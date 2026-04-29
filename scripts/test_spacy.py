import spacy

nlp = spacy.load("en_core_sci_sm", disable=["tagger", "parser", "ner", "lemmatizer", "textcat"])
nlp.add_pipe("sentencizer")

text = "Title: A cure for diabetes.\nAbstract: We present a new cure. It works well."
doc = nlp(text)
for i, sent in enumerate(doc.sents):
    print(f"Sent {i}: {sent.text!r}")
