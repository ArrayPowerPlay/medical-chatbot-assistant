"""
Smoke test for type classifier and preamble stripper.
Run: python scripts/evaluation/test_generation_improvements.py
"""
import sys
sys.path.insert(0, 'd:/workspace/Repo/medical-chatbot-assistant')

from src.generation.prompt_builder import classify_question_type
from scripts.evaluation.shared.generation_bioasq_common import strip_preamble

test_cases = [
    ('Is the protein Papilin secreted?', 'yesno'),
    ('Are long non coding RNAs spliced?', 'yesno'),
    ('List signaling molecules (ligands) that interact with the receptor EGFR?', 'list'),
    ('Which miRNAs could be used as potential biomarkers for epithelial ovarian cancer?', 'list'),
    ('Which acetylcholinesterase inhibitors are used for treatment of myasthenia gravis?', 'list'),
    ('Name synonym of Acrokeratosis paraneoplastica.', 'factoid'),
    ('Orteronel was developed for treatment of which cancer?', 'factoid'),
    ('Where is the protein Pannexin1 located?', 'factoid'),
    ('Is Hirschsprung disease a mendelian or a multifactorial disorder?', 'summary'),
    ('What is the aim of the Human Chromosome-centric Proteome Project (C-HPP)?', 'summary'),
    ('What is the effect of ivabradine in heart failure after myocardial infarction?', 'summary'),
]

print('=== Classifier Accuracy Test ===')
correct = 0
for body, expected in test_cases:
    predicted = classify_question_type(body)
    ok = predicted == expected
    correct += ok
    marker = 'OK  ' if ok else 'FAIL'
    print(f'  [{marker}] predicted={predicted:8s} expected={expected:8s} | {body[:65]}')

print(f'\nAccuracy: {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.1f}%')

print()
print('=== Preamble Stripper Test ===')
preambles = [
    ('Based on the provided context, Metformin is used for type 2 diabetes.', True),
    ('According to the documents, the answer is yes.', True),
    ('The provided context indicates that papilin is secreted.', True),
    ('From the given context, EGF is a ligand of EGFR.', True),
    ('As stated in the provided sources, ivabradine decreases heart rate.', True),
    ('Yes, papilin is a secreted protein.', False),
    ('Metformin is a biguanide drug used to treat type 2 diabetes.', False),
]
strip_correct = 0
for text, should_strip in preambles:
    result = strip_preamble(text)
    was_stripped = result != text
    ok = was_stripped == should_strip
    strip_correct += ok
    marker = 'OK  ' if ok else 'FAIL'
    print(f'  [{marker}] stripped={str(was_stripped):5s} (expected {str(should_strip):5s})')
    print(f'         IN:  {text[:75]}')
    if was_stripped:
        print(f'         OUT: {result[:75]}')
    print()

print(f'Stripper accuracy: {strip_correct}/{len(preambles)} = {strip_correct/len(preambles)*100:.1f}%')
