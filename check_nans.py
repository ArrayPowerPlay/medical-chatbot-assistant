import json
import math

def check():
    ckpt_path = 'results/test_results/bioasq/baseline_vector/generation/ragas_checkpoint.json'
    preds_path = 'results/test_results/bioasq/baseline_vector/generation/predictions.jsonl'
    
    with open(ckpt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    preds = {}
    with open(preds_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            preds[r['question_id']] = r
            
    nan_ids = []
    for qid, m in data.items():
        if any(isinstance(v, float) and math.isnan(v) for v in m.values()):
            nan_ids.append(qid)
            
    print(f"Total NaNs: {len(nan_ids)}")
    for qid in nan_ids[:5]:
        p = preds.get(qid, {})
        ctx = p.get('contexts', [])
        ans = p.get('generated_answer', '')[:20]
        print(f"{qid} - Contexts: {len(ctx)} items. Answer: {ans}")

if __name__ == '__main__':
    check()
