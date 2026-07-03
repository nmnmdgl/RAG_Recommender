import re
import zipfile
import requests
import time
import os

LOCAL_API_URL = "https://shl-recommender-naman.onrender.com/chat"
LATENCY_BUDGET_SECONDS = 30  # from the assignment's own cap


def parse_markdown_turns(md_content):
    """Extracts user messages from the provided SHL markdown format."""
    turns = []
    blocks = md_content.split("**User**")
    
    for block in blocks[1:]: # Skip the header before the first user turn
        lines = block.split('\n')
        user_text = []
        for line in lines:
            # Extract text inside the blockquote
            if line.strip().startswith('>'):
                user_text.append(line.replace('>', '').strip())
            # Stop grabbing text when we hit the agent's section
            elif "**Agent**" in line:
                break 
        
        if user_text:
            turns.append(" ".join(user_text))
            
    return turns


def parse_gold_shortlist(md_content: str):
    """Same logic as retrieval_eval.py: the LAST markdown table in the
    trace is the converged gold shortlist. Reused here so evaluate.py
    scores the live API against the same ground truth, not just the
    retrieval layer."""
    tables = re.findall(r"\|\s*#\s*\|.*?\n(?:\|.*\n)+", md_content)
    if not tables:
        return []
    last_table = tables[-1]
    rows = [r for r in last_table.strip().split("\n") if r.startswith("|")][2:]
    names = []
    for row in rows:
        cols = [c.strip() for c in row.strip("|").split("|")]
        if len(cols) >= 2:
            names.append(cols[1])
    return names


def score(gold_names, predicted_names):
    """Recall AND precision -- recall alone can't catch the LLM adding
    extra items that technically resolve against the catalog (pass
    validate_recommendations) but weren't actually asked for."""
    gold_set = {g.lower() for g in gold_names}
    pred_set = {p.lower() for p in predicted_names}
    if not gold_set:
        return None, None
    recall = len(gold_set & pred_set) / len(gold_set)
    precision = len(gold_set & pred_set) / len(pred_set) if pred_set else 0.0
    return recall, precision

def run_local_evaluation():
    print("📋 Starting Full Batch Evaluation...\n")
    
    if not os.path.exists("sample_conversations.zip"):
        print("❌ Error: 'sample_conversations.zip' not found in this folder!")
        return

    trace_results = []  # one summary dict per trace, printed as a table at the end

    with zipfile.ZipFile("sample_conversations.zip", "r") as z:
        for file_name in z.namelist():
            # Only process the Markdown conversation files
            if not file_name.endswith(".md"):
                continue
                
            print(f"==================================================")
            print(f"▶️ RUNNING TRACE: {file_name}")
            print(f"==================================================")
            
            md_content = z.read(file_name).decode("utf-8")
            user_messages = parse_markdown_turns(md_content)
            gold = parse_gold_shortlist(md_content)
            
            payload_history = []
            latencies = []
            prev_names = None  # rec names from the previous turn, for flap detection
            last_names = []
            connection_failed = False
            
            for turn_idx, user_text in enumerate(user_messages):
                print(f"\n[Turn {turn_idx + 1}]")
                print(f"👤 USER: {user_text}")
                
                payload_history.append({"role": "user", "content": user_text})
                
                try:
                    start = time.time()
                    response = requests.post(LOCAL_API_URL, json={"messages": payload_history}, timeout=60)
                    elapsed = time.time() - start
                    latencies.append(elapsed)
                    data = response.json()
                    
                    print(f"🤖 AGENT: {data.get('reply')}")
                    
                    recs = data.get('recommendations', [])
                    names = [r.get('name', '') for r in recs]
                    last_names = names
                    budget_flag = " ⚠️ OVER 30s BUDGET" if elapsed > LATENCY_BUDGET_SECONDS else ""
                    print(f"📊 RECS: {len(recs)} | 🏁 END CONVO: {data.get('end_of_conversation')} | ⏱️ {elapsed:.1f}s{budget_flag}")
                    
                    if recs:
                        for idx, rec in enumerate(recs):
                            print(f"    {idx+1}. [{rec.get('test_type')}] {rec.get('name')} | URL: {rec.get('url')}")

                    # Flapping check: did the set of names shrink or swap
                    # (near-duplicate substitution) on a turn that added no
                    # new items? A pure superset/unchanged transition is
                    # fine; a shrink or a same-size-but-different-items
                    # swap is the bug the consistency fix targets.
                    if prev_names is not None and prev_names and names:
                        prev_set, cur_set = set(prev_names), set(names)
                        if not prev_set.issubset(cur_set):
                            lost = prev_set - cur_set
                            print(f"    ⚠️ FLAP: turn dropped/swapped previously-committed item(s): {sorted(lost)}")
                    prev_names = names

                    # Add agent response back to history for the next turn
                    payload_history.append({"role": "assistant", "content": data.get('reply', '')})
                    
                except Exception as e:
                    print(f"❌ Connection Error: Is FastAPI running? {e}")
                    connection_failed = True
                    break
                
                time.sleep(5) # Small pause to avoid overwhelming the Groq API
            print("\n")

            time.sleep(65)  # Pause between traces to avoid overwhelming the API

            if not connection_failed:
                recall, precision = score(gold, last_names)
                trace_results.append({
                    "file": file_name,
                    "gold": len(gold),
                    "final_recs": len(last_names),
                    "recall": recall,
                    "precision": precision,
                    "mean_latency": sum(latencies) / len(latencies) if latencies else None,
                    "max_latency": max(latencies) if latencies else None,
                })

    if trace_results:
        print("=" * 100)
        print("SUMMARY (final-turn recommendations vs. gold shortlist)")
        print("=" * 100)
        header = f"{'trace':30s} {'gold':>4s} {'final':>5s} {'recall':>7s} {'precision':>9s} {'mean_lat':>9s} {'max_lat':>8s}"
        print(header)
        for r in trace_results:
            recall_s = f"{r['recall']:.2f}" if r['recall'] is not None else "  n/a"
            prec_s = f"{r['precision']:.2f}" if r['precision'] is not None else "  n/a"
            mean_s = f"{r['mean_latency']:.1f}s" if r['mean_latency'] is not None else "n/a"
            max_s = f"{r['max_latency']:.1f}s" if r['max_latency'] is not None else "n/a"
            print(f"{r['file']:30s} {r['gold']:4d} {r['final_recs']:5d} {recall_s:>7s} {prec_s:>9s} {mean_s:>9s} {max_s:>8s}")
        recalls = [r['recall'] for r in trace_results if r['recall'] is not None]
        precisions = [r['precision'] for r in trace_results if r['precision'] is not None]
        max_lats = [r['max_latency'] for r in trace_results if r['max_latency'] is not None]
        print("-" * 100)
        if recalls:
            print(f"Mean end-to-end recall:    {sum(recalls)/len(recalls):.3f}")
        if precisions:
            print(f"Mean end-to-end precision: {sum(precisions)/len(precisions):.3f}")
        if max_lats:
            over_budget = sum(1 for m in max_lats if m > LATENCY_BUDGET_SECONDS)
            print(f"Slowest single turn:       {max(max_lats):.1f}s  |  traces with a turn over {LATENCY_BUDGET_SECONDS}s: {over_budget}/{len(max_lats)}")


if __name__ == "__main__":
    run_local_evaluation()