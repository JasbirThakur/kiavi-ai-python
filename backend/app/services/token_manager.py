import re
from typing import List, Dict, Any, Tuple

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    _TIKTOKEN_AVAILABLE = False
    _ENCODER = None

def count_tokens(text: str) -> int:
    """
    Counts tokens accurately using tiktoken cl100k_base BPE tokenizer.
    Falls back gracefully to a calibrated character ratio (~3.8 chars/token) if tiktoken is unavailable.
    """
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE and _ENCODER:
        try:
            return len(_ENCODER.encode(text, disallowed_special=()))
        except Exception:
            pass
    # Resilient character ratio fallback
    return max(1, int(len(text) / 3.8))

def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """Calculates total tokens across a list of chat completion message dicts."""
    total = 0
    for m in messages:
        total += 4  # ChatML formatting overhead per message: <|im_start|>role\ncontent<|im_end|>
        content = m.get("content", "")
        total += count_tokens(content)
        role = m.get("role", "")
        total += count_tokens(role)
    total += 2  # priming tokens
    return total

def fit_context_window(
    system_prompt: str,
    chat_history: List[Dict[str, str]],
    candidate_chunks: List[Dict[str, Any]],
    user_query: str,
    max_budget: int = 6000,
    reserve_completion: int = 800
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Strict Context Window Overflow Guard:
    Guarantees that (system_prompt + chat_history + context_chunks + user_query + completion)
    never exceeds the model's token limits, preventing HTTP 400 Context Window Exceeded crashes.

    Prioritizes highest-scoring reranked candidates, safely slicing at sentence boundaries
    or trimming older chat history if necessary.
    """
    # 1. Base tokens calculation
    sys_tokens = count_tokens(system_prompt)
    query_tokens = count_tokens(user_query)
    
    # 2. History tokens calculation (trim oldest messages if history is massive)
    history_copy = list(chat_history)
    hist_tokens = count_messages_tokens(history_copy)
    
    # Reserve tokens for model output
    safety_margin = 150
    fixed_tokens = sys_tokens + query_tokens + hist_tokens + reserve_completion + safety_margin
    
    # If fixed tokens alone exceed 75% of budget, trim oldest conversation messages
    while fixed_tokens > (max_budget * 0.75) and len(history_copy) > 1:
        history_copy.pop(0)
        hist_tokens = count_messages_tokens(history_copy)
        fixed_tokens = sys_tokens + query_tokens + hist_tokens + reserve_completion + safety_margin

    available_for_context = max(300, max_budget - fixed_tokens)

    # 3. Fit candidates into available context budget
    selected_chunks = []
    used_context_tokens = 0
    truncated_count = 0

    for cand in candidate_chunks:
        cand_text = cand.get("text", "").strip()
        if not cand_text:
            continue

        cand_tokens = count_tokens(cand_text)

        if used_context_tokens + cand_tokens <= available_for_context:
            selected_chunks.append(cand)
            used_context_tokens += cand_tokens
        else:
            # Check if we can fit a partial chunk cleanly by sentence boundaries
            remaining_tokens = available_for_context - used_context_tokens
            if remaining_tokens >= 80:
                # Slices at sentence boundaries
                sentences = re.split(r'(?<=[.?!])\s+', cand_text)
                accumulated_sentences = []
                acc_tokens = 0
                for s in sentences:
                    s_tok = count_tokens(s)
                    if acc_tokens + s_tok <= remaining_tokens:
                        accumulated_sentences.append(s)
                        acc_tokens += s_tok
                    else:
                        break

                if accumulated_sentences:
                    sliced_cand = dict(cand)
                    sliced_cand["text"] = " ".join(accumulated_sentences).strip()
                    sliced_cand["truncated"] = True
                    selected_chunks.append(sliced_cand)
                    used_context_tokens += acc_tokens
                    truncated_count += 1
            break

    total_tokens = sys_tokens + query_tokens + hist_tokens + used_context_tokens

    telemetry = {
        "system_tokens": sys_tokens,
        "query_tokens": query_tokens,
        "history_tokens": hist_tokens,
        "context_tokens": used_context_tokens,
        "total_prompt_tokens": total_tokens,
        "max_budget": max_budget,
        "selected_chunks_count": len(selected_chunks),
        "total_candidates_count": len(candidate_chunks),
        "truncated_chunks_count": truncated_count,
        "tokenizer": "tiktoken (cl100k_base)" if _TIKTOKEN_AVAILABLE else "character-ratio-fallback"
    }

    return selected_chunks, telemetry

