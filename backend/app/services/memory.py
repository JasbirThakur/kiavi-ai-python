from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from pathlib import Path
from app.db import models
from app.config.settings import PROMPTS_DIR
from app.services.llm import call_llm_sync
from app.utils.logger import logger

def get_conversation_history(
    conversation_id: str,
    db: Session,
    max_messages: int = 10
) -> List[Dict[str, str]]:
    """
    Retrieves recent messages from a conversation in chronological order.
    Returns a list of dicts: [{"role": "user"|"assistant", "content": "..."}]
    """
    messages = (
        db.query(models.Message)
        .filter(models.Message.conversationId == conversation_id)
        .order_by(models.Message.createdAt.desc())
        .limit(max_messages)
        .all()
    )
    
    # Reverse to restore chronological order
    messages = list(reversed(messages))
    
    return [
        {
            "role": "user" if m.role == "user" else "assistant",
            "content": m.content
        }
        for m in messages
    ]

def summarize_conversation_history(
    messages: List[Dict[str, str]],
    existing_summary: Optional[str] = None
) -> str:
    """
    Generates a concise factual summary of older conversation history
    to prevent prompt token overflow while preserving essential context.
    """
    if not messages:
        return existing_summary or ""

    summary_prompt_path = PROMPTS_DIR / "summarization.txt"
    if summary_prompt_path.exists():
        prompt_template = summary_prompt_path.read_text(encoding="utf-8")
    else:
        prompt_template = "Summarize the key context, entities, and decisions from this conversation:\n{conversation}\nSummary:"

    conv_text = ""
    if existing_summary:
        conv_text += f"Previous Summary:\n{existing_summary}\n\n"
    
    for m in messages:
        conv_text += f"{m['role'].capitalize()}: {m['content']}\n"

    prompt = prompt_template.format(conversation=conv_text)
    
    summary = call_llm_sync([{"role": "user", "content": prompt}], max_tokens=300)
    return summary.strip()

def build_context_window(
    conversation_id: Optional[str],
    db: Session,
    max_recent_turns: int = 6
) -> Dict[str, Any]:
    """
    Constructs an optimized context window:
    - If total messages <= max_recent_turns * 2: returns full recent messages
    - If conversation is long: summarizes older messages and keeps recent turns
    """
    if not conversation_id:
        return {"messages": [], "summary": ""}

    all_messages = (
        db.query(models.Message)
        .filter(models.Message.conversationId == conversation_id)
        .order_by(models.Message.createdAt.asc())
        .all()
    )

    if not all_messages:
        return {"messages": [], "summary": ""}

    formatted = [
        {"role": "user" if m.role == "user" else "assistant", "content": m.content}
        for m in all_messages
    ]

    threshold = max_recent_turns * 2
    if len(formatted) <= threshold:
        return {"messages": formatted, "summary": ""}

    older_messages = formatted[:-threshold]
    recent_messages = formatted[-threshold:]
    summary = summarize_conversation_history(older_messages)

    return {
        "messages": recent_messages,
        "summary": summary
    }

