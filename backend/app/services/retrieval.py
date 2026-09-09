from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.db import models
from app.config.settings import RELEVANCE_FLOOR, RESCUE_FLOOR, TOP_K_CHUNKS
from app.utils.logger import logger

def search_relevant_chunks(
    db: Session,
    query_vec: List[float],
    bot_id: Optional[str] = None,
    org_id: Optional[str] = None,
    top_k: int = TOP_K_CHUNKS,
    relevance_floor: float = RELEVANCE_FLOOR,
    search_terms: Optional[List[str]] = None
) -> List[Tuple[str, float]]:
    """
    Performs hybrid retrieval using pgvector cosine distance and keyword matching.
    Returns list of tuples: [(chunk_content, similarity_score)]
    """
    results: List[Tuple[str, float]] = []

    try:
        # Base query joining DocumentChunk with BotSource
        query = (
            db.query(
                models.DocumentChunk,
                models.DocumentChunk.embedding.cosine_distance(query_vec).label("distance"),
                models.BotSource.title
            )
            .join(models.BotSource, models.DocumentChunk.sourceId == models.BotSource.id)
        )

        # Filter by botId OR isUniversal (or matching orgId)
        if bot_id:
            bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first()
            current_org_id = bot.orgId if bot else org_id
            
            if current_org_id:
                query = query.filter(
                    or_(
                        models.BotSource.botId == bot_id,
                        (models.BotSource.isUniversal == True) & (models.BotSource.orgId == current_org_id)
                    )
                )
            else:
                query = query.filter(
                    or_(
                        models.BotSource.botId == bot_id,
                        models.BotSource.isUniversal == True
                    )
                )

        # Order by cosine distance ascending (closest vectors first)
        raw_chunks = query.order_by("distance").limit(top_k * 2).all()

        for chunk, distance, title in raw_chunks:
            similarity = 1.0 - float(distance) if distance is not None else 0.0
            if similarity >= relevance_floor:
                results.append((chunk.content, similarity))
                if len(results) >= top_k:
                    break

        # Fallback: Keyword search if vector similarity didn't produce enough results
        if len(results) < top_k and search_terms:
            keyword_filters = [models.DocumentChunk.content.ilike(f"%{term}%") for term in search_terms if len(term) > 3]
            if keyword_filters:
                kw_query = (
                    db.query(models.DocumentChunk, models.BotSource.title)
                    .join(models.BotSource, models.DocumentChunk.sourceId == models.BotSource.id)
                    .filter(or_(*keyword_filters))
                )
                if bot_id:
                    kw_query = kw_query.filter(
                        or_(models.BotSource.botId == bot_id, models.BotSource.isUniversal == True)
                    )
                kw_matches = kw_query.limit(top_k - len(results)).all()
                for chunk, title in kw_matches:
                    if not any(r[0] == chunk.content for r in results):
                        results.append((chunk.content, RESCUE_FLOOR))

    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)

    return results

