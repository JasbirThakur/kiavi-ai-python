from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import models
import numpy as np
import schemas
from fastapi.middleware.cors import CORSMiddleware
from database import engine, get_db
import json
from sentence_transformers import SentenceTransformer
# Database tables create karna
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Kiavi Agentic AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Production mein hum ise "http://localhost:3000" kar denge
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- AI MODEL LOADING ---
print("Loading AI Embedding Model (Pehli baar thoda time lag sakta hai)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("AI Model Successfully Loaded!")
# ------------------------

# 1. Test Route
@app.get("/")
def read_root():
    return {"status": "Success", "message": "Backend is running!"}

# 2. Naya User Create karne ka API Endpoint
@app.post("/api/users/")
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check karein ki email pehle se toh nahi hai
    existing_user = db.query(models.User).filter(models.User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Naya user database mein daalein
    db_user = models.User(
        name=user.name, 
        email=user.email, 
        passwordHash=user.password # Real app mein hum ise hash karenge
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return {"message": "User successfully created!", "user_id": db_user.id}
# 3. Naya AI Bot Create karne ka API Endpoint
@app.post("/api/bots/")
def create_bot(bot: schemas.BotCreate, db: Session = Depends(get_db)):
    # Check karein ki organization exist karti hai ya nahi
    org = db.query(models.Organization).filter(models.Organization.id == bot.orgId).first()
    
    # Testing ke liye agar org nahi milti toh hum ek dummy organization bana dete hain
    if not org:
        org = models.Organization(id=bot.orgId, name="Default Kiavi Org", ownerId="c_dummyowner")
        db.add(org)
        db.commit()

    # Naya Bot database mein save karein
    db_bot = models.Bot(
        name=bot.name,
        orgId=bot.orgId,
        systemPrompt=bot.systemPrompt or "You are a helpful AI assistant for Kiavi IQ."
    )
    db.add(db_bot)
    db.commit()
    db.refresh(db_bot)

    return {
        "message": "AI Bot successfully created!",
        "bot_id": db_bot.id,
        "bot_name": db_bot.name
    }
# 4. Naya Knowledge Base Create karne ka API Endpoint
@app.post("/api/knowledge-bases/")
def create_knowledge_base(kb: schemas.KnowledgeBaseCreate, db: Session = Depends(get_db)):
    # Pehle check karein ki Bot exist karta hai ya nahi
    bot = db.query(models.Bot).filter(models.Bot.id == kb.botId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot nahi mila. Kripya sahi Bot ID daalein.")

    # Naya Knowledge Base database mein save karein
    db_kb = models.KnowledgeBase(
        name=kb.name, 
        botId=kb.botId
    )
    db.add(db_kb)
    db.commit()
    db.refresh(db_kb)

    return {
        "message": "Knowledge Base successfully created!",
        "kb_id": db_kb.id,
        "kb_name": db_kb.name
    }

# 5. Document Chunk Create karne ka API Endpoint (WITH AI EMBEDDINGS)
@app.post("/api/chunks/")
def create_document_chunk(chunk: schemas.DocumentChunkCreate, db: Session = Depends(get_db)):
    # Pehle check karein ki Knowledge Base exist karta hai ya nahi
    kb = db.query(models.KnowledgeBase).filter(models.KnowledgeBase.id == chunk.knowledgeBaseId).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge Base nahi mila.")

    # --- AI MAGIC: Text ko Vectors (Numbers) mein convert karein ---
    vector_list = embedding_model.encode(chunk.content).tolist()
    embedding_str = json.dumps(vector_list) # List ko String banaya taaki DB mein save ho sake
    # ---------------------------------------------------------------

    # Naya Chunk database mein save karein
    db_chunk = models.DocumentChunk(
        content=chunk.content,
        knowledgeBaseId=chunk.knowledgeBaseId,
        embedding=embedding_str  # <-- Ab yahan null nahi, actual AI data aayega!
    )
    db.add(db_chunk)
    db.commit()
    db.refresh(db_chunk)

    return {
        "message": "Document Chunk successfully saved with AI Embeddings!",
        "chunk_id": db_chunk.id,
        "content_preview": db_chunk.content[:50] + "...",
        "embedding_size": len(vector_list) # Ye dikhayega ki kitne numbers generate hue
    }

# 6. Sawaal Puchne ka (Chat/Search) API Endpoint!
@app.post("/api/chat/")
def chat_with_bot(request: schemas.ChatRequest, db: Session = Depends(get_db)):
    # 1. Check karein ki Bot exist karta hai ya nahi
    bot = db.query(models.Bot).filter(models.Bot.id == request.botId).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot nahi mila.")

    # 2. User ke sawaal ko AI vectors (384 numbers) mein convert karein
    question_vector = embedding_model.encode(request.question)

    best_match_score = -1
    best_answer = "Sorry, mere knowledge base mein iska jawab nahi hai."

    # 3. Bot ke saare chunks ko scan karein (In-Memory Vector Search)
    for kb in bot.knowledgeBases:
        for chunk in kb.chunks:
            if chunk.embedding:
                # Database ke string array ko wapas Numbers ki list mein badlein
                chunk_vector = np.array(json.loads(chunk.embedding))
                
                # Math Magic: Cosine Similarity (Dono arrays kitne similar hain)
                dot_product = np.dot(question_vector, chunk_vector)
                norm_a = np.linalg.norm(question_vector)
                norm_b = np.linalg.norm(chunk_vector)
                similarity_score = dot_product / (norm_a * norm_b)

                # Agar ye chunk pichle wale se better match hai, toh ise save kar lein
                if similarity_score > best_match_score:
                    best_match_score = similarity_score
                    best_answer = chunk.content

    return {
        "bot_id": bot.id,
        "question": request.question,
        "best_answer": best_answer,
        "confidence_score": round(float(best_match_score), 4) # Match percentage
    }