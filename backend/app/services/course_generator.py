import json
from typing import Dict, Any
from sqlalchemy.orm import Session
from app.db import models
from app.db.models import UserRole
from app.services.rag import nvidia_client, groq_client, NVIDIA_LLM_MODEL
from app.services.audit import log_audit_event
from app.utils.logger import logger

AI_COURSE_SYSTEM_PROMPT = """You are an expert Industrial Training Specialist and Curriculum Architect for heavy engineering, medical devices, and automotive manufacturing.

Your task is to analyze technical documentation (manuals, SOPs, datasheets) and transform it into a structured, certified corporate training course.

You MUST reply ONLY with a valid, parseable JSON object adhering exactly to this structure:
{
  "title": "Course Title (e.g. Surgical Catheter Sterilization & Safety Protocols)",
  "description": "2-3 sentence overview of target learning objectives and regulatory significance.",
  "category": "Domain Category (e.g. Medical Devices, Door Hardware, Automotive Systems)",
  "passing_score": 80,
  "validity_months": 24,
  "modules": [
    {
      "title": "Module 1: Title",
      "lessons": [
        {
          "title": "Lesson 1.1: Title",
          "content_type": "DOC",
          "duration_seconds": 300,
          "content": "Comprehensive instructional text covering technical parameters, torque, safety warnings, and step-by-step procedures grounded strictly in the source text."
        }
      ]
    }
  ],
  "quiz": {
    "title": "Course Certification Assessment",
    "passing_score": 80,
    "questions": [
      {
        "question": "Clear technical assessment question testing safety or precise specifications?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_option_index": 0,
        "explanation": "Detailed explanation citing the manual why this answer is correct."
      }
    ]
  }
}

Do not include any conversational preamble, markdown backticks, or trailing commentary. Return ONLY the raw JSON object."""

def generate_course_from_document(
    db: Session,
    source_id: str,
    org_id: str,
    current_user: models.User,
    client_ip: str = "unknown"
) -> Dict[str, Any]:
    """
    Parses an approved technical manual from BotSources and auto-generates a structured course syllabus,
    lesson content, and assessment quiz questions using LLM inference.
    Saves the generated course into PostgreSQL in draft mode for human trainer review.
    """
    doc = db.query(models.BotSource).filter(models.BotSource.id == source_id).first()
    if not doc:
        raise ValueError(f"Document with ID '{source_id}' not found")

    chunks = db.query(models.DocumentChunk).filter(
        models.DocumentChunk.sourceId == doc.id
    ).order_by(models.DocumentChunk.createdAt.asc()).limit(15).all()

    if not chunks:
        raise ValueError("Document has no indexed text chunks to generate a course from.")

    context_text = "\n\n---\n\n".join([f"[Section {i+1}]: {c.content}" for i, c in enumerate(chunks)])
    user_prompt = f"Document Title: {doc.title}\nDocument Version: {doc.version}\n\nTechnical Content Excerpts:\n{context_text[:6000]}"

    # LLM Inference
    course_json = None
    messages = [
        {"role": "system", "content": AI_COURSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # Try NVIDIA NIM
    if nvidia_client:
        try:
            nv_model = NVIDIA_LLM_MODEL if NVIDIA_LLM_MODEL else "meta/llama-3.2-11b-vision-instruct"
            resp = nvidia_client.chat.completions.create(
                model=nv_model,
                messages=messages,
                temperature=0.2,
                max_tokens=2500,
                timeout=45.0
            )
            raw = resp.choices[0].message.content or ""
            # Clean possible markdown wrapping
            raw_clean = raw.strip()
            if raw_clean.startswith("```json"):
                raw_clean = raw_clean[7:]
            if raw_clean.startswith("```"):
                raw_clean = raw_clean[3:]
            if raw_clean.endswith("```"):
                raw_clean = raw_clean[:-3]
            course_json = json.loads(raw_clean.strip())
            logger.info(f"✅ AI Course successfully authored via NVIDIA NIM ({nv_model})")
        except Exception as e:
            logger.warning(f"⚠️ NVIDIA NIM course generation failed: {e}")

    # Fallback to Groq if needed
    if not course_json and groq_client:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.2,
                max_tokens=2500,
                timeout=25.0
            )
            raw = resp.choices[0].message.content or ""
            raw_clean = raw.strip()
            if raw_clean.startswith("```json"):
                raw_clean = raw_clean[7:]
            if raw_clean.startswith("```"):
                raw_clean = raw_clean[3:]
            if raw_clean.endswith("```"):
                raw_clean = raw_clean[:-3]
            course_json = json.loads(raw_clean.strip())
            logger.info("✅ AI Course successfully authored via Groq Fallback")
        except Exception as e:
            logger.warning(f"⚠️ Groq course generation fallback failed: {e}")

    # Fallback template if LLM is offline
    if not course_json:
        course_json = {
            "title": f"Technical Installation & Operation: {doc.title}",
            "description": f"Standard operating procedures and compliance guidelines derived from {doc.title} ({doc.version}).",
            "category": "Technical Operations",
            "passing_score": 80,
            "validity_months": 24,
            "modules": [
                {
                    "title": "Module 1: Core Specifications & Safety Protocol",
                    "lessons": [
                        {
                            "title": "Lesson 1.1: System Overview & Requirements",
                            "content_type": "DOC",
                            "duration_seconds": 300,
                            "content": chunks[0].content[:800] if len(chunks) > 0 else "Review documentation thoroughly."
                        },
                        {
                            "title": "Lesson 1.2: Standard Operating Procedure",
                            "content_type": "DOC",
                            "duration_seconds": 450,
                            "content": chunks[1].content[:800] if len(chunks) > 1 else "Follow manufacturer guidelines."
                        }
                    ]
                }
            ],
            "quiz": {
                "title": "Compliance & Safety Verification Quiz",
                "passing_score": 80,
                "questions": [
                    {
                        "question": f"Which document serves as the verified baseline for this procedure?",
                        "options": [doc.title, "Unverified General Standard", "Draft Notes", "Outdated Spec"],
                        "correct_option_index": 0,
                        "explanation": f"The approved manual {doc.title} ({doc.version}) is the sole verified standard."
                    }
                ]
            }
        }

    # Persist the newly generated course into PostgreSQL
    course = models.Course(
        orgId=org_id,
        title=course_json.get("title", f"Course: {doc.title}"),
        description=course_json.get("description", "Auto-generated technical training course."),
        category=course_json.get("category", "Engineering"),
        passingScore=course_json.get("passing_score", 80),
        validityMonths=course_json.get("validity_months", 24),
        sourceDocId=doc.id,
        isPublished=False # Stays draft until human trainer review
    )
    db.add(course)
    db.commit()
    db.refresh(course)

    # Persist Modules & Lessons
    modules_data = course_json.get("modules", [])
    for m_idx, m_item in enumerate(modules_data):
        module = models.CourseModule(
            courseId=course.id,
            title=m_item.get("title", f"Module {m_idx + 1}"),
            orderIndex=m_idx
        )
        db.add(module)
        db.commit()
        db.refresh(module)

        for l_idx, l_item in enumerate(m_item.get("lessons", [])):
            lesson = models.Lesson(
                moduleId=module.id,
                title=l_item.get("title", f"Lesson {m_idx + 1}.{l_idx + 1}"),
                content=l_item.get("content", "Review source technical documentation."),
                contentType=l_item.get("content_type", "DOC"),
                durationSeconds=l_item.get("duration_seconds", 300),
                orderIndex=l_idx
            )
            db.add(lesson)
        db.commit()

    # Persist Quiz
    quiz_data = course_json.get("quiz")
    if quiz_data:
        quiz = models.Quiz(
            courseId=course.id,
            title=quiz_data.get("title", "Assessment Quiz"),
            passingScore=quiz_data.get("passing_score", 80)
        )
        db.add(quiz)
        db.commit()
        db.refresh(quiz)

        for q_item in quiz_data.get("questions", []):
            qq = models.QuizQuestion(
                quizId=quiz.id,
                question=q_item.get("question", "Verification Question"),
                optionsJson=json.dumps(q_item.get("options", ["A", "B", "C", "D"])),
                correctOptionIndex=q_item.get("correct_option_index", 0),
                explanation=q_item.get("explanation", "Grounded in verified technical documentation.")
            )
            db.add(qq)
        db.commit()

    log_audit_event(
        db=db,
        org_id=org_id,
        action="AI_COURSE_AUTHORED",
        resource_type="course",
        resource_id=course.id,
        user=current_user,
        details={
            "source_doc_title": doc.title,
            "generated_course_title": course.title,
            "modules_count": len(modules_data)
        },
        ip_address=client_ip
    )

    return {
        "status": "success",
        "course_id": course.id,
        "title": course.title,
        "category": course.category,
        "modules_count": len(modules_data),
        "is_published": course.isPublished,
        "generated_syllabus": course_json
    }

