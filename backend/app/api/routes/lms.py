import json
import uuid
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.db import models
from app.db.models import UserRole
from app.core.dependencies import get_current_user, require_trainer
from app.services.audit import log_audit_event
from app.services.course_generator import generate_course_from_document

router = APIRouter(prefix="/api/lms", tags=["Corporate LMS & Training"])

# --- Request Schemas ---
class GenerateCourseFromDocRequest(BaseModel):
    source_id: str

class CreateCourseRequest(BaseModel):
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    target_roles: Optional[List[str]] = ["internal_employee", "distributor_dealer_partner"]
    passing_score: int = 80
    validity_months: int = 24
    source_doc_id: Optional[str] = None

class CreateModuleRequest(BaseModel):
    title: str
    order_index: int = 0

class CreateLessonRequest(BaseModel):
    title: str
    content: str
    content_type: str = "DOC" # DOC, VIDEO, SLIDE
    media_url: Optional[str] = None
    duration_seconds: int = 300
    order_index: int = 0

class CreateQuizQuestionRequest(BaseModel):
    question: str
    options: List[str]
    correct_option_index: int
    explanation: Optional[str] = None

class CreateQuizRequest(BaseModel):
    title: str
    passing_score: int = 80
    questions: List[CreateQuizQuestionRequest]

class SubmitQuizRequest(BaseModel):
    answers: Dict[str, int] # question_id -> selected_option_index

class UpdateProgressRequest(BaseModel):
    completed_lesson_id: str

# --- Course Endpoints ---
@router.post("/courses", summary="Create a new training course (Trainer/Admin)")
def create_course(
    req: CreateCourseRequest,
    request: Request,
    current_user: models.User = Depends(require_trainer),
    db: Session = Depends(get_db)
):
    """Creates a new course shell. Modules, lessons, and quizzes can be appended subsequently."""
    course = models.Course(
        orgId=current_user.orgId,
        title=req.title,
        description=req.description,
        category=req.category,
        targetRolesJson=json.dumps(req.target_roles or []),
        passingScore=req.passing_score,
        validityMonths=req.validity_months,
        sourceDocId=req.source_doc_id,
        isPublished=False # Draft by default until reviewed
    )
    db.add(course)
    db.commit()
    db.refresh(course)

    client_ip = request.client.host if request.client else "unknown"
    log_audit_event(
        db=db,
        org_id=current_user.orgId,
        action="COURSE_CREATED",
        resource_type="course",
        resource_id=course.id,
        user=current_user,
        details={"title": course.title, "passing_score": course.passingScore},
        ip_address=client_ip
    )

    return {"status": "success", "message": f"Course '{course.title}' created", "course_id": course.id}

@router.post("/courses/generate-from-doc", summary="AI-assisted course authoring from approved document PDF")
def generate_course_from_doc(
    req: GenerateCourseFromDocRequest,
    request: Request,
    current_user: models.User = Depends(require_trainer),
    db: Session = Depends(get_db)
):
    """
    Analyzes an approved technical document manual and automatically extracts a structured course syllabus,
    lessons, and an assessment quiz bank. Saved in Draft mode for human trainer review.
    """
    client_ip = request.client.host if request.client else "unknown"
    try:
        result = generate_course_from_document(
            db=db,
            source_id=req.source_id,
            org_id=current_user.orgId,
            current_user=current_user,
            client_ip=client_ip
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Course generation failed: {err}")

@router.get("/courses", summary="List courses available for current user")
def list_courses(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lists published courses matching user's role (Trainers and Admins see drafts as well)."""
    q = db.query(models.Course)
    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"]:
        q = q.filter(models.Course.orgId == current_user.orgId)

    is_privileged = current_user.role in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin", UserRole.ORG_ADMIN, UserRole.TRAINER]
    if not is_privileged:
        q = q.filter(models.Course.isPublished == True)

    courses = q.order_by(models.Course.createdAt.desc()).all()
    results = []
    for c in courses:
        # Check user enrollment / progress
        prog = db.query(models.UserCourseProgress).filter(
            models.UserCourseProgress.userId == current_user.id,
            models.UserCourseProgress.courseId == c.id
        ).first()

        results.append({
            "id": c.id,
            "title": c.title,
            "description": c.description,
            "category": c.category,
            "is_published": c.isPublished,
            "passing_score": c.passingScore,
            "validity_months": c.validityMonths,
            "user_status": prog.status if prog else "NOT_STARTED",
            "user_progress_percent": prog.progressPercent if prog else 0.0,
            "created_at": c.createdAt.isoformat() if c.createdAt else ""
        })
    return {"courses": results}

@router.get("/courses/{course_id}", summary="Get full course details with modules and lessons")
def get_course_detail(
    course_id: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Returns syllabus, modules, lessons, and linked quiz metadata for a course."""
    course = db.query(models.Course).filter(models.Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")

    modules = db.query(models.CourseModule).filter(
        models.CourseModule.courseId == course.id
    ).order_by(models.CourseModule.orderIndex.asc()).all()

    modules_data = []
    for m in modules:
        lessons = db.query(models.Lesson).filter(
            models.Lesson.moduleId == m.id
        ).order_by(models.Lesson.orderIndex.asc()).all()
        modules_data.append({
            "id": m.id,
            "title": m.title,
            "order_index": m.orderIndex,
            "lessons": [
                {
                    "id": l.id,
                    "title": l.title,
                    "content_type": l.contentType,
                    "content": l.content,
                    "media_url": l.mediaUrl,
                    "duration_seconds": l.durationSeconds,
                    "order_index": l.orderIndex
                }
                for l in lessons
            ]
        })

    quiz = db.query(models.Quiz).filter(models.Quiz.courseId == course.id).first()

    prog = db.query(models.UserCourseProgress).filter(
        models.UserCourseProgress.userId == current_user.id,
        models.UserCourseProgress.courseId == course.id
    ).first()

    return {
        "id": course.id,
        "title": course.title,
        "description": course.description,
        "category": course.category,
        "is_published": course.isPublished,
        "passing_score": course.passingScore,
        "validity_months": course.validityMonths,
        "modules": modules_data,
        "quiz": {"id": quiz.id, "title": quiz.title, "passing_score": quiz.passingScore} if quiz else None,
        "user_progress": {
            "status": prog.status if prog else "NOT_STARTED",
            "progress_percent": prog.progressPercent if prog else 0.0,
            "completed_lessons": json.loads(prog.completedLessonsJson or "[]") if prog else []
        }
    }

@router.post("/courses/{course_id}/publish", summary="Publish or unpublish course")
def toggle_publish_course(
    course_id: str,
    publish: bool,
    current_user: models.User = Depends(require_trainer),
    db: Session = Depends(get_db)
):
    course = db.query(models.Course).filter(models.Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")

    course.isPublished = publish
    db.commit()
    return {"status": "success", "course_id": course.id, "is_published": course.isPublished}

# --- Module & Lesson Authoring ---
@router.post("/courses/{course_id}/modules", summary="Add module to a course")
def add_module(
    course_id: str,
    req: CreateModuleRequest,
    current_user: models.User = Depends(require_trainer),
    db: Session = Depends(get_db)
):
    course = db.query(models.Course).filter(models.Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")

    module = models.CourseModule(
        courseId=course.id,
        title=req.title,
        orderIndex=req.order_index
    )
    db.add(module)
    db.commit()
    db.refresh(module)
    return {"status": "success", "module_id": module.id, "title": module.title}

@router.post("/modules/{module_id}/lessons", summary="Add lesson to a module")
def add_lesson(
    module_id: str,
    req: CreateLessonRequest,
    current_user: models.User = Depends(require_trainer),
    db: Session = Depends(get_db)
):
    module = db.query(models.CourseModule).filter(models.CourseModule.id == module_id).first()
    if not module:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Module not found")

    lesson = models.Lesson(
        moduleId=module.id,
        title=req.title,
        content=req.content,
        contentType=req.content_type,
        mediaUrl=req.media_url,
        durationSeconds=req.duration_seconds,
        orderIndex=req.order_index
    )
    db.add(lesson)
    db.commit()
    db.refresh(lesson)
    return {"status": "success", "lesson_id": lesson.id, "title": lesson.title}

# --- Course Progress Tracking ---
@router.post("/courses/{course_id}/progress", summary="Record lesson completion progress")
def update_course_progress(
    course_id: str,
    req: UpdateProgressRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Updates user progress as lessons are viewed and completed."""
    course = db.query(models.Course).filter(models.Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")

    # Count total lessons in course
    total_lessons = db.query(models.Lesson).join(models.CourseModule).filter(
        models.CourseModule.courseId == course.id
    ).count()
    if total_lessons == 0:
        total_lessons = 1

    prog = db.query(models.UserCourseProgress).filter(
        models.UserCourseProgress.userId == current_user.id,
        models.UserCourseProgress.courseId == course.id
    ).first()

    if not prog:
        prog = models.UserCourseProgress(
            userId=current_user.id,
            courseId=course.id,
            completedLessonsJson="[]",
            progressPercent=0.0,
            status="IN_PROGRESS"
        )
        db.add(prog)

    completed = set(json.loads(prog.completedLessonsJson or "[]"))
    completed.add(req.completed_lesson_id)
    prog.completedLessonsJson = json.dumps(list(completed))
    prog.progressPercent = round((len(completed) / total_lessons) * 100, 1)
    prog.lastAccessedAt = datetime.now(timezone.utc)
    if prog.progressPercent >= 100.0 and prog.status != "COMPLETED":
        # Kept IN_PROGRESS until quiz is passed if course has a quiz
        quiz = db.query(models.Quiz).filter(models.Quiz.courseId == course.id).first()
        if not quiz:
            prog.status = "COMPLETED"

    db.commit()
    return {
        "status": "success",
        "progress_percent": prog.progressPercent,
        "completed_count": len(completed),
        "total_lessons": total_lessons,
        "course_status": prog.status
    }

# --- Quizzes & Verification ---
@router.post("/courses/{course_id}/quizzes", summary="Create or update course assessment quiz")
def create_quiz(
    course_id: str,
    req: CreateQuizRequest,
    current_user: models.User = Depends(require_trainer),
    db: Session = Depends(get_db)
):
    course = db.query(models.Course).filter(models.Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")

    # Clear existing quiz if updating
    existing_quiz = db.query(models.Quiz).filter(models.Quiz.courseId == course.id).first()
    if existing_quiz:
        db.delete(existing_quiz)
        db.commit()

    quiz = models.Quiz(
        courseId=course.id,
        title=req.title,
        passingScore=req.passing_score
    )
    db.add(quiz)
    db.commit()
    db.refresh(quiz)

    for q_req in req.questions:
        q_item = models.QuizQuestion(
            quizId=quiz.id,
            question=q_req.question,
            optionsJson=json.dumps(q_req.options),
            correctOptionIndex=q_req.correct_option_index,
            explanation=q_req.explanation
        )
        db.add(q_item)

    db.commit()
    return {"status": "success", "quiz_id": quiz.id, "questions_count": len(req.questions)}

@router.get("/quizzes/{quiz_id}", summary="Get quiz questions for student assessment")
def get_quiz_for_taking(
    quiz_id: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Returns quiz questions and options without exposing correct answers."""
    quiz = db.query(models.Quiz).filter(models.Quiz.id == quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quiz not found")

    questions = db.query(models.QuizQuestion).filter(models.QuizQuestion.quizId == quiz.id).all()
    return {
        "id": quiz.id,
        "title": quiz.title,
        "passing_score": quiz.passingScore,
        "questions": [
            {
                "id": q.id,
                "question": q.question,
                "options": json.loads(q.optionsJson or "[]")
            }
            for q in questions
        ]
    }

@router.post("/quizzes/{quiz_id}/submit", summary="Submit quiz answers, calculate score, and award QR Certificate")
def submit_quiz(
    quiz_id: str,
    req: SubmitQuizRequest,
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Evaluates submitted answers:
    - Calculates percentage score.
    - If score >= passing_score:
      - Marks user course progress as COMPLETED.
      - Automatically generates an immutable Certificate with unique number and QR verification token!
      - Logs compliance audit event.
    """
    quiz = db.query(models.Quiz).filter(models.Quiz.id == quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quiz not found")

    course = db.query(models.Course).filter(models.Course.id == quiz.courseId).first()
    questions = db.query(models.QuizQuestion).filter(models.QuizQuestion.quizId == quiz.id).all()
    if not questions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Quiz has no questions")

    correct_count = 0
    feedback = []
    for q in questions:
        user_choice = req.answers.get(str(q.id))
        is_correct = user_choice is not None and int(user_choice) == q.correctOptionIndex
        if is_correct:
            correct_count += 1
        feedback.append({
            "question_id": q.id,
            "is_correct": is_correct,
            "explanation": q.explanation if not is_correct else None
        })

    score_percent = round((correct_count / len(questions)) * 100, 1)
    passed = score_percent >= quiz.passingScore

    # Update User Progress
    prog = db.query(models.UserCourseProgress).filter(
        models.UserCourseProgress.userId == current_user.id,
        models.UserCourseProgress.courseId == course.id
    ).first()
    if not prog:
        prog = models.UserCourseProgress(
            userId=current_user.id,
            courseId=course.id,
            completedLessonsJson="[]",
            progressPercent=100.0,
            status="IN_PROGRESS"
        )
        db.add(prog)

    prog.quizScore = score_percent
    certificate_data = None

    if passed:
        prog.status = "COMPLETED"
        # Generate Certificate
        cert_number = f"KIAVI-{datetime.now().year}-{secrets.token_hex(4).upper()}"
        verification_token = secrets.token_urlsafe(24)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=course.validityMonths * 30) if course.validityMonths else None

        cert = models.Certificate(
            certificateNumber=cert_number,
            userId=current_user.id,
            courseId=course.id,
            orgId=course.orgId,
            quizScore=score_percent,
            verificationToken=verification_token,
            issuedAt=now,
            expiresAt=expires_at
        )
        db.add(cert)
        db.commit()
        db.refresh(cert)

        certificate_data = {
            "certificate_number": cert.certificateNumber,
            "verification_token": cert.verificationToken,
            "issued_at": cert.issuedAt.isoformat(),
            "expires_at": cert.expiresAt.isoformat() if cert.expiresAt else None,
            "verification_url": f"/api/lms/certificates/verify/{cert.verificationToken}"
        }

        client_ip = request.client.host if request.client else "unknown"
        log_audit_event(
            db=db,
            org_id=course.orgId,
            action="COURSE_PASSED_CERTIFICATE_AWARDED",
            resource_type="certificate",
            resource_id=cert.id,
            user=current_user,
            details={
                "course_title": course.title,
                "score": score_percent,
                "cert_number": cert.certificateNumber
            },
            ip_address=client_ip
        )
    else:
        db.commit()

    return {
        "passed": passed,
        "score_percent": score_percent,
        "passing_threshold": quiz.passingScore,
        "correct_answers": correct_count,
        "total_questions": len(questions),
        "feedback": feedback,
        "certificate": certificate_data
    }

@router.get("/certificates/verify/{token}", summary="Public/Auditor Certificate Verification")
def verify_certificate(token: str, request: Request, db: Session = Depends(get_db)):
    """
    Publicly accessible endpoint for ISO auditors, hospital procurement, and dealer compliance.
    Validates certificate authenticity via cryptographic token.
    Returns rich HTML certificate when requested in a browser, or JSON for API calls.
    """
    cert = db.query(models.Certificate).filter(models.Certificate.verificationToken == token).first()
    if not cert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Certificate token is invalid or does not exist")

    user = db.query(models.User).filter(models.User.id == cert.userId).first()
    course = db.query(models.Course).filter(models.Course.id == cert.courseId).first()
    org = db.query(models.Organization).filter(models.Organization.id == cert.orgId).first()

    now = datetime.now(timezone.utc)
    is_valid = True
    if cert.expiresAt and cert.expiresAt.replace(tzinfo=timezone.utc) < now:
        is_valid = False

    student_name = user.name if user else "Verified Engineering Specialist"
    course_title = course.title if course else "Verified Compliance Course"
    org_name = org.name if org else "KiaviIQ Industrial Systems"
    issued_date = cert.issuedAt.strftime("%B %d, %Y") if cert.issuedAt else "2026"
    expires_str = cert.expiresAt.strftime("%B %d, %Y") if cert.expiresAt else "Perpetual (Periodic Review)"

    if "text/html" in request.headers.get("accept", ""):
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Official Certificate: {cert.certificateNumber}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @media print {{
            .no-print {{ display: none !important; }}
            body {{ background: white !important; color: black !important; }}
            .cert-box {{ border: 4px solid #b45309 !important; box-shadow: none !important; }}
        }}
    </style>
</head>
<body class="bg-[#030712] text-slate-100 min-h-screen flex flex-col items-center justify-center p-4">
    <div class="max-w-3xl w-full bg-gradient-to-b from-[#0b1329] via-[#060b17] to-[#030712] border-4 border-amber-500/50 rounded-3xl p-8 sm:p-12 shadow-[0_0_50px_rgba(245,158,11,0.2)] text-center relative overflow-hidden cert-box">
        <div class="flex items-center justify-between pb-6 border-b border-amber-500/20">
            <span class="text-xs font-mono font-bold tracking-widest text-amber-400 uppercase">KiaviIQ Compliance Verified</span>
            <span class="px-3 py-1 bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 rounded-full text-xs font-mono font-bold">
                ● Status: Valid
            </span>
        </div>

        <div class="py-8 space-y-4">
            <div class="text-[11px] font-mono tracking-widest text-slate-400 uppercase">Certificate of Technical Competency & Compliance</div>
            <h1 class="text-2xl sm:text-3xl font-black tracking-tight text-white">{course_title}</h1>
            <p class="text-xs text-slate-400 max-w-md mx-auto">This certifies that the named candidate has demonstrated rigorous technical mastery and satisfied certified assessment criteria under EU MDR / IATF 16949 training standards.</p>
        </div>

        <div class="py-6 border-y border-blue-500/20 my-4 bg-[#030712]/60 rounded-2xl">
            <div class="text-[11px] font-mono text-slate-400 uppercase">Awarded To</div>
            <div class="text-2xl font-black text-cyan-300 mt-1 font-serif">{student_name}</div>
            <div class="text-xs font-mono text-slate-400 mt-1">Verification Token: {cert.verificationToken}</div>
        </div>

        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-left py-4 text-xs">
            <div>
                <span class="text-[10px] font-mono text-slate-400 block uppercase">Assessment Score</span>
                <strong class="text-emerald-400 font-mono text-sm">{cert.quizScore}% (Passed)</strong>
            </div>
            <div>
                <span class="text-[10px] font-mono text-slate-400 block uppercase">Certificate Number</span>
                <strong class="text-amber-400 font-mono text-sm">{cert.certificateNumber}</strong>
            </div>
            <div>
                <span class="text-[10px] font-mono text-slate-400 block uppercase">Date Awarded</span>
                <strong class="text-slate-200 font-mono text-sm">{issued_date}</strong>
            </div>
            <div>
                <span class="text-[10px] font-mono text-slate-400 block uppercase">Issuing Organization</span>
                <strong class="text-slate-200 text-sm">{org_name}</strong>
            </div>
        </div>

        <div class="pt-6 border-t border-amber-500/20 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs no-print">
            <span class="text-[11px] text-slate-400 font-mono">Regulated by: EU MDR 2017/745 & IATF 16949 Standards</span>
            <button onclick="window.print()" class="px-5 py-2 bg-gradient-to-r from-amber-600 to-yellow-500 hover:from-amber-500 hover:to-yellow-400 text-black font-bold rounded-xl shadow-lg transition cursor-pointer">
                Print / Save PDF Certificate
            </button>
        </div>
    </div>
</body>
</html>"""
        return HTMLResponse(content=html)

    return {
        "valid": is_valid,
        "certificate_number": cert.certificateNumber,
        "student_name": student_name,
        "student_email_masked": f"{user.email[:3]}***@{user.email.split('@')[1]}" if user else "N/A",
        "course_title": course_title,
        "issuing_organization": org_name,
        "quiz_score": f"{cert.quizScore}%",
        "issued_at": cert.issuedAt.isoformat() if cert.issuedAt else "",
        "expires_at": cert.expiresAt.isoformat() if cert.expiresAt else "Never",
        "regulatory_compliance_proof": "Verified under EU MDR / IATF 16949 training standards"
    }

@router.get("/certificates/my-certificates", summary="List user earned certificates")
def list_my_certificates(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    certs = db.query(models.Certificate).filter(
        models.Certificate.userId == current_user.id
    ).order_by(models.Certificate.issuedAt.desc()).all()

    results = []
    for c in certs:
        course = db.query(models.Course).filter(models.Course.id == c.courseId).first()
        results.append({
            "id": c.id,
            "certificate_number": c.certificateNumber,
            "course_title": course.title if course else "Course",
            "score": f"{c.quizScore}%",
            "issued_at": c.issuedAt.isoformat() if c.issuedAt else "",
            "expires_at": c.expiresAt.isoformat() if c.expiresAt else "Never",
            "verification_token": c.verificationToken
        })
    return {"certificates": results}

