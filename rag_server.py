import os
import uuid
import logging
from flask import Flask, request, Response, render_template, send_file
from flask_cors import CORS
from app import chat
from rag_working import (
    delete_document, list_document_names, get_courses, index_single_document, 
    update_document_status, init_db_tables, verify_user, create_user, 
    get_user_by_id, get_all_users, add_user_to_course, get_user_courses
)
from io import BytesIO
from html import escape
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
app.logger.setLevel(logging.INFO)

# Configure CORS for local development: allow frontend calls from any origin.
CORS(app, resources={r"/*": {"origins": "*"}})


@app.before_request
def log_route_call():
    """Log all incoming route calls with method and path."""
    query_params = dict(request.args) if request.args else {}
    payload = None
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        payload = request.get_json(silent=True)
    app.logger.info(
        "REQUEST %s %s | args=%s | json=%s",
        request.method,
        request.path,
        query_params,
        payload,
    )


@app.after_request
def log_route_response(response):
    """Log response status for each request."""
    app.logger.info(
        "RESPONSE %s %s -> %s",
        request.method,
        request.path,
        response.status,
    )
    return response

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database tables on startup
try:
    init_db_tables()
except Exception as e:
    print(f"Warning: Could not initialize database tables: {e}")


@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------
# AUTHENTICATION ENDPOINTS
# --------------------------------------------------
@app.route("/login", methods=["POST"])
def login_route():
    """Authenticate user with email and password."""
    data = request.json or {}
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    
    if not email or not password:
        return {"error": "email and password are required"}, 400
    
    user = verify_user(email, password)
    if user:
        return {"status": "success", "user": user}, 200
    return {"error": "Invalid email or password"}, 401


@app.route("/register", methods=["POST"])
def register_route():
    """Register a new user."""
    data = request.json or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    role = data.get("role", "student").strip().lower()
    
    if not all([name, email, password]):
        return {"error": "name, email, and password are required"}, 400
    
    if role not in ["student", "instructor", "admin"]:
        return {"error": "role must be student, instructor, or admin"}, 400
    
    user = create_user(name, email, password, role)
    if user:
        return {"status": "success", "user": user}, 201
    return {"error": "Email already exists or registration failed"}, 400


@app.route("/user/<int:user_id>", methods=["GET"])
def get_user_route(user_id):
    """Get user details by ID."""
    user = get_user_by_id(user_id)
    if user:
        return {"user": user}, 200
    return {"error": "User not found"}, 404


@app.route("/users", methods=["GET"])
def get_users_route():
    """Get all users (optional role filter)."""
    role = request.args.get("role", "").strip().lower()
    role = role if role in ["student", "instructor", "admin"] else None
    users = get_all_users(role)
    return {"users": users}, 200


@app.route("/user/<int:user_id>/courses", methods=["GET"])
def get_user_courses_route(user_id):
    """Get courses for a specific user."""
    courses = get_user_courses(user_id)
    return {"courses": courses}, 200


@app.route("/user/<int:user_id>/enroll", methods=["POST"])
def enroll_user_route(user_id):
    """Enroll user in a course."""
    data = request.json or {}
    course_id = data.get("course_id")
    
    if not course_id:
        return {"error": "course_id is required"}, 400
    
    try:
        course_id = int(course_id)
    except (ValueError, TypeError):
        return {"error": "course_id must be an integer"}, 400
    
    if add_user_to_course(user_id, course_id):
        return {"status": "success", "message": "User enrolled successfully"}, 200
    return {"error": "Enrollment failed"}, 400


@app.route("/courses", methods=["GET"])
def courses_route():
    """Return list of courses for document upload course selector."""
    try:
        courses = get_courses()
        return {"courses": courses}
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/upload_document", methods=["POST"])
def upload_document_route():
    """
    Upload one document and index it under the selected course.
    Form data: file (file), and either course_id (int) or course_name (string).
    """
    if "file" not in request.files:
        return {"error": "No file part"}, 400
    file = request.files["file"]
    if file.filename == "":
        return {"error": "No file selected"}, 400
    course_id_raw = request.form.get("course_id")
    course_name = str(request.form.get("course_name") or "").strip()

    selected_course = None
    if course_name:
        selected_course = course_name
    elif course_id_raw:
        try:
            course_id = int(course_id_raw)
        except (ValueError, TypeError):
            return {"error": "course_id must be an integer"}, 400

        valid_course_ids = {int(c["id"]) for c in get_courses() if "id" in c}
        if course_id not in valid_course_ids:
            return {"error": "Invalid course_id for upload"}, 400
        selected_course = course_id
    else:
        return {"error": "course_id or course_name is required"}, 400

    uploader_role = str(request.form.get("uploaded_by_role") or "").strip().lower()
    if uploader_role not in ("admin", "instructor", "student"):
        uploader_role = "instructor"

    safe_name = os.path.basename(file.filename) or "document"
    ext = os.path.splitext(safe_name)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    try:
        file.save(save_path)
    except Exception as e:
        return {"error": f"Failed to save file: {e}"}, 500

    initial_status = "approved" if uploader_role in ("admin", "instructor") else "pending"
    success, message, document_id = index_single_document(save_path, selected_course, initial_status=initial_status)
    if not success:
        return {"error": message, "document_id": document_id}, 400
    return {"status": "indexed", "message": message, "document_id": document_id}

chat_histories = {}
last_responses = {}


def markdown_to_story(text: str):
    """
    Convert a subset of markdown (**bold**, * bullets) into
    ReportLab story elements (Paragraphs and bullet lists).
    """
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    h1_style = styles.get("Heading1", normal_style)
    h2_style = styles.get("Heading2", normal_style)
    h3_style = styles.get("Heading3", normal_style)

    story = []
    bullet_items = []

    def flush_bullets():
        nonlocal bullet_items
        if bullet_items:
            list_items = [
                ListItem(Paragraph(item, normal_style)) for item in bullet_items
            ]
            story.append(ListFlowable(list_items, bulletType="bullet"))
            story.append(Spacer(1, 6))
            bullet_items = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            flush_bullets()
            story.append(Spacer(1, 8))
            continue

        stripped = line.lstrip()

        # Headings: #, ##, ###
        if stripped.startswith("### "):
            flush_bullets()
            heading_text = stripped[4:]
            safe = escape(heading_text)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            story.append(Paragraph(safe, h3_style))
            story.append(Spacer(1, 8))
            continue
        if stripped.startswith("## "):
            flush_bullets()
            heading_text = stripped[3:]
            safe = escape(heading_text)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            story.append(Paragraph(safe, h2_style))
            story.append(Spacer(1, 8))
            continue
        if stripped.startswith("# "):
            flush_bullets()
            heading_text = stripped[2:]
            safe = escape(heading_text)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            story.append(Paragraph(safe, h1_style))
            story.append(Spacer(1, 10))
            continue

        # Bullets: *, -
        is_bullet = stripped.startswith("* ") or stripped.startswith("- ")
        if is_bullet:
            item_text = stripped[2:]
            safe = escape(item_text)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            bullet_items.append(safe)
        else:
            flush_bullets()
            safe = escape(line)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            para = Paragraph(safe, normal_style)
            story.append(para)
            story.append(Spacer(1, 4))

    flush_bullets()
    return story

@app.route("/chat", methods=["POST"])
def chat_route():
    data = request.json or {}
    query = data.get("query")
    session_id = data.get("session_id")
    user_id = data.get("user_id")  # student id: retrieval limited to their enrolled courses
    user_role = data.get("user_role")
    selected_course = data.get("selected_course")
    if selected_course is None:
        # Backward compatibility for older clients
        selected_course = data.get("course_name")
    if selected_course is None:
        selected_course = data.get("course_id")

    if not query:
        return {"error": "query is required"}, 400

    if selected_course is None or not str(selected_course).strip():
        return {"error": "selected_course is required. Choose a course before chatting."}, 400

    # Enforce a single selected course per request.
    if isinstance(selected_course, (list, tuple, set, dict)):
        return {"error": "selected_course must be a single course value, not a list."}, 400

    allowed_courses = {
        "ai/ml": "AI/ML",
        "web development": "Web development",
        "cloud computing": "Cloud Computing",
        "data science": "Data science",
        "datascience": "Data science",
    }
    selected_course_key = str(selected_course).strip().lower()
    if selected_course_key not in allowed_courses:
        return {"error": "selected_course must be one of: AI/ML, Web development, Cloud Computing, Data science"}, 400
    selected_course = allowed_courses[selected_course_key]

    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    if user_id is not None:
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            user_id = None

    def stream_with_context(query, sid, uid, role, selected):
        collected = ""
        for chunk in chat(query, chat_histories[sid], user_id=uid, user_role=role, selected_course=selected, session_id=sid):
            if chunk:
                collected += chunk
            yield chunk

        last_responses[sid] = collected

    return Response(
        stream_with_context(query, session_id, user_id, user_role, selected_course),
        mimetype="text/event-stream",
    )


@app.route("/download_pdf", methods=["GET"])
def download_pdf():
    session_id = request.args.get("session_id")
    if not session_id:
        return {"error": "session_id is required"}, 400

    response_text = last_responses.get(session_id)
    if not response_text:
        return {"error": "No response available for this session"}, 400

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    story = markdown_to_story(response_text)
    doc.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="response.pdf",
        mimetype="application/pdf",
    )


@app.route("/list_documents", methods=["GET"])
def list_documents_route():
    """Return distinct file_name, source, document_id from document_chunks (same DB the app uses)."""
    try:
        docs = list_document_names()
        return {"documents": docs}
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/set_document_status", methods=["POST"])
def set_document_status_route():
    """
    Update indexed document status.
    Expected JSON: {"status": "pending|approved|rejected", "source": "..."} or {"file_name": "..."}
    """
    data = request.get_json(silent=True) or {}
    status = str(data.get("status") or "").strip().lower()
    file_name = data.get("file_name")
    source = data.get("source")

    if status not in ("pending", "approved", "rejected"):
        return {"error": "status must be one of pending, approved, rejected"}, 400

    if not file_name and not source:
        return {"error": "Provide file_name or source"}, 400

    updated_count = update_document_status(status=status, file_name=file_name, source=source)
    if updated_count == 0:
        return {"status": "not_found_or_error", "updated_count": 0}, 404

    return {"status": "updated", "updated_count": updated_count}


@app.route("/approve_document", methods=["POST"])
def approve_document_route():
    data = request.get_json(silent=True) or {}
    payload = {
        "status": "approved",
        "file_name": data.get("file_name"),
        "source": data.get("source"),
    }
    request_data = payload
    status = request_data["status"]
    updated_count = update_document_status(status=status, file_name=request_data.get("file_name"), source=request_data.get("source"))
    if updated_count == 0:
        return {"status": "not_found_or_error", "updated_count": 0}, 404
    return {"status": "updated", "updated_count": updated_count}


@app.route("/reject_document", methods=["POST"])
def reject_document_route():
    data = request.get_json(silent=True) or {}
    payload = {
        "status": "rejected",
        "file_name": data.get("file_name"),
        "source": data.get("source"),
    }
    request_data = payload
    status = request_data["status"]
    updated_count = update_document_status(status=status, file_name=request_data.get("file_name"), source=request_data.get("source"))
    if updated_count == 0:
        return {"status": "not_found_or_error", "updated_count": 0}, 404
    return {"status": "updated", "updated_count": updated_count}


@app.route("/delete_document", methods=["POST"])
def delete_document_route():
    """
    Delete a document's chunks from pgvector by document_id, file_name, and/or source.
    Expected JSON body: { "document_id": 1 } or { "file_name": "..." } or { "source": "..." }
    """
    data = request.get_json(silent=True) or {}
    document_id = data.get("document_id")
    file_name = data.get("file_name")
    source = data.get("source")

    if document_id is None and not file_name and not source:
        return {"error": "Provide document_id or file_name or source"}, 400

    if document_id is not None:
        try:
            document_id = int(document_id)
        except (ValueError, TypeError):
            return {"error": "document_id must be an integer"}, 400

    deleted_count = delete_document(document_id=document_id, file_name=file_name, source=source)
    if deleted_count == 0:
        return {"status": "not_found_or_error", "deleted_count": 0}, 404

    return {"status": "deleted", "deleted_count": deleted_count}

if __name__ == "__main__":
    app.run(debug=False, port=8000)