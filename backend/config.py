"""Shared backend configuration values."""

import os

ALLOWED_COURSES = {
    "ai/ml": "AI/ML",
    "ai ml": "AI/ML",
    "aiml": "AI/ML",
    "iot": "IoT",
    "internet of things": "IoT",
    "cyber security": "Cyber Security",
    "cybersecurity": "Cyber Security",
    "data science": "Data Science",
    "datascience": "Data Science",
    "robotics": "Robotics",
    "web development": "Web development",
    "webdevelopment": "Web development",
    "cloud computing": "Cloud Computing",
    "cloudcomputing": "Cloud Computing",
}

STAFF_ROLES = {"instructor", "admin"}

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "uploads",
)
