import json
import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from docx import Document


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "gemini-3.6-flash"
MAX_FILE_MB = 10


st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# API Key
# ============================================================

def get_api_key():
    """Read Gemini API key from Streamlit Secrets or environment."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


# ============================================================
# DOCX Text Extraction
# ============================================================

def extract_docx_text(data: bytes) -> str:
    """Extract text from paragraphs and tables in a DOCX file."""
    with tempfile.NamedTemporaryFile(
        suffix=".docx",
        delete=False
    ) as tmp:
        tmp.write(data)
        path = tmp.name

    try:
        document = Document(path)
        text_parts = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                text_parts.append(text)

        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    text_parts.append(" | ".join(cells))

        return "\n".join(text_parts)

    finally:
        Path(path).unlink(missing_ok=True)


# ============================================================
# JSON Parsing
# ============================================================

def parse_json_response(text: str) -> dict:
    """Parse Gemini JSON even if markdown fences are returned."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


# ============================================================
# Gemini Analysis
# ============================================================

def analyze_resume(uploaded_file, job_description: str) -> dict:
    """Analyze the uploaded resume with Gemini."""
    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY to "
            ".streamlit/secrets.toml locally or Streamlit Cloud Secrets."
        )

    file_bytes = uploaded_file.getvalue()

    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(
            f"File is too large. Please upload a file smaller than "
            f"{MAX_FILE_MB} MB."
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an expert ATS resume evaluator, recruiter, and career coach.

Analyze the resume carefully and provide an estimated ATS-readiness score.

IMPORTANT:
- This is an estimated ATS score, NOT the score from a specific commercial ATS.
- Never invent information about the candidate.
- Do not recommend keyword stuffing.
- Recommendations must be realistic and actionable.
- If a job description is supplied, prioritize alignment with it.
- If no job description is supplied, evaluate against general ATS best practices.
- Return ONLY valid JSON. Do not use Markdown or code fences.

JOB DESCRIPTION:
{job_description.strip() or "No job description provided."}

Use exactly this JSON structure:

{{
  "ats_score": 0,
  "score_summary": "Short overall assessment",
  "categories": [
    {{
      "name": "ATS Formatting",
      "score": 0,
      "feedback": "Specific feedback"
    }},
    {{
      "name": "Keywords and Skills",
      "score": 0,
      "feedback": "Specific feedback"
    }},
    {{
      "name": "Experience Impact",
      "score": 0,
      "feedback": "Specific feedback"
    }},
    {{
      "name": "Clarity and Readability",
      "score": 0,
      "feedback": "Specific feedback"
    }},
    {{
      "name": "Sections and Structure",
      "score": 0,
      "feedback": "Specific feedback"
    }}
  ],
  "strengths": [
    "Strength 1",
    "Strength 2",
    "Strength 3"
  ],
  "improvements": [
    {{
      "priority": "High",
      "issue": "Specific issue",
      "recommendation": "Specific recommendation",
      "example": "Example of a better version"
    }},
    {{
      "priority": "Medium",
      "issue": "Specific issue",
      "recommendation": "Specific recommendation",
      "example": "Example of a better version"
    }}
  ],
  "missing_keywords": [
    "keyword 1",
    "keyword 2"
  ],
  "quick_wins": [
    "Quick improvement 1",
    "Quick improvement 2",
    "Quick improvement 3"
  ],
  "ats_warnings": [
    "Warning 1",
    "Warning 2"
  ]
}}

SCORING:
- ats_score: integer from 0 to 100.
- Every category score: integer from 0 to 100.
- Be conservative rather than giving inflated scores.
- Consider standard headings, parsing-friendly formatting, keyword relevance,
  measurable achievements, skills, experience relevance, clarity, and structure.
"""

    suffix = Path(uploaded_file.name).suffix.lower()

    # --------------------------------------------------------
    # PDF: Gemini Files API
    # --------------------------------------------------------

    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        try:
            uploaded = client.files.upload(file=temp_path)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, uploaded],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

    # --------------------------------------------------------
    # DOCX: Extract text locally
    # --------------------------------------------------------

    elif suffix == ".docx":
        resume_text = extract_docx_text(file_bytes)

        if not resume_text.strip():
            raise ValueError(
                "The DOCX file does not contain readable text."
            )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                prompt,
                "RESUME TEXT:\n" + resume_text,
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

    else:
        raise ValueError(
            "Unsupported file type. Please upload a PDF or DOCX."
        )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return parse_json_response(response.text)


# ============================================================
# UI Helpers
# ============================================================

def score_status(score: int) -> str:
    if score >= 80:
        return "🟢 Strong"
    if score >= 60:
        return "🟡 Needs Improvement"
    return "🔴 Weak"


def safe_score(value, default=0) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return default


# ============================================================
# Main UI
# ============================================================

st.title("📄 ATS Resume Analyzer")

st.markdown(
    """
Upload your resume and get an **AI-powered ATS readiness analysis**,
including your estimated score, strengths, missing keywords,
ATS warnings, and specific improvements.
"""
)

st.info(
    "💡 Tip: Add a job description for a more targeted keyword "
    "and job-match analysis."
)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    st.write(f"**AI Model:** `{MODEL_NAME}`")
    st.write(f"**Maximum file size:** `{MAX_FILE_MB} MB`")

    st.divider()

    st.caption(
        "Your Gemini API key is read from Streamlit Secrets "
        "or the GEMINI_API_KEY environment variable."
    )

    st.caption(
        "ATS scores are estimates and should not be treated "
        "as guarantees from a specific ATS platform."
    )


# ------------------------------------------------------------
# Resume Upload
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "📎 Upload your resume",
    type=["pdf", "docx"],
    help="Supported formats: PDF and DOCX. Maximum size: 10 MB.",
)


# ------------------------------------------------------------
# Job Description
# ------------------------------------------------------------

job_description = st.text_area(
    "💼 Job description (optional)",
    height=220,
    placeholder=(
        "Paste the job description here. "
        "The analyzer will compare your resume against it "
        "and identify relevant/missing keywords."
    ),
)


# ------------------------------------------------------------
# Analyze
# ------------------------------------------------------------

if st.button(
    "🚀 Analyze Resume",
    type="primary",
    use_container_width=True,
):

    if uploaded_file is None:
        st.warning("Please upload a PDF or DOCX resume first.")

    else:
        with st.spinner(
            "Analyzing your resume with Gemini 2.5 Flash..."
        ):
            try:
                analysis = analyze_resume(
                    uploaded_file,
                    job_description,
                )

                st.session_state["analysis"] = analysis

            except json.JSONDecodeError:
                st.error(
                    "Gemini returned an invalid JSON response. "
                    "Please try again."
                )

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


# ============================================================
# Results
# ============================================================

analysis = st.session_state.get("analysis")

if analysis:

    st.divider()

    # --------------------------------------------------------
    # Overall Score
    # --------------------------------------------------------

    score = safe_score(
        analysis.get("ats_score", 0)
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            "Estimated ATS Score",
            f"{score}/100",
        )

        st.progress(score / 100)

        st.write(
            f"**Status:** {score_status(score)}"
        )

    with col2:
        st.subheader("Overall Assessment")

        st.write(
            analysis.get(
                "score_summary",
                "No summary available.",
            )
        )

        st.caption(
            "This score represents AI-estimated ATS readiness "
            "based on the resume content and formatting signals "
            "available to the model."
        )

    # --------------------------------------------------------
    # Score Breakdown
    # --------------------------------------------------------

    st.subheader("📊 Score Breakdown")

    categories = analysis.get("categories", [])

    if categories:

        columns = st.columns(
            min(len(categories), 5)
        )

        for column, category in zip(columns, categories):

            category_score = safe_score(
                category.get("score", 0)
            )

            with column:
                st.metric(
                    category.get(
                        "name",
                        "Category",
                    ),
                    f"{category_score}/100",
                )

                st.progress(
                    category_score / 100
                )

                st.caption(
                    category.get(
                        "feedback",
                        "",
                    )
                )

    # --------------------------------------------------------
    # Strengths
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("✅ Strengths")

        strengths = analysis.get(
            "strengths",
            []
        )

        if strengths:
            for strength in strengths:
                st.write(f"• {strength}")
        else:
            st.write("No strengths were returned.")

    # --------------------------------------------------------
    # Quick Wins
    # --------------------------------------------------------

    with right:

        st.subheader("⚡ Quick Wins")

        quick_wins = analysis.get(
            "quick_wins",
            []
        )

        if quick_wins:
            for item in quick_wins:
                st.write(f"• {item}")
        else:
            st.write("No quick wins were returned.")

    # --------------------------------------------------------
    # Missing Keywords
    # --------------------------------------------------------

    st.subheader("🔑 Missing / Recommended Keywords")

    keywords = analysis.get(
        "missing_keywords",
        []
    )

    if keywords:
        keyword_text = " ".join(
            f"`{keyword}`"
            for keyword in keywords
        )
        st.markdown(keyword_text)
    else:
        st.success(
            "No major missing keywords were identified."
        )

    # --------------------------------------------------------
    # ATS Warnings
    # --------------------------------------------------------

    st.subheader("⚠️ ATS Warnings")

    warnings = analysis.get(
        "ats_warnings",
        []
    )

    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success(
            "No major ATS warnings were identified."
        )

    # --------------------------------------------------------
    # Detailed Improvements
    # --------------------------------------------------------

    st.subheader("🛠️ Recommended Improvements")

    improvements = analysis.get(
        "improvements",
        []
    )

    if improvements:

        for index, item in enumerate(improvements, 1):

            priority = item.get(
                "priority",
                "Medium",
            )

            issue = item.get(
                "issue",
                f"Improvement {index}",
            )

            with st.expander(
                f"{priority}: {issue}"
            ):

                recommendation = item.get(
                    "recommendation",
                    "",
                )

                example = item.get(
                    "example",
                    "",
                )

                if recommendation:
                    st.markdown(
                        f"**Recommendation:** {recommendation}"
                    )

                if example:
                    st.markdown(
                        f"**Example:** {example}"
                    )

    else:
        st.write(
            "No detailed improvements were returned."
        )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    st.divider()

    if st.button("🔄 Clear Analysis"):
        st.session_state.pop("analysis", None)
        st.rerun()
