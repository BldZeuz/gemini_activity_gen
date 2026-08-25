from __future__ import annotations

import asyncio
import hmac
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, field_validator

from curriculum import (
    MATATAG_COMPETENCIES,
    MatatagCompetency,
    get_competency,
    list_competencies,
)


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "MATATAG Grade 1-3 Reading Activity Generator API"
APP_VERSION = "2.2.0"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
APP_API_KEY = os.getenv("APP_API_KEY", "").strip()

DEBUG_GEMINI_ERRORS = os.getenv("DEBUG_GEMINI_ERRORS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_default_origins = (
    "http://localhost,"
    "http://127.0.0.1,"
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

MAX_ITEMS = 20

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("matatag_activity_api")


# ============================================================
# TYPES / CONSTANTS
# ============================================================

Difficulty = Literal["easy", "medium", "challenging"]

ActivityType = Literal[
    "multiple_choice",
    "rhyming",
    "phonics",
    "reading_comprehension",
    "fill_in_blank",
    "true_false",
    "word_matching",
    "sight_words",
    "sentence_building",
    "oral_reading",
    "word_sorting",
    "vocabulary",
    "sequencing",
]

CHOICE_BASED_ACTIVITY_TYPES = {
    "multiple_choice",
    "rhyming",
    "phonics",
    "word_matching",
    "sight_words",
    "sentence_building",
    "word_sorting",
    "vocabulary",
    "sequencing",
}


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ActivityRequest(BaseModel):
    """Request sent by the teacher-facing backend.

    The teacher selects a curriculum competency from this API's MATATAG
    registry. The client never supplies free-form curriculum text as the
    source of truth.
    """

    model_config = ConfigDict(extra="forbid")

    grade: Literal[1, 2, 3]
    quarter: Literal[1, 2, 3, 4]
    competency_code: str = Field(min_length=3, max_length=80)
    activity_type: ActivityType
    difficulty: Difficulty = "easy"
    number_of_items: int = Field(default=5, ge=1, le=MAX_ITEMS)
    topic: str | None = Field(default=None, max_length=200)
    teacher_notes: str | None = Field(default=None, max_length=1000)

    @field_validator("competency_code", mode="before")
    @classmethod
    def normalize_competency_code(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("competency_code must not be blank")
        return value.strip().upper()

    @field_validator("topic", "teacher_notes", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("value must be text")
        value = value.strip()
        return value or None


class GeneratedActivityItem(BaseModel):
    """Only the content Gemini is allowed to generate for an item."""

    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1, le=MAX_ITEMS)
    prompt: str = Field(min_length=1, max_length=800)
    choices: list[str] = Field(default_factory=list, max_length=8)
    answer: str = Field(min_length=1, max_length=300)
    explanation: str = Field(
        min_length=1,
        max_length=500,
        description="Short teacher-facing explanation of the correct answer.",
    )


class GeneratedActivity(BaseModel):
    """Structured output produced by Gemini.

    Curriculum metadata is intentionally excluded. The server adds trusted
    MATATAG metadata after generation so the model cannot invent or alter it.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=120)
    instructions: str = Field(min_length=1, max_length=500)
    passage: str | None = Field(
        default=None,
        max_length=2500,
        description="Short reading passage when needed; otherwise null.",
    )
    items: list[GeneratedActivityItem] = Field(min_length=1, max_length=MAX_ITEMS)


class ActivityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    grade: Literal[1, 2, 3]
    quarter: Literal[1, 2, 3, 4]
    competency_code: str
    competency: str
    learning_area: str
    subdomain: str
    activity_type: str
    difficulty: Difficulty
    instructions: str
    passage: str | None
    items: list[GeneratedActivityItem]
    curriculum_source: str
    alignment_note: str | None = None


class CompetencyResponse(BaseModel):
    code: str
    grade: int
    quarter: int
    learning_area: str
    subdomain: str
    competency: str
    allowed_activity_types: list[str]
    source_title: str
    source_year: int
    language_note: str | None = None


class MatatagMetadataResponse(BaseModel):
    name: str
    grades: list[int]
    quarters: list[int]
    competency_count: int
    activity_types: list[str]
    grade_1_note: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
    curriculum_records: int


# ============================================================
# SECURITY / CLIENT HELPERS
# ============================================================

def require_internal_key(x_app_key: str | None = Header(default=None)) -> None:
    """Protect generation from direct public use.

    Your Laravel/PHP/Node backend sends X-App-Key. Teachers and students should
    never need to know this value.
    """

    if not APP_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="APP_API_KEY is not configured on the server.",
        )

    if x_app_key is None or not hmac.compare_digest(x_app_key, APP_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-App-Key.",
        )


def get_gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY is not configured on the server.",
        )

    return genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# MATATAG HELPERS
# ============================================================

def competency_to_response(item: MatatagCompetency) -> CompetencyResponse:
    return CompetencyResponse(
        code=item.code,
        grade=item.grade,
        quarter=item.quarter,
        learning_area=item.learning_area,
        subdomain=item.subdomain,
        competency=item.competency,
        allowed_activity_types=list(item.allowed_activity_types),
        source_title=item.source_title,
        source_year=item.source_year,
        language_note=item.language_note,
    )


def resolve_and_validate_competency(request: ActivityRequest) -> MatatagCompetency:
    competency = get_competency(request.competency_code)

    if competency is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown MATATAG competency code: {request.competency_code}",
        )

    if competency.grade != request.grade:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"{competency.code} belongs to Grade {competency.grade}, "
                f"not Grade {request.grade}."
            ),
        )

    if competency.quarter != request.quarter:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"{competency.code} belongs to Quarter {competency.quarter}, "
                f"not Quarter {request.quarter}."
            ),
        )

    if request.activity_type not in competency.allowed_activity_types:
        allowed = ", ".join(competency.allowed_activity_types)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Activity type '{request.activity_type}' is not mapped to "
                f"{competency.code}. Allowed types: {allowed}."
            ),
        )

    return competency


# ============================================================
# PROMPTING
# ============================================================

def build_prompt(request: ActivityRequest, competency: MatatagCompetency) -> str:
    topic_line = request.topic or "No optional theme."
    notes_line = request.teacher_notes or "No additional teacher notes."

    grade_1_language_rule = ""
    if competency.grade == 1 and competency.language_note:
        grade_1_language_rule = f"""
GRADE 1 LANGUAGE NOTE
- The official Grade 1 MATATAG Reading and Literacy competency is designed for L1.
- This application is generating an English adaptation for foundational reading practice.
- Preserve the literacy skill itself; do not claim that this is an official Grade 1 English competency.
""".strip()

    return f"""
You are generating one teacher-reviewable reading activity for a Philippine
Grade {request.grade} learner.

LOCKED MATATAG CURRICULUM RECORD
- Competency code: {competency.code}
- Grade: {competency.grade}
- Quarter: {competency.quarter}
- Learning area: {competency.learning_area}
- Subdomain: {competency.subdomain}
- Competency: {competency.competency}
- Curriculum source: {competency.source_title}

REQUEST
- Activity type: {request.activity_type}
- Difficulty: {request.difficulty}
- Number of items: {request.number_of_items}
- Optional topic/theme: {topic_line}
- Teacher notes: {notes_line}

{grade_1_language_rule}

NON-NEGOTIABLE RULES
1. Every item must directly practice or assess competency {competency.code}.
2. Do not replace, broaden, invent, or reinterpret the selected competency.
3. Generate exactly {request.number_of_items} items numbered 1 through {request.number_of_items}.
4. Use age-appropriate English vocabulary, sentence length, and concepts for Grade {request.grade}.
5. Each scored item must have one unambiguous correct answer.
6. Keep distractors plausible but clearly incorrect.
7. Do not use trick questions or double negatives.
8. Avoid personally identifying information, brands, sexual content, graphic violence,
   discrimination, stereotypes, politics, or mature themes.
9. The optional topic is only a theme. If it conflicts with the competency, ignore the theme.
10. Teacher explanations must be short and factual.

ACTIVITY-TYPE RULES
- multiple_choice: use 3 or 4 choices; answer must exactly equal one choice.
- rhyming: directly test recognizing or producing rhymes as appropriate to the competency.
- phonics: focus on sound-letter or word-pattern skills required by the competency.
- reading_comprehension: provide one short passage; every answer must be supported by the passage.
- fill_in_blank: use a clear blank such as _____; choices are optional unless helpful.
- true_false: choices must be exactly ["True", "False"].
- word_matching: prompt is the left-side item; choices are possible matches; answer is the correct match.
- sight_words: directly practice recognition or meaning of the selected sight/high-frequency words skill.
- sentence_building: choices may be words/chunks used to construct the correct sentence.
- oral_reading: provide short words/sentences/passages suitable for oral reading; choices may be empty.
- word_sorting: provide category/group choices and one correct category for each item.
- vocabulary: test the vocabulary relationship required by the selected competency.
- sequencing: provide possible sequence/order choices in a simple child-friendly format.

OUTPUT RULES
- Return only the structured JSON required by the provided schema.
- Do not add curriculum metadata to the generated JSON; the server supplies trusted MATATAG metadata.
""".strip()


# ============================================================
# GENERATED-CONTENT VALIDATION
# ============================================================

def validate_generated_activity(
    activity: GeneratedActivity,
    request: ActivityRequest,
) -> list[str]:
    errors: list[str] = []

    if len(activity.items) != request.number_of_items:
        errors.append(
            f"Expected exactly {request.number_of_items} items, got {len(activity.items)}."
        )

    expected_numbers = list(range(1, request.number_of_items + 1))
    actual_numbers = [item.number for item in activity.items]
    if actual_numbers != expected_numbers:
        errors.append(
            f"Item numbers must be exactly {expected_numbers}; got {actual_numbers}."
        )

    if request.activity_type == "reading_comprehension" and not activity.passage:
        errors.append("Reading comprehension requires a non-empty passage.")

    if request.activity_type != "reading_comprehension" and activity.passage:
        # Not always an error: some competencies can legitimately benefit from a
        # very short supporting text. We intentionally allow it.
        pass

    for item in activity.items:
        normalized_choices = [choice.strip().casefold() for choice in item.choices]

        if len(normalized_choices) != len(set(normalized_choices)):
            errors.append(f"Item {item.number} contains duplicate choices.")

        if request.activity_type == "multiple_choice":
            if len(item.choices) not in {3, 4}:
                errors.append(
                    f"Item {item.number} must have exactly 3 or 4 choices for multiple choice."
                )

        if request.activity_type in CHOICE_BASED_ACTIVITY_TYPES:
            if len(item.choices) < 2:
                errors.append(
                    f"Item {item.number} requires at least 2 choices for {request.activity_type}."
                )

            if item.answer.strip().casefold() not in normalized_choices:
                errors.append(
                    f"Item {item.number} answer must exactly match one of its choices."
                )

        if request.activity_type == "true_false":
            if normalized_choices != ["true", "false"]:
                errors.append(
                    f"Item {item.number} true/false choices must be exactly ['True', 'False']."
                )

            if item.answer.strip().casefold() not in {"true", "false"}:
                errors.append(
                    f"Item {item.number} has an invalid true/false answer."
                )

        if request.activity_type == "fill_in_blank" and "_____" not in item.prompt:
            errors.append(
                f"Item {item.number} fill-in-the-blank prompt should contain _____."
            )

    return errors



class GeminiDiagnosticResponse(BaseModel):
    ok: bool
    model: str
    sdk_version: str
    api_key_configured: bool
    api_key_prefix: str | None = None
    gemini_code: int | None = None
    gemini_status: str | None = None
    gemini_message: str | None = None
    response_text: str | None = None


def get_google_genai_version() -> str:
    try:
        return version("google-genai")
    except PackageNotFoundError:
        return "unknown"


def safe_gemini_error(exc: Exception) -> tuple[int | None, str | None, str]:
    """Extract useful Gemini error information without exposing credentials."""
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        api_status = getattr(exc, "status", None)
        message = getattr(exc, "message", None) or "Gemini API request failed."
        return code, api_status, str(message)

    return None, None, f"{type(exc).__name__}: {str(exc)}"


def gemini_http_detail(exc: Exception) -> str:
    code, api_status, message = safe_gemini_error(exc)

    if code is not None:
        status_text = f" {api_status}" if api_status else ""
        return f"Gemini API error {code}{status_text}: {message}"

    return f"Gemini generation failed: {message}"


# ============================================================
# GEMINI GENERATION
# ============================================================

async def generate_with_gemini(
    request: ActivityRequest,
    competency: MatatagCompetency,
) -> ActivityResponse:
    client = get_gemini_client()
    base_prompt = build_prompt(request, competency)
    validation_feedback = ""

    # Allow one repair attempt if Gemini returns structurally valid JSON that
    # fails our application-level semantic checks.
    for attempt in range(2):
        prompt = base_prompt

        if validation_feedback:
            prompt += (
                "\n\nREPAIR REQUIRED\n"
                "Your previous response failed server validation. Regenerate the entire "
                "activity and fix every issue below:\n"
                + validation_feedback
            )

        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeneratedActivity,
                    temperature=0.35,
                    max_output_tokens=8192,
                ),
            )

            if not response.text or not response.text.strip():
                raise ValueError("Gemini returned an empty response.")

            generated = GeneratedActivity.model_validate_json(response.text)

        except Exception as exc:
            logger.exception(
                "Gemini generation error on attempt %s using model %s",
                attempt + 1,
                GEMINI_MODEL,
            )

            if attempt == 0:
                validation_feedback = (
                    "- Return valid JSON matching the provided structured-output schema.\n"
                    "- Do not include Markdown code fences or prose outside the JSON."
                )
                continue

            # APIError exposes the actual HTTP code/status/message from Gemini.
            # These fields are safe to show to the developer and do not include
            # the API key. This makes 400 vs 401 vs 403 vs 429 immediately clear.
            detail = gemini_http_detail(exc)

            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=detail,
            ) from exc

        errors = validate_generated_activity(generated, request)
        if not errors:
            return ActivityResponse(
                title=generated.title,
                grade=competency.grade,
                quarter=competency.quarter,
                competency_code=competency.code,
                competency=competency.competency,
                learning_area=competency.learning_area,
                subdomain=competency.subdomain,
                activity_type=request.activity_type,
                difficulty=request.difficulty,
                instructions=generated.instructions,
                passage=generated.passage,
                items=generated.items,
                curriculum_source=competency.source_title,
                alignment_note=competency.language_note,
            )

        validation_feedback = "\n".join(f"- {error}" for error in errors)
        logger.warning(
            "Generated activity failed semantic validation on attempt %s: %s",
            attempt + 1,
            validation_feedback,
        )

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Gemini returned an activity that failed validation twice.",
    )


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Generates teacher-reviewable Grade 1-3 reading activities aligned to "
        "a reading-focused MATATAG competency registry."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-App-Key"],
)


# ============================================================
# SYSTEM ROUTES
# ============================================================

@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "matatag_metadata": "/matatag/metadata",
        "matatag_competencies": "/matatag/competencies",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=GEMINI_MODEL,
        curriculum_records=len(MATATAG_COMPETENCIES),
    )


# ============================================================
# MATATAG ROUTES
# ============================================================

@app.get(
    "/matatag/metadata",
    response_model=MatatagMetadataResponse,
    tags=["matatag"],
)
async def matatag_metadata() -> MatatagMetadataResponse:
    activity_types = sorted(
        {
            activity_type
            for item in MATATAG_COMPETENCIES
            for activity_type in item.allowed_activity_types
        }
    )

    return MatatagMetadataResponse(
        name="MATATAG reading-focused activity registry",
        grades=[1, 2, 3],
        quarters=[1, 2, 3, 4],
        competency_count=len(MATATAG_COMPETENCIES),
        activity_types=activity_types,
        grade_1_note=(
            "Grade 1 records come from MATATAG Reading and Literacy (L1). "
            "This English-reading application generates clearly labeled English "
            "adaptations of those foundational competencies."
        ),
    )


@app.get(
    "/matatag/competencies",
    response_model=list[CompetencyResponse],
    tags=["matatag"],
)
async def get_matatag_competencies(
    # Query-string values arrive as text (for example ?grade=2&quarter=1).
    # Use constrained ints here so FastAPI/Pydantic can coerce "2" -> 2
    # instead of rejecting it as a Literal mismatch.
    grade: int | None = Query(default=None, ge=1, le=3),
    quarter: int | None = Query(default=None, ge=1, le=4),
    subdomain: str | None = Query(default=None, max_length=120),
) -> list[CompetencyResponse]:
    values = list_competencies(
        grade=grade,
        quarter=quarter,
        subdomain=subdomain,
    )
    return [competency_to_response(item) for item in values]


@app.get(
    "/matatag/competencies/{competency_code}",
    response_model=CompetencyResponse,
    tags=["matatag"],
)
async def get_matatag_competency(competency_code: str) -> CompetencyResponse:
    competency = get_competency(competency_code)
    if competency is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown MATATAG competency code: {competency_code}",
        )

    return competency_to_response(competency)



# ============================================================
# GEMINI DIAGNOSTIC
# ============================================================

@app.get(
    "/gemini/diagnostic",
    response_model=GeminiDiagnosticResponse,
    tags=["system"],
    dependencies=[Depends(require_internal_key)],
)
async def gemini_diagnostic() -> GeminiDiagnosticResponse:
    """
    Test only Gemini authentication/model access.

    This deliberately does NOT use MATATAG, structured output, or the activity
    schema, so failures here point to the API key/model/quota rather than our
    curriculum-generation logic.
    """
    prefix = None
    if GEMINI_API_KEY:
        prefix = GEMINI_API_KEY[:5] + "…" if len(GEMINI_API_KEY) > 5 else "configured"

    base = {
        "model": GEMINI_MODEL,
        "sdk_version": get_google_genai_version(),
        "api_key_configured": bool(GEMINI_API_KEY),
        "api_key_prefix": prefix,
    }

    if not GEMINI_API_KEY:
        return GeminiDiagnosticResponse(
            ok=False,
            **base,
            gemini_message="GEMINI_API_KEY is not configured.",
        )

    client = get_gemini_client()

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents="Reply with exactly: OK",
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=16,
            ),
        )

        return GeminiDiagnosticResponse(
            ok=True,
            **base,
            response_text=(response.text or "").strip(),
        )

    except Exception as exc:
        logger.exception("Gemini diagnostic failed using model %s", GEMINI_MODEL)
        code, api_status, message = safe_gemini_error(exc)

        return GeminiDiagnosticResponse(
            ok=False,
            **base,
            gemini_code=code,
            gemini_status=api_status,
            gemini_message=message,
        )


# ============================================================
# GENERATION ROUTE
# ============================================================

@app.post(
    "/generate-activity",
    response_model=ActivityResponse,
    tags=["activities"],
    dependencies=[Depends(require_internal_key)],
)
async def generate_activity(request: ActivityRequest) -> ActivityResponse:
    competency = resolve_and_validate_competency(request)
    return await generate_with_gemini(request, competency)