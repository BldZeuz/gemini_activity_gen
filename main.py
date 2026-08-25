from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
import uuid
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, field_validator

from curriculum import (
    G1_NOTE,
    MATATAG_COMPETENCIES,
    GroupedReadingCompetency,
    MatatagCompetency,
    get_grouped_competency,
    grouped_competencies_for_grade,
    list_competencies,
    records_for_group,
    unique_alignment_statements,
)


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "MATATAG Adaptive Reading Bundle Generator API"
APP_VERSION = "3.0.0"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
APP_API_KEY = os.getenv("APP_API_KEY", "").strip()

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

MAX_VARIANTS_PER_LEVEL = 5
MAX_FOLLOW_UP_QUESTIONS = 4
MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "32768"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("matatag_bundle_api")


# ============================================================
# TYPES / CONSTANTS
# ============================================================

Difficulty = Literal["easy", "medium", "hard"]
CompetencyKey = Literal[
    "foundational_reading",
    "reading_fluency",
    "reading_comprehension",
]
ActivityType = Literal[
    "word_reading",
    "sight_word_reading",
    "phonics_reading",
    "sentence_reading",
    "passage_reading",
    "timed_reading",
    "repeated_reading",
    "reading_comprehension",
]

DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")

# These are application difficulty bands, NOT official DepEd/MATATAG thresholds.
# They are used to keep Easy/Medium/Hard generation consistently progressive.
# The speech/recommender layer should use actual learner performance to adapt.
READING_LENGTH_BANDS: dict[str, dict[int, dict[str, tuple[int, int]]]] = {
    "word_reading": {
        1: {"easy": (5, 7), "medium": (8, 10), "hard": (11, 14)},
        2: {"easy": (6, 8), "medium": (9, 12), "hard": (13, 16)},
        3: {"easy": (8, 10), "medium": (11, 14), "hard": (15, 20)},
    },
    "sight_word_reading": {
        1: {"easy": (5, 7), "medium": (8, 10), "hard": (11, 14)},
        2: {"easy": (6, 8), "medium": (9, 12), "hard": (13, 16)},
        3: {"easy": (8, 10), "medium": (11, 14), "hard": (15, 20)},
    },
    "phonics_reading": {
        1: {"easy": (5, 7), "medium": (8, 10), "hard": (11, 14)},
        2: {"easy": (6, 8), "medium": (9, 12), "hard": (13, 16)},
        3: {"easy": (8, 10), "medium": (11, 14), "hard": (15, 20)},
    },
    "sentence_reading": {
        1: {"easy": (5, 8), "medium": (9, 14), "hard": (15, 22)},
        2: {"easy": (8, 14), "medium": (15, 24), "hard": (25, 36)},
        3: {"easy": (12, 20), "medium": (21, 34), "hard": (35, 50)},
    },
    "passage_reading": {
        1: {"easy": (20, 30), "medium": (31, 45), "hard": (46, 65)},
        2: {"easy": (30, 45), "medium": (46, 70), "hard": (71, 100)},
        3: {"easy": (40, 60), "medium": (61, 90), "hard": (91, 130)},
    },
    "timed_reading": {
        1: {"easy": (30, 45), "medium": (46, 65), "hard": (66, 90)},
        2: {"easy": (45, 65), "medium": (66, 95), "hard": (96, 130)},
        3: {"easy": (60, 85), "medium": (86, 120), "hard": (121, 160)},
    },
    "repeated_reading": {
        1: {"easy": (30, 45), "medium": (46, 65), "hard": (66, 90)},
        2: {"easy": (45, 65), "medium": (66, 95), "hard": (96, 130)},
        3: {"easy": (60, 85), "medium": (86, 120), "hard": (121, 160)},
    },
    "reading_comprehension": {
        1: {"easy": (25, 35), "medium": (36, 50), "hard": (51, 70)},
        2: {"easy": (35, 50), "medium": (51, 75), "hard": (76, 105)},
        3: {"easy": (45, 65), "medium": (66, 95), "hard": (96, 135)},
    },
}

DIFFICULTY_GUIDANCE = {
    "easy": (
        "Use highly familiar, controlled, decodable language; short structures; "
        "low cognitive load; and direct support for the selected competency."
    ),
    "medium": (
        "Use grade-appropriate vocabulary and moderately longer structures while "
        "keeping the same competency and target skill."
    ),
    "hard": (
        "Use the upper end of grade-appropriate vocabulary, length, and sentence "
        "complexity, but DO NOT introduce a different or above-grade competency."
    ),
}


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================


class BundleRequest(BaseModel):
    """Teacher-facing request for one adaptive reading bundle.

    The teacher selects a grade, one of the three grouped reading competencies,
    an oral-reading-compatible activity type, and 1-5 variants per difficulty.
    Quarter and individual MATATAG codes are intentionally not required.
    """

    model_config = ConfigDict(extra="forbid")

    grade: Literal[1, 2, 3]
    competency: CompetencyKey
    activity_type: ActivityType
    variants_per_level: int = Field(default=3, ge=1, le=MAX_VARIANTS_PER_LEVEL)
    topic: str | None = Field(default=None, max_length=200)
    teacher_notes: str | None = Field(default=None, max_length=1000)

    @field_validator("topic", "teacher_notes", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("value must be text")
        value = value.strip()
        return value or None


class FollowUpQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500)
    choices: list[str] = Field(min_length=2, max_length=4)
    answer: str = Field(min_length=1, max_length=250)
    explanation: str = Field(min_length=1, max_length=500)


class GeneratedVariant(BaseModel):
    """Content Gemini is allowed to generate.

    difficulty, variant numbering, reference_text, and word_count are NOT trusted
    to Gemini. The server attaches/derives those fields after validation.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=140)
    instructions: str = Field(min_length=1, max_length=500)
    display_text: str = Field(min_length=1, max_length=6000)
    target_skills: list[str] = Field(min_length=1, max_length=6)
    reading_features: list[str] = Field(min_length=1, max_length=8)
    follow_up_questions: list[FollowUpQuestion] = Field(
        default_factory=list,
        max_length=MAX_FOLLOW_UP_QUESTIONS,
    )


class GeneratedLevels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    easy: list[GeneratedVariant]
    medium: list[GeneratedVariant]
    hard: list[GeneratedVariant]


class GeneratedBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_title: str = Field(min_length=1, max_length=160)
    levels: GeneratedLevels


class ActivityVariantResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    difficulty: Difficulty
    variant_number: int = Field(ge=1, le=MAX_VARIANTS_PER_LEVEL)
    variant_label: str
    title: str
    instructions: str
    display_text: str
    reference_text: str
    word_count: int = Field(ge=1)
    target_skills: list[str]
    reading_features: list[str]
    follow_up_questions: list[FollowUpQuestion]
    speech_ready: bool = True


class BundleLevelsResponse(BaseModel):
    easy: list[ActivityVariantResponse]
    medium: list[ActivityVariantResponse]
    hard: list[ActivityVariantResponse]


class MatatagAlignmentSummary(BaseModel):
    curriculum: str
    grade: int
    grouped_competency: str
    grouped_competency_label: str
    official_record_codes: list[str]
    official_quarters: list[int]
    source_titles: list[str]
    note: str | None = None


class ActivityBundleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_id: str
    bundle_title: str
    grade: Literal[1, 2, 3]
    competency: CompetencyKey
    competency_label: str
    activity_type: ActivityType
    variants_per_level: int
    total_activities: int
    levels: BundleLevelsResponse
    matatag_alignment: MatatagAlignmentSummary
    difficulty_policy_note: str
    speech_integration_note: str


class GroupedCompetencyResponse(BaseModel):
    key: str
    label: str
    grade: int
    description: str
    allowed_activity_types: list[str]
    matatag_record_count: int


class MatatagRecordResponse(BaseModel):
    code: str
    grade: int
    quarter: int
    learning_area: str
    subdomain: str
    competency: str
    source_title: str
    source_year: int
    language_note: str | None = None


class GroupAlignmentResponse(BaseModel):
    grade: int
    competency: str
    competency_label: str
    description: str
    records: list[MatatagRecordResponse]


class MetadataResponse(BaseModel):
    name: str
    version: str
    grades: list[int]
    competencies: list[str]
    activity_types: list[str]
    max_variants_per_level: int
    difficulty_levels: list[str]
    curriculum_record_count: int
    note: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    model: str
    curriculum_records: int


class GeminiDiagnosticResponse(BaseModel):
    ok: bool
    model: str
    sdk_version: str
    api_key_configured: bool
    gemini_code: int | None = None
    gemini_status: str | None = None
    gemini_message: str | None = None
    response_text: str | None = None


# ============================================================
# SECURITY / GEMINI HELPERS
# ============================================================


def require_internal_key(x_app_key: str | None = Header(default=None)) -> None:
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


def get_google_genai_version() -> str:
    try:
        return version("google-genai")
    except PackageNotFoundError:
        return "unknown"


def safe_gemini_error(exc: Exception) -> tuple[int | None, str | None, str]:
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
# CURRICULUM / GROUP HELPERS
# ============================================================


def grouped_competency_to_response(
    grade: int,
    item: GroupedReadingCompetency,
) -> GroupedCompetencyResponse:
    records = records_for_group(grade=grade, group_key=item.key)
    return GroupedCompetencyResponse(
        key=item.key,
        label=item.label,
        grade=grade,
        description=item.description,
        allowed_activity_types=list(item.allowed_activity_types),
        matatag_record_count=len(records),
    )


def record_to_response(item: MatatagCompetency) -> MatatagRecordResponse:
    return MatatagRecordResponse(
        code=item.code,
        grade=item.grade,
        quarter=item.quarter,
        learning_area=item.learning_area,
        subdomain=item.subdomain,
        competency=item.competency,
        source_title=item.source_title,
        source_year=item.source_year,
        language_note=item.language_note,
    )


def resolve_group_and_validate(request: BundleRequest) -> GroupedReadingCompetency:
    group = get_grouped_competency(request.competency)
    if group is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown grouped competency: {request.competency}",
        )

    if request.activity_type not in group.allowed_activity_types:
        allowed = ", ".join(group.allowed_activity_types)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Activity type '{request.activity_type}' is not allowed for "
                f"'{group.label}'. Allowed types: {allowed}."
            ),
        )

    alignment_records = records_for_group(grade=request.grade, group_key=group.key)
    if not alignment_records:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No MATATAG reading records are mapped to Grade {request.grade} / "
                f"{group.label}."
            ),
        )

    return group


def build_alignment_summary(
    *,
    grade: int,
    group: GroupedReadingCompetency,
) -> MatatagAlignmentSummary:
    records = records_for_group(grade=grade, group_key=group.key)
    return MatatagAlignmentSummary(
        curriculum="MATATAG K to 10 Curriculum",
        grade=grade,
        grouped_competency=group.key,
        grouped_competency_label=group.label,
        official_record_codes=[item.code for item in records],
        official_quarters=sorted({item.quarter for item in records}),
        source_titles=sorted({item.source_title for item in records}),
        note=(G1_NOTE if grade == 1 else None),
    )


# ============================================================
# READING CONTENT HELPERS
# ============================================================


_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def reference_text_from_display(text: str) -> str:
    """Create the speech-recognition reference text from displayed content.

    This intentionally preserves punctuation/casing but collapses whitespace so
    a word list rendered over multiple lines becomes a clean reference string.
    """
    return " ".join(text.split())


def comparable_text(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.casefold()))


def variant_label(number: int) -> str:
    return chr(ord("A") + number - 1)


def get_length_band(
    *,
    grade: int,
    activity_type: str,
    difficulty: Difficulty,
) -> tuple[int, int]:
    return READING_LENGTH_BANDS[activity_type][grade][difficulty]


def length_band_prompt(request: BundleRequest) -> str:
    lines: list[str] = []
    for difficulty in DIFFICULTIES:
        low, high = get_length_band(
            grade=request.grade,
            activity_type=request.activity_type,
            difficulty=difficulty,
        )
        lines.append(
            f"- {difficulty.upper()}: {low}-{high} spoken words. "
            f"{DIFFICULTY_GUIDANCE[difficulty]}"
        )
    return "\n".join(lines)


# ============================================================
# PROMPTING
# ============================================================


def build_prompt(
    request: BundleRequest,
    group: GroupedReadingCompetency,
) -> str:
    alignment_statements = unique_alignment_statements(
        grade=request.grade,
        group_key=group.key,
    )
    alignment_text = "\n".join(f"- {statement}" for statement in alignment_statements)

    topic_line = request.topic or "No optional theme; choose safe child-friendly contexts."
    notes_line = request.teacher_notes or "No additional teacher notes."

    grade_1_rule = ""
    if request.grade == 1:
        grade_1_rule = f"""
GRADE 1 LANGUAGE NOTE
- {G1_NOTE}
- Generate English reading practice as an application adaptation while preserving
  the foundational literacy intent of the underlying MATATAG records.
""".strip()

    comprehension_rule = (
        "Each variant MUST include 2-3 short follow-up comprehension questions with "
        "2-4 choices each. Every answer must be directly supported by display_text."
        if request.activity_type == "reading_comprehension"
        else "follow_up_questions MUST be an empty array for every variant."
    )

    activity_rules = {
        "word_reading": (
            "Generate a space/newline-separated list of real grade-appropriate words. "
            "Do not use sentences. Target accurate oral word reading."
        ),
        "sight_word_reading": (
            "Generate a list of familiar grade-appropriate sight/high-frequency words. "
            "Do not use sentences."
        ),
        "phonics_reading": (
            "Generate real decodable English words that practice grade-appropriate "
            "sound-letter and word-pattern skills. Avoid nonsense words."
        ),
        "sentence_reading": (
            "Generate one or more short connected sentences appropriate to the grade."
        ),
        "passage_reading": (
            "Generate one coherent short narrative or informational passage designed "
            "for oral reading."
        ),
        "timed_reading": (
            "Generate one coherent passage suitable for a timed oral-reading attempt. "
            "Do not mention a target WPM; the assessment engine measures performance."
        ),
        "repeated_reading": (
            "Generate one coherent passage suitable for intentionally reading the same "
            "text more than once to observe fluency change."
        ),
        "reading_comprehension": (
            "Generate one coherent passage that the learner reads aloud first, followed "
            "by comprehension questions answered after reading."
        ),
    }

    return f"""
You generate adaptive oral-reading activity BUNDLES for Grade 1-3 learners in a
Philippine reading intervention application.

TEACHER REQUEST
- Grade: {request.grade}
- Grouped competency: {group.label} ({group.key})
- Activity type: {request.activity_type}
- Variants required PER difficulty level: {request.variants_per_level}
- Optional topic/theme: {topic_line}
- Teacher notes: {notes_line}

INTERNAL MATATAG ALIGNMENT
The grouped competency is an application-facing grouping. Keep every generated
reading activity within the following Grade {request.grade} MATATAG reading intents:
{alignment_text}

{grade_1_rule}

BUNDLE REQUIREMENT
Generate exactly THREE difficulty levels: easy, medium, hard.
Each level must contain exactly {request.variants_per_level} DIFFERENT variants.
Total activities = {request.variants_per_level * 3}.

DIFFICULTY POLICY
The competency stays the SAME across easy, medium, and hard. Difficulty changes
only through controlled text length, familiarity, decoding load, vocabulary,
sentence complexity, and independence. Never turn a harder level into a different
skill or above-grade curriculum objective.

Length guidance (application difficulty bands; not official MATATAG thresholds):
{length_band_prompt(request)}

VARIANT POLICY
- Variants within one level must be equivalent in difficulty but use different text.
- Do not duplicate a passage, sentence, or word list anywhere in the bundle.
- Avoid near-duplicates that merely swap a name or one noun.
- Make every variant independently usable by the recommender.

ACTIVITY-TYPE RULE
{activity_rules[request.activity_type]}

SPEECH-RECOGNITION COMPATIBILITY
- display_text must contain ONLY the words/sentences/passage the child is expected
  to read aloud. Do not include headings, labels, numbering, answer choices, or
  instructions inside display_text.
- The server will derive reference_text automatically from display_text.
- Use ordinary English spelling and punctuation.
- Do not use emoji, IPA, slash-separated phonemes, bracket annotations, or symbols
  that a speech recognizer should not be expected to transcribe.

COMPREHENSION RULE
{comprehension_rule}

METADATA RULES
- target_skills: 1-4 concise reading skills actually practiced by this variant.
- reading_features: 1-6 concise observable text features, e.g. "CVC words",
  "short sentences", "high-frequency words", "two-sentence passage".
- Do not invent MATATAG codes or quarters in generated JSON. The server owns the
  trusted curriculum alignment metadata.

SAFETY / AGE APPROPRIATENESS
Use child-safe, culturally neutral or Philippine-friendly everyday contexts.
Avoid personally identifying information, brands, politics, stereotypes, sexual
content, graphic violence, frightening mature themes, and trick questions.

OUTPUT
Return only JSON matching the supplied structured-output schema.
""".strip()


# ============================================================
# GEMINI JSON SCHEMA
# ============================================================


def follow_up_question_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "choices": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {"type": "string"},
            },
            "answer": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": ["question", "choices", "answer", "explanation"],
    }


def variant_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "instructions": {"type": "string"},
            "display_text": {"type": "string"},
            "target_skills": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {"type": "string"},
            },
            "reading_features": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {"type": "string"},
            },
            "follow_up_questions": {
                "type": "array",
                "maxItems": MAX_FOLLOW_UP_QUESTIONS,
                "items": follow_up_question_schema(),
            },
        },
        "required": [
            "title",
            "instructions",
            "display_text",
            "target_skills",
            "reading_features",
            "follow_up_questions",
        ],
    }


def build_bundle_json_schema(variants_per_level: int) -> dict:
    level_schema = {
        "type": "array",
        "minItems": variants_per_level,
        "maxItems": variants_per_level,
        "items": variant_schema(),
    }

    return {
        "type": "object",
        "properties": {
            "bundle_title": {"type": "string"},
            "levels": {
                "type": "object",
                "properties": {
                    "easy": level_schema,
                    "medium": level_schema,
                    "hard": level_schema,
                },
                "required": ["easy", "medium", "hard"],
            },
        },
        "required": ["bundle_title", "levels"],
    }


# ============================================================
# SEMANTIC VALIDATION
# ============================================================


def level_variants(bundle: GeneratedBundle, difficulty: Difficulty) -> list[GeneratedVariant]:
    return getattr(bundle.levels, difficulty)


def validate_follow_up_question(question: FollowUpQuestion, label: str) -> list[str]:
    errors: list[str] = []
    normalized_choices = [choice.strip().casefold() for choice in question.choices]
    if len(normalized_choices) != len(set(normalized_choices)):
        errors.append(f"{label} contains duplicate answer choices.")
    if question.answer.strip().casefold() not in normalized_choices:
        errors.append(f"{label} answer must exactly match one of its choices.")
    return errors


def validate_generated_bundle(
    bundle: GeneratedBundle,
    request: BundleRequest,
) -> list[str]:
    errors: list[str] = []
    all_texts: dict[str, str] = {}
    average_lengths: dict[str, float] = {}

    for difficulty in DIFFICULTIES:
        variants = level_variants(bundle, difficulty)

        if len(variants) != request.variants_per_level:
            errors.append(
                f"{difficulty} must contain exactly {request.variants_per_level} variants; "
                f"got {len(variants)}."
            )
            continue

        counts: list[int] = []
        target_low, target_high = get_length_band(
            grade=request.grade,
            activity_type=request.activity_type,
            difficulty=difficulty,
        )
        # Give Gemini modest tolerance while preventing wildly wrong lengths.
        accepted_low = max(1, int(target_low * 0.80))
        accepted_high = max(target_high, int(target_high * 1.20))

        for index, variant in enumerate(variants, start=1):
            label = f"{difficulty} variant {variant_label(index)}"
            normalized = comparable_text(variant.display_text)
            if not normalized:
                errors.append(f"{label} has no readable words.")
                continue

            if normalized in all_texts:
                errors.append(
                    f"{label} duplicates reading text from {all_texts[normalized]}."
                )
            else:
                all_texts[normalized] = label

            count = word_count(variant.display_text)
            counts.append(count)
            if count < accepted_low or count > accepted_high:
                errors.append(
                    f"{label} has {count} words; expected roughly {target_low}-{target_high} "
                    f"for this grade/activity/difficulty."
                )

            # Prevent metadata repetition/noise.
            skills = [value.strip().casefold() for value in variant.target_skills]
            if len(skills) != len(set(skills)):
                errors.append(f"{label} contains duplicate target_skills.")

            features = [value.strip().casefold() for value in variant.reading_features]
            if len(features) != len(set(features)):
                errors.append(f"{label} contains duplicate reading_features.")

            if request.activity_type == "reading_comprehension":
                if not (2 <= len(variant.follow_up_questions) <= 3):
                    errors.append(
                        f"{label} must have 2-3 comprehension questions."
                    )
                for q_index, question in enumerate(variant.follow_up_questions, start=1):
                    errors.extend(
                        validate_follow_up_question(question, f"{label} question {q_index}")
                    )
            elif variant.follow_up_questions:
                errors.append(
                    f"{label} must have no follow_up_questions for {request.activity_type}."
                )

        if counts:
            average_lengths[difficulty] = sum(counts) / len(counts)

    # The level averages should become progressively longer. Length is not the
    # only difficulty factor, but this catches obvious inversions.
    if all(key in average_lengths for key in DIFFICULTIES):
        if not (
            average_lengths["easy"] < average_lengths["medium"]
            < average_lengths["hard"]
        ):
            errors.append(
                "Average spoken word count must progress Easy < Medium < Hard."
            )

    return errors


# ============================================================
# RESPONSE CONSTRUCTION
# ============================================================


def build_variant_response(
    *,
    difficulty: Difficulty,
    index: int,
    generated: GeneratedVariant,
) -> ActivityVariantResponse:
    reference_text = reference_text_from_display(generated.display_text)
    return ActivityVariantResponse(
        difficulty=difficulty,
        variant_number=index,
        variant_label=variant_label(index),
        title=generated.title.strip(),
        instructions=generated.instructions.strip(),
        display_text=generated.display_text.strip(),
        reference_text=reference_text,
        word_count=word_count(reference_text),
        target_skills=[value.strip() for value in generated.target_skills],
        reading_features=[value.strip() for value in generated.reading_features],
        follow_up_questions=generated.follow_up_questions,
        speech_ready=True,
    )


def build_response(
    *,
    request: BundleRequest,
    group: GroupedReadingCompetency,
    generated: GeneratedBundle,
) -> ActivityBundleResponse:
    level_data: dict[str, list[ActivityVariantResponse]] = {}
    for difficulty in DIFFICULTIES:
        level_data[difficulty] = [
            build_variant_response(
                difficulty=difficulty,
                index=index,
                generated=variant,
            )
            for index, variant in enumerate(
                level_variants(generated, difficulty),
                start=1,
            )
        ]

    return ActivityBundleResponse(
        generation_id=str(uuid.uuid4()),
        bundle_title=generated.bundle_title.strip(),
        grade=request.grade,
        competency=request.competency,
        competency_label=group.label,
        activity_type=request.activity_type,
        variants_per_level=request.variants_per_level,
        total_activities=request.variants_per_level * 3,
        levels=BundleLevelsResponse(**level_data),
        matatag_alignment=build_alignment_summary(grade=request.grade, group=group),
        difficulty_policy_note=(
            "Easy/Medium/Hard are application-defined adaptive difficulty bands. "
            "They are not official MATATAG proficiency cut scores."
        ),
        speech_integration_note=(
            "Send the learner's audio file plus the selected variant.reference_text "
            "to the reading-analysis API. The recommender should choose one stored "
            "variant at a time and use attempt history to adapt the next selection."
        ),
    )


# ============================================================
# GEMINI GENERATION
# ============================================================


async def generate_bundle_with_gemini(
    request: BundleRequest,
    group: GroupedReadingCompetency,
) -> ActivityBundleResponse:
    client = get_gemini_client()
    base_prompt = build_prompt(request, group)
    validation_feedback = ""

    for attempt in range(2):
        prompt = base_prompt
        if validation_feedback:
            prompt += (
                "\n\nREPAIR REQUIRED\n"
                "Your previous bundle failed server validation. Regenerate the ENTIRE "
                "bundle and fix every issue below:\n"
                + validation_feedback
            )

        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=build_bundle_json_schema(
                        request.variants_per_level
                    ),
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )

            if not response.text or not response.text.strip():
                raise ValueError("Gemini returned an empty response.")

            generated = GeneratedBundle.model_validate_json(response.text)

        except Exception as exc:
            logger.exception(
                "Gemini bundle generation error on attempt %s using model %s",
                attempt + 1,
                GEMINI_MODEL,
            )

            if isinstance(exc, genai_errors.APIError):
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=gemini_http_detail(exc),
                ) from exc

            if attempt == 0:
                validation_feedback = (
                    "- Return valid JSON matching the provided structured-output schema.\n"
                    "- Do not include Markdown fences or prose outside the JSON."
                )
                continue

            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=gemini_http_detail(exc),
            ) from exc

        errors = validate_generated_bundle(generated, request)
        if not errors:
            return build_response(request=request, group=group, generated=generated)

        validation_feedback = "\n".join(f"- {error}" for error in errors)
        logger.warning(
            "Generated bundle failed semantic validation on attempt %s: %s",
            attempt + 1,
            validation_feedback,
        )

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Gemini returned a bundle that failed validation twice.",
    )


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Generates adaptive Grade 1-3 oral-reading bundles aligned internally to "
        "a reading-focused MATATAG curriculum registry."
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
        "metadata": "/metadata",
        "competencies": "/competencies?grade=2",
        "generate_bundle": "/generate-bundle",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        model=GEMINI_MODEL,
        curriculum_records=len(MATATAG_COMPETENCIES),
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["system"])
async def metadata() -> MetadataResponse:
    return MetadataResponse(
        name=APP_NAME,
        version=APP_VERSION,
        grades=[1, 2, 3],
        competencies=[
            "foundational_reading",
            "reading_fluency",
            "reading_comprehension",
        ],
        activity_types=sorted(READING_LENGTH_BANDS.keys()),
        max_variants_per_level=MAX_VARIANTS_PER_LEVEL,
        difficulty_levels=list(DIFFICULTIES),
        curriculum_record_count=len(MATATAG_COMPETENCIES),
        note=(
            "Teachers select grouped reading competencies; quarter and individual "
            "MATATAG codes are retained only as internal alignment metadata."
        ),
    )


# ============================================================
# TEACHER-FACING COMPETENCY ROUTES
# ============================================================


@app.get(
    "/competencies",
    response_model=list[GroupedCompetencyResponse],
    tags=["competencies"],
)
async def get_grouped_competencies(
    grade: int = Query(..., ge=1, le=3),
) -> list[GroupedCompetencyResponse]:
    return [
        grouped_competency_to_response(grade, item)
        for item in grouped_competencies_for_grade(grade)
    ]


@app.get(
    "/competencies/{grade}/{competency_key}",
    response_model=GroupedCompetencyResponse,
    tags=["competencies"],
)
async def get_grouped_competency_detail(
    grade: int,
    competency_key: str,
) -> GroupedCompetencyResponse:
    if grade not in {1, 2, 3}:
        raise HTTPException(status_code=404, detail="Grade must be 1, 2, or 3.")

    item = get_grouped_competency(competency_key)
    if item is None:
        raise HTTPException(status_code=404, detail="Unknown grouped competency.")

    return grouped_competency_to_response(grade, item)


# ============================================================
# MATATAG ALIGNMENT / AUDIT ROUTES
# ============================================================


@app.get(
    "/matatag/alignment",
    response_model=GroupAlignmentResponse,
    tags=["matatag"],
)
async def matatag_group_alignment(
    grade: int = Query(..., ge=1, le=3),
    competency: str = Query(...),
) -> GroupAlignmentResponse:
    group = get_grouped_competency(competency)
    if group is None:
        raise HTTPException(status_code=404, detail="Unknown grouped competency.")

    records = records_for_group(grade=grade, group_key=group.key)
    return GroupAlignmentResponse(
        grade=grade,
        competency=group.key,
        competency_label=group.label,
        description=group.description,
        records=[record_to_response(item) for item in records],
    )


# Backward-safe audit route. Teacher UIs should use /competencies, not this.
@app.get(
    "/matatag/competencies",
    response_model=list[MatatagRecordResponse],
    tags=["matatag"],
)
async def raw_matatag_records(
    grade: int | None = Query(default=None, ge=1, le=3),
    quarter: int | None = Query(default=None, ge=1, le=4),
    subdomain: str | None = Query(default=None, max_length=120),
) -> list[MatatagRecordResponse]:
    values = list_competencies(
        grade=grade,
        quarter=quarter,
        subdomain=subdomain,
    )
    return [record_to_response(item) for item in values]


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
    base = {
        "model": GEMINI_MODEL,
        "sdk_version": get_google_genai_version(),
        "api_key_configured": bool(GEMINI_API_KEY),
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
            config=types.GenerateContentConfig(max_output_tokens=16),
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
# BUNDLE GENERATION ROUTES
# ============================================================


@app.post(
    "/generate-bundle",
    response_model=ActivityBundleResponse,
    tags=["bundles"],
    dependencies=[Depends(require_internal_key)],
)
async def generate_bundle(request: BundleRequest) -> ActivityBundleResponse:
    group = resolve_group_and_validate(request)
    return await generate_bundle_with_gemini(request, group)


# Alias retained so existing backend routing can be migrated gradually.
# It accepts the NEW v3 BundleRequest contract, not the old v2 activity payload.
@app.post(
    "/generate-activity",
    response_model=ActivityBundleResponse,
    tags=["bundles"],
    dependencies=[Depends(require_internal_key)],
    deprecated=True,
)
async def generate_activity_alias(request: BundleRequest) -> ActivityBundleResponse:
    group = resolve_group_and_validate(request)
    return await generate_bundle_with_gemini(request, group)
