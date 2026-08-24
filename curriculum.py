from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


ActivityType = str


@dataclass(frozen=True)
class MatatagCompetency:
    code: str
    grade: int
    quarter: int
    learning_area: str
    subdomain: str
    competency: str
    allowed_activity_types: tuple[ActivityType, ...]
    source_title: str
    source_year: int = 2023
    language_note: str | None = None


G1_SOURCE = "MATATAG Curriculum: Reading and Literacy (Grade 1), August 2023"
G23_SOURCE = "MATATAG Curriculum: English (Grades 2-10), August 2023"
G1_NOTE = (
    "MATATAG Grade 1 Reading and Literacy is formally designed for the learner's "
    "first language (L1). This service generates an English adaptation of the "
    "selected foundational competency for this application's English-reading use case. "
    "English becomes a MATATAG learning area starting in Grade 2."
)


def c(
    code: str,
    grade: int,
    quarter: int,
    learning_area: str,
    subdomain: str,
    competency: str,
    activity_types: Iterable[ActivityType],
    *,
    source_title: str,
    language_note: str | None = None,
) -> MatatagCompetency:
    return MatatagCompetency(
        code=code,
        grade=grade,
        quarter=quarter,
        learning_area=learning_area,
        subdomain=subdomain,
        competency=competency,
        allowed_activity_types=tuple(activity_types),
        source_title=source_title,
        language_note=language_note,
    )


# Reading-focused MATATAG subset for this application. This intentionally excludes
# competencies that are mainly oral communication/composition and not central to
# the app's reading support scope.
MATATAG_COMPETENCIES: tuple[MatatagCompetency, ...] = (
    # ------------------------------------------------------------------
    # GRADE 1 — READING AND LITERACY (L1); generated content is an English
    # adaptation and must be labeled as such by the API.
    # ------------------------------------------------------------------
    c("RL1PA-I-1", 1, 1, "Reading and Literacy (L1)", "Phonological Awareness", "Chant nursery rhymes and poems.", ["rhyming", "oral_reading"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1PA-I-2", 1, 1, "Reading and Literacy (L1)", "Phonological Awareness", "Segment two- to three-syllable words into syllables.", ["phonics", "word_sorting", "multiple_choice"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1PA-I-3", 1, 1, "Reading and Literacy (L1)", "Phonological Awareness", "Identify rhyming words in nursery rhymes, poems, and chants.", ["rhyming", "multiple_choice", "true_false"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1PA-I-4", 1, 1, "Reading and Literacy (L1)", "Phonological Awareness", "Produce two or three words that rhyme.", ["rhyming", "fill_in_blank", "word_matching"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1PA-I-5", 1, 1, "Reading and Literacy (L1)", "Phonological Awareness", "Identify initial speech sounds.", ["phonics", "multiple_choice", "word_sorting"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1PWS-I-5", 1, 1, "Reading and Literacy (L1)", "Phonics and Word Study", "Sound out words accurately.", ["phonics", "oral_reading", "multiple_choice"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1VWK-I-3", 1, 1, "Reading and Literacy (L1)", "Vocabulary and Word Knowledge", "Read high-frequency words accurately for meaning.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1CAT-I-1", 1, 1, "Reading and Literacy (L1)", "Comprehending and Analyzing Text", "Comprehend age-appropriate stories.", ["reading_comprehension", "multiple_choice", "sequencing", "true_false"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1CAT-I-2", 1, 1, "Reading and Literacy (L1)", "Comprehending and Analyzing Text", "Comprehend age-appropriate informational text.", ["reading_comprehension", "multiple_choice", "true_false"], source_title=G1_SOURCE, language_note=G1_NOTE),

    c("RL1PWS-II-1", 1, 2, "Reading and Literacy (L1)", "Phonics and Word Study", "Produce the sounds represented by letters in L1.", ["phonics", "multiple_choice"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1PWS-II-3", 1, 2, "Reading and Literacy (L1)", "Phonics and Word Study", "Isolate beginning and/or ending consonant and vowel sounds in words.", ["phonics", "multiple_choice", "word_sorting"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1PWS-II-4", 1, 2, "Reading and Literacy (L1)", "Phonics and Word Study", "Substitute individual sounds in simple words to form new words.", ["phonics", "fill_in_blank", "word_matching"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1PWS-II-5", 1, 2, "Reading and Literacy (L1)", "Phonics and Word Study", "Sound out words accurately.", ["phonics", "oral_reading", "multiple_choice"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1VWK-II-3", 1, 2, "Reading and Literacy (L1)", "Vocabulary and Word Knowledge", "Read high-frequency words accurately for meaning.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1BPK-II-1", 1, 2, "Reading and Literacy (L1)", "Book and Print Knowledge", "Recognize environmental print and common symbols.", ["multiple_choice", "word_matching", "true_false"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1BPK-II-2", 1, 2, "Reading and Literacy (L1)", "Book and Print Knowledge", "Recognize basic parts of a book.", ["multiple_choice", "word_matching", "true_false"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1BPK-II-3", 1, 2, "Reading and Literacy (L1)", "Book and Print Knowledge", "Recognize proper eye movement and print direction during reading.", ["multiple_choice", "true_false", "sequencing"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1CAT-II-1", 1, 2, "Reading and Literacy (L1)", "Comprehending and Analyzing Text", "Comprehend age-appropriate stories.", ["reading_comprehension", "multiple_choice", "sequencing", "true_false"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1CAT-II-2", 1, 2, "Reading and Literacy (L1)", "Comprehending and Analyzing Text", "Comprehend age-appropriate informational text.", ["reading_comprehension", "multiple_choice", "true_false"], source_title=G1_SOURCE, language_note=G1_NOTE),

    c("RL1PWS-III-3", 1, 3, "Reading and Literacy (L1)", "Phonics and Word Study", "Isolate beginning and/or ending consonant and vowel sounds in words.", ["phonics", "multiple_choice", "word_sorting"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1PWS-III-4", 1, 3, "Reading and Literacy (L1)", "Phonics and Word Study", "Substitute individual sounds in simple words to form new words.", ["phonics", "fill_in_blank", "word_matching"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1PWS-III-5", 1, 3, "Reading and Literacy (L1)", "Phonics and Word Study", "Sound out words accurately.", ["phonics", "oral_reading", "multiple_choice"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1VWK-III-3", 1, 3, "Reading and Literacy (L1)", "Vocabulary and Word Knowledge", "Read high-frequency words accurately for meaning.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1CAT-III-1", 1, 3, "Reading and Literacy (L1)", "Comprehending and Analyzing Text", "Read sentences with appropriate speed, accuracy, and expression.", ["oral_reading", "reading_comprehension"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1CAT-III-2", 1, 3, "Reading and Literacy (L1)", "Comprehending and Analyzing Text", "Comprehend age-appropriate stories.", ["reading_comprehension", "multiple_choice", "sequencing", "true_false"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1CAT-III-3", 1, 3, "Reading and Literacy (L1)", "Comprehending and Analyzing Text", "Comprehend age-appropriate informational text.", ["reading_comprehension", "multiple_choice", "true_false"], source_title=G1_SOURCE, language_note=G1_NOTE),

    c("RL1PWS-IV-3", 1, 4, "Reading and Literacy (L1)", "Phonics and Word Study", "Isolate beginning and/or ending consonant and vowel sounds in words.", ["phonics", "multiple_choice", "word_sorting"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1PWS-IV-4", 1, 4, "Reading and Literacy (L1)", "Phonics and Word Study", "Substitute individual sounds in simple words to form new words.", ["phonics", "fill_in_blank", "word_matching"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1PWS-IV-5", 1, 4, "Reading and Literacy (L1)", "Phonics and Word Study", "Sound out words accurately.", ["phonics", "oral_reading", "multiple_choice"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1VWK-IV-3", 1, 4, "Reading and Literacy (L1)", "Vocabulary and Word Knowledge", "Read high-frequency words accurately for meaning.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1CAT-IV-1", 1, 4, "Reading and Literacy (L1)", "Comprehending and Analyzing Text", "Read sentences with appropriate speed, accuracy, and expression.", ["oral_reading", "reading_comprehension"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1CAT-IV-2", 1, 4, "Reading and Literacy (L1)", "Comprehending and Analyzing Text", "Comprehend age-appropriate stories.", ["reading_comprehension", "multiple_choice", "sequencing", "true_false"], source_title=G1_SOURCE, language_note=G1_NOTE),
    c("RL1CAT-IV-3", 1, 4, "Reading and Literacy (L1)", "Comprehending and Analyzing Text", "Comprehend age-appropriate informational text.", ["reading_comprehension", "multiple_choice", "true_false"], source_title=G1_SOURCE, language_note=G1_NOTE),

    # ------------------------------------------------------------------
    # GRADE 2 — ENGLISH
    # ------------------------------------------------------------------
    c("EN2PA-I-1", 2, 1, "English", "Phonological Awareness", "Recognize rhymes in chants, poems, and stories heard.", ["rhyming", "multiple_choice", "true_false"], source_title=G23_SOURCE),
    c("EN2PA-I-2", 2, 1, "English", "Phonological Awareness", "Segment onset and rime.", ["phonics", "multiple_choice", "word_sorting"], source_title=G23_SOURCE),
    c("EN2PWS-I-1", 2, 1, "English", "Phonics and Word Study", "Identify Grade 2-appropriate sight words.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN2PWS-I-3", 2, 1, "English", "Phonics and Word Study", "Read CVC-pattern words accurately and automatically.", ["phonics", "oral_reading", "multiple_choice"], source_title=G23_SOURCE),
    c("EN2VWK-I-1", 2, 1, "English", "Vocabulary and Word Knowledge", "Identify high-frequency words accurately.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN2VWK-I-5", 2, 1, "English", "Vocabulary and Word Knowledge", "Read patterned words correctly for meaning.", ["phonics", "vocabulary", "multiple_choice"], source_title=G23_SOURCE),
    c("EN2CAT-I-1", 2, 1, "English", "Comprehending and Analyzing Text", "Read grade-level sentences with appropriate speed, accuracy, and expression.", ["oral_reading", "reading_comprehension"], source_title=G23_SOURCE),
    c("EN2CAT-I-2", 2, 1, "English", "Comprehending and Analyzing Text", "Comprehend stories, including key elements and event relationships.", ["reading_comprehension", "multiple_choice", "sequencing", "true_false"], source_title=G23_SOURCE),
    c("EN2CAT-I-3", 2, 1, "English", "Comprehending and Analyzing Text", "Comprehend informational texts and identify significant details.", ["reading_comprehension", "multiple_choice", "true_false"], source_title=G23_SOURCE),

    c("EN2PA-II-1", 2, 2, "English", "Phonological Awareness", "Recognize rhymes in chants, poems, and stories heard.", ["rhyming", "multiple_choice", "true_false"], source_title=G23_SOURCE),
    c("EN2PA-II-2", 2, 2, "English", "Phonological Awareness", "Segment onset and rime.", ["phonics", "multiple_choice", "word_sorting"], source_title=G23_SOURCE),
    c("EN2PWS-II-1", 2, 2, "English", "Phonics and Word Study", "Identify Grade 2-appropriate sight words.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN2PWS-II-2", 2, 2, "English", "Phonics and Word Study", "Read words accurately and automatically using grade-appropriate word patterns.", ["phonics", "oral_reading", "multiple_choice"], source_title=G23_SOURCE),
    c("EN2VWK-II-1", 2, 2, "English", "Vocabulary and Word Knowledge", "Identify high-frequency words accurately.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN2VWK-II-5", 2, 2, "English", "Vocabulary and Word Knowledge", "Read patterned words correctly for meaning.", ["phonics", "vocabulary", "multiple_choice"], source_title=G23_SOURCE),
    c("EN2CAT-II-1", 2, 2, "English", "Comprehending and Analyzing Text", "Read grade-level sentences with appropriate speed, accuracy, and expression.", ["oral_reading", "reading_comprehension"], source_title=G23_SOURCE),
    c("EN2CAT-II-2", 2, 2, "English", "Comprehending and Analyzing Text", "Comprehend stories, including key elements and event relationships.", ["reading_comprehension", "multiple_choice", "sequencing", "true_false"], source_title=G23_SOURCE),
    c("EN2CAT-II-3", 2, 2, "English", "Comprehending and Analyzing Text", "Comprehend informational texts and identify significant details.", ["reading_comprehension", "multiple_choice", "true_false"], source_title=G23_SOURCE),

    c("EN2PA-III-1", 2, 3, "English", "Phonological Awareness", "Recognize rhymes in chants, poems, and stories heard.", ["rhyming", "multiple_choice", "true_false"], source_title=G23_SOURCE),
    c("EN2PA-III-2", 2, 3, "English", "Phonological Awareness", "Segment onset and rime.", ["phonics", "multiple_choice", "word_sorting"], source_title=G23_SOURCE),
    c("EN2PWS-III-1", 2, 3, "English", "Phonics and Word Study", "Identify Grade 2-appropriate sight words.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN2PWS-III-2", 2, 3, "English", "Phonics and Word Study", "Read words accurately and automatically using grade-appropriate word patterns.", ["phonics", "oral_reading", "multiple_choice"], source_title=G23_SOURCE),
    c("EN2VWK-III-1", 2, 3, "English", "Vocabulary and Word Knowledge", "Identify high-frequency words accurately.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN2VWK-III-5", 2, 3, "English", "Vocabulary and Word Knowledge", "Read patterned words correctly for meaning.", ["phonics", "vocabulary", "multiple_choice"], source_title=G23_SOURCE),
    c("EN2CAT-III-1", 2, 3, "English", "Comprehending and Analyzing Text", "Read grade-level sentences with appropriate speed, accuracy, and expression.", ["oral_reading", "reading_comprehension"], source_title=G23_SOURCE),
    c("EN2CAT-III-2", 2, 3, "English", "Comprehending and Analyzing Text", "Comprehend stories, including key elements and event relationships.", ["reading_comprehension", "multiple_choice", "sequencing", "true_false"], source_title=G23_SOURCE),
    c("EN2CAT-III-3", 2, 3, "English", "Comprehending and Analyzing Text", "Comprehend informational texts and identify significant details.", ["reading_comprehension", "multiple_choice", "true_false"], source_title=G23_SOURCE),

    c("EN2PA-IV-1", 2, 4, "English", "Phonological Awareness", "Recognize rhymes in chants, poems, and stories heard.", ["rhyming", "multiple_choice", "true_false"], source_title=G23_SOURCE),
    c("EN2PA-IV-2", 2, 4, "English", "Phonological Awareness", "Segment onset and rime.", ["phonics", "multiple_choice", "word_sorting"], source_title=G23_SOURCE),
    c("EN2PWS-IV-1", 2, 4, "English", "Phonics and Word Study", "Identify Grade 2-appropriate sight words.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN2PWS-IV-2", 2, 4, "English", "Phonics and Word Study", "Read words accurately and automatically using grade-appropriate word patterns.", ["phonics", "oral_reading", "multiple_choice"], source_title=G23_SOURCE),
    c("EN2VWK-IV-1", 2, 4, "English", "Vocabulary and Word Knowledge", "Identify high-frequency words accurately.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN2VWK-IV-5", 2, 4, "English", "Vocabulary and Word Knowledge", "Identify synonyms and antonyms.", ["vocabulary", "word_matching", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN2VWK-IV-6", 2, 4, "English", "Vocabulary and Word Knowledge", "Read patterned words correctly for meaning.", ["phonics", "vocabulary", "multiple_choice"], source_title=G23_SOURCE),
    c("EN2CAT-IV-1", 2, 4, "English", "Comprehending and Analyzing Text", "Read grade-level sentences with appropriate speed, accuracy, and expression.", ["oral_reading", "reading_comprehension"], source_title=G23_SOURCE),
    c("EN2CAT-IV-2", 2, 4, "English", "Comprehending and Analyzing Text", "Comprehend stories, including key elements and event relationships.", ["reading_comprehension", "multiple_choice", "sequencing", "true_false"], source_title=G23_SOURCE),
    c("EN2CAT-IV-3", 2, 4, "English", "Comprehending and Analyzing Text", "Comprehend informational texts and identify significant details.", ["reading_comprehension", "multiple_choice", "true_false"], source_title=G23_SOURCE),

    # ------------------------------------------------------------------
    # GRADE 3 — ENGLISH
    # ------------------------------------------------------------------
    c("EN3PWS-I-1", 3, 1, "English", "Phonics and Word Study", "Identify Grade 3-appropriate sight words.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN3PWS-I-2", 3, 1, "English", "Phonics and Word Study", "Read words accurately and automatically using Grade 3 word patterns.", ["phonics", "oral_reading", "multiple_choice"], source_title=G23_SOURCE),
    c("EN3VWK-I-1", 3, 1, "English", "Vocabulary and Word Knowledge", "Identify high-frequency words accurately.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN3VWK-I-5", 3, 1, "English", "Vocabulary and Word Knowledge", "Identify synonyms and antonyms.", ["vocabulary", "word_matching", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN3VWK-I-6", 3, 1, "English", "Vocabulary and Word Knowledge", "Read patterned words correctly for meaning.", ["vocabulary", "phonics", "multiple_choice"], source_title=G23_SOURCE),
    c("EN3CAT-I-1", 3, 1, "English", "Comprehending and Analyzing Text", "Read grade-level sentences with appropriate speed, accuracy, and expression.", ["oral_reading", "reading_comprehension"], source_title=G23_SOURCE),
    c("EN3CAT-I-2", 3, 1, "English", "Comprehending and Analyzing Text", "Comprehend stories, including key elements, five-event sequencing, inference, cause/effect, prediction, and summary.", ["reading_comprehension", "multiple_choice", "sequencing", "true_false"], source_title=G23_SOURCE),
    c("EN3CAT-I-3", 3, 1, "English", "Comprehending and Analyzing Text", "Comprehend informational texts, identify significant details and text type, and draw conclusions.", ["reading_comprehension", "multiple_choice", "true_false"], source_title=G23_SOURCE),

    c("EN3PWS-II-1", 3, 2, "English", "Phonics and Word Study", "Identify Grade 3-appropriate sight words.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN3PWS-II-2", 3, 2, "English", "Phonics and Word Study", "Read words accurately and automatically using Grade 3 word patterns.", ["phonics", "oral_reading", "multiple_choice"], source_title=G23_SOURCE),
    c("EN3VWK-II-1", 3, 2, "English", "Vocabulary and Word Knowledge", "Identify high-frequency words accurately.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN3VWK-II-5", 3, 2, "English", "Vocabulary and Word Knowledge", "Identify synonyms and antonyms.", ["vocabulary", "word_matching", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN3VWK-II-6", 3, 2, "English", "Vocabulary and Word Knowledge", "Read patterned words correctly for meaning.", ["vocabulary", "phonics", "multiple_choice"], source_title=G23_SOURCE),
    c("EN3CAT-II-1", 3, 2, "English", "Comprehending and Analyzing Text", "Read grade-level sentences with appropriate speed, accuracy, and expression.", ["oral_reading", "reading_comprehension"], source_title=G23_SOURCE),
    c("EN3CAT-II-2", 3, 2, "English", "Comprehending and Analyzing Text", "Comprehend stories and analyze important events and relationships.", ["reading_comprehension", "multiple_choice", "sequencing", "true_false"], source_title=G23_SOURCE),
    c("EN3CAT-II-3", 3, 2, "English", "Comprehending and Analyzing Text", "Comprehend informational texts and draw conclusions from significant details.", ["reading_comprehension", "multiple_choice", "true_false"], source_title=G23_SOURCE),

    c("EN3PWS-III-1", 3, 3, "English", "Phonics and Word Study", "Identify Grade 3-appropriate sight words.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN3VWK-III-1", 3, 3, "English", "Vocabulary and Word Knowledge", "Identify high-frequency words accurately.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN3VWK-III-5", 3, 3, "English", "Vocabulary and Word Knowledge", "Identify synonyms and antonyms.", ["vocabulary", "word_matching", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN3VWK-III-6", 3, 3, "English", "Vocabulary and Word Knowledge", "Read patterned words correctly for meaning.", ["vocabulary", "phonics", "multiple_choice"], source_title=G23_SOURCE),
    c("EN3CAT-III-1", 3, 3, "English", "Comprehending and Analyzing Text", "Read grade-level sentences with appropriate speed, accuracy, and expression.", ["oral_reading", "reading_comprehension"], source_title=G23_SOURCE),
    c("EN3CAT-III-2", 3, 3, "English", "Comprehending and Analyzing Text", "Comprehend stories and analyze important events and relationships.", ["reading_comprehension", "multiple_choice", "sequencing", "true_false"], source_title=G23_SOURCE),
    c("EN3CAT-III-3", 3, 3, "English", "Comprehending and Analyzing Text", "Comprehend informational texts and draw conclusions from significant details.", ["reading_comprehension", "multiple_choice", "true_false"], source_title=G23_SOURCE),

    c("EN3PWS-IV-1", 3, 4, "English", "Phonics and Word Study", "Identify Grade 3-appropriate sight words.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN3VWK-IV-1", 3, 4, "English", "Vocabulary and Word Knowledge", "Identify high-frequency words accurately.", ["sight_words", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN3VWK-IV-5", 3, 4, "English", "Vocabulary and Word Knowledge", "Identify synonyms and antonyms.", ["vocabulary", "word_matching", "multiple_choice", "fill_in_blank"], source_title=G23_SOURCE),
    c("EN3VWK-IV-6", 3, 4, "English", "Vocabulary and Word Knowledge", "Read patterned words correctly for meaning.", ["vocabulary", "phonics", "multiple_choice"], source_title=G23_SOURCE),
    c("EN3CAT-IV-1", 3, 4, "English", "Comprehending and Analyzing Text", "Read grade-level sentences with appropriate speed, accuracy, and expression.", ["oral_reading", "reading_comprehension"], source_title=G23_SOURCE),
    c("EN3CAT-IV-2", 3, 4, "English", "Comprehending and Analyzing Text", "Comprehend stories, including sequencing at least five events and summarizing.", ["reading_comprehension", "multiple_choice", "sequencing", "true_false"], source_title=G23_SOURCE),
    c("EN3CAT-IV-3", 3, 4, "English", "Comprehending and Analyzing Text", "Comprehend informational texts, identify significant details and text type, and draw conclusions.", ["reading_comprehension", "multiple_choice", "true_false"], source_title=G23_SOURCE),
)


COMPETENCY_BY_CODE = {item.code.upper(): item for item in MATATAG_COMPETENCIES}


def get_competency(code: str) -> MatatagCompetency | None:
    return COMPETENCY_BY_CODE.get(code.strip().upper())


def list_competencies(
    *,
    grade: int | None = None,
    quarter: int | None = None,
    subdomain: str | None = None,
) -> list[MatatagCompetency]:
    values = list(MATATAG_COMPETENCIES)
    if grade is not None:
        values = [item for item in values if item.grade == grade]
    if quarter is not None:
        values = [item for item in values if item.quarter == quarter]
    if subdomain:
        needle = subdomain.strip().casefold()
        values = [item for item in values if needle in item.subdomain.casefold()]
    return values
