# MATATAG Grade 1-3 Reading Activity Generator API

A deployable FastAPI microservice that generates **MATATAG-aligned reading
activities** with the Gemini API.

The important difference in v2 is that the teacher does **not** type an
arbitrary curriculum competency. The backend owns a locked MATATAG competency
registry. The teacher selects a competency code from the API, and Gemini is only
used to generate activity content for that selected competency.

## MATATAG alignment model

```text
Teacher
  |
  v
Grade -> Quarter -> MATATAG competency code
  |
  v
Allowed activity types for that competency
  |
  v
POST /generate-activity
  |
  v
Gemini generates activity content only
  |
  v
FastAPI attaches trusted MATATAG metadata
  |
  v
Teacher preview/edit -> Save -> Publish
```

### Important Grade 1 note

MATATAG does not offer English as the formal language learning area in Grade 1.
Grade 1 uses **Language** and **Reading and Literacy** in the learner's first
language (L1). English and Filipino begin as learning areas in Grade 2.

Because this application is specifically for English reading, Grade 1 output is
therefore labeled by the API as an **English adaptation of the selected Grade 1
MATATAG Reading and Literacy foundational competency**. The service never calls
it an official "MATATAG English Grade 1" competency.

Grades 2 and 3 use MATATAG English competency codes.

## Curriculum scope in this starter

`curriculum.py` contains a curated **reading-focused subset** of 100 competency
records for Grades 1-3 across Quarters 1-4. It focuses on:

- phonological awareness
- phonics and word study
- sight/high-frequency words
- vocabulary and word knowledge
- book and print knowledge (Grade 1)
- oral reading / fluency
- narrative comprehension
- informational-text comprehension

This is intentionally narrower than the complete MATATAG English curriculum
because the product scope is reading support. You can expand the registry later
with grammar, writing, oral-language, and composing competencies if those become
part of the application.

The canonical curriculum guides used to structure the registry are:

- `MATATAG Curriculum: Reading and Literacy (Grade 1), August 2023`
- `MATATAG Curriculum: English (Grades 2-10), August 2023`

Always retain the official DepEd curriculum guides as the authoritative source.
The concise competency labels in `curriculum.py` are application-facing summaries
used for selection and prompting.

## Security

If an API key has ever been pasted into a chat, issue tracker, repository, or
other message, revoke/rotate it and use a fresh key. Never place a real Gemini
key in this repository.

```text
Web / Mobile app
      |
      v
Main application backend (Laravel/PHP/Node/etc.)
      |
      | X-App-Key (server-to-server only)
      v
This FastAPI service on Render
      |
      | GEMINI_API_KEY (Render secret)
      v
Gemini API
```

The mobile/web application should never contain `GEMINI_API_KEY` or
`APP_API_KEY`.

## Files

```text
main.py          FastAPI service and Gemini generation
curriculum.py    Locked MATATAG reading competency registry
render.yaml      Render Blueprint
requirements.txt Python dependencies
.env.example     Environment variable template
```

## Deploy to Render

Create a Web Service from the GitHub repository or use `render.yaml` as a
Blueprint.

Manual settings:

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

Set these environment variables:

```env
GEMINI_API_KEY=YOUR_NEW_GEMINI_KEY
GEMINI_MODEL=gemini-2.5-flash-lite
APP_API_KEY=YOUR_RANDOM_SERVER_TO_SERVER_SECRET
ALLOWED_ORIGINS=https://your-web-app.example.com
```

Generate an internal key with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## API flow

### 1. Load Grade/Quarter competencies

Your teacher UI first calls:

```http
GET /matatag/competencies?grade=2&quarter=1
```

Example record:

```json
{
  "code": "EN2PA-I-1",
  "grade": 2,
  "quarter": 1,
  "learning_area": "English",
  "subdomain": "Phonological Awareness",
  "competency": "Recognize rhymes in chants, poems, and stories heard.",
  "allowed_activity_types": [
    "rhyming",
    "multiple_choice",
    "true_false"
  ],
  "source_title": "MATATAG Curriculum: English (Grades 2-10), August 2023",
  "source_year": 2023,
  "language_note": null
}
```

This endpoint does not call Gemini and does not require the Gemini key.

Useful filters:

```http
GET /matatag/competencies?grade=1&quarter=2
GET /matatag/competencies?grade=3&quarter=4
GET /matatag/competencies?grade=2&quarter=1&subdomain=Phonological
GET /matatag/competencies/EN2PA-I-1
GET /matatag/metadata
```

### 2. Generate using the selected MATATAG code

```http
POST /generate-activity
Content-Type: application/json
X-App-Key: YOUR_INTERNAL_SERVER_KEY
```

```json
{
  "grade": 2,
  "quarter": 1,
  "competency_code": "EN2PA-I-1",
  "activity_type": "rhyming",
  "difficulty": "easy",
  "number_of_items": 5,
  "topic": "Animals"
}
```

The API verifies all of these before Gemini is called:

1. the competency code exists in the MATATAG registry;
2. the code belongs to the requested grade;
3. the code belongs to the requested quarter;
4. the chosen activity type is mapped to that competency.

For example, asking for `sentence_building` with a rhyme-recognition competency
will return HTTP 422 instead of letting the LLM improvise.

### 3. Response

Curriculum metadata is attached by FastAPI from the registry; Gemini does not
create it.

```json
{
  "alignment": {
    "curriculum": "MATATAG K to 10 Curriculum",
    "alignment_status": "matatag-aligned",
    "source_title": "MATATAG Curriculum: English (Grades 2-10), August 2023",
    "source_year": 2023,
    "grade": 2,
    "quarter": 1,
    "learning_area": "English",
    "subdomain": "Phonological Awareness",
    "competency_code": "EN2PA-I-1",
    "competency": "Recognize rhymes in chants, poems, and stories heard.",
    "language_note": null
  },
  "activity_type": "rhyming",
  "difficulty": "easy",
  "title": "Animal Rhyme Time",
  "instructions": "Choose the word that rhymes.",
  "passage": null,
  "items": [
    {
      "number": 1,
      "prompt": "Which word rhymes with cat?",
      "choices": ["hat", "dog", "sun"],
      "answer": "hat",
      "explanation": "Cat and hat share the same ending sound."
    }
  ]
}
```

## Grade 1 example

First query:

```http
GET /matatag/competencies?grade=1&quarter=1
```

Then generate:

```json
{
  "grade": 1,
  "quarter": 1,
  "competency_code": "RL1PA-I-3",
  "activity_type": "rhyming",
  "difficulty": "easy",
  "number_of_items": 5
}
```

The response will explicitly carry a `language_note` explaining that this is an
English adaptation of a Grade 1 L1 Reading and Literacy competency.

## PHP/Laravel-style flow

Your Laravel backend should first populate the teacher's competency dropdown
from the FastAPI curriculum endpoint, or copy/cache those records in your own
MySQL database.

Generation call:

```php
<?php

$payload = [
    'grade' => 2,
    'quarter' => 1,
    'competency_code' => 'EN2PA-I-1',
    'activity_type' => 'rhyming',
    'difficulty' => 'easy',
    'number_of_items' => 5,
    'topic' => 'Animals',
];

$ch = curl_init(getenv('ACTIVITY_AI_URL') . '/generate-activity');

curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => [
        'Content-Type: application/json',
        'X-App-Key: ' . getenv('ACTIVITY_AI_KEY'),
    ],
    CURLOPT_POSTFIELDS => json_encode($payload),
    CURLOPT_TIMEOUT => 60,
]);

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$error = curl_error($ch);
curl_close($ch);

if ($response === false || $httpCode >= 400) {
    throw new RuntimeException("Activity API failed: HTTP $httpCode $error $response");
}

$activity = json_decode($response, true, 512, JSON_THROW_ON_ERROR);
```

Your PHP backend keeps:

```env
ACTIVITY_AI_URL=https://YOUR-SERVICE.onrender.com
ACTIVITY_AI_KEY=the_same_value_as_render_APP_API_KEY
```

## Recommended teacher UI

```text
Grade
  |
Quarter
  |
MATATAG Subdomain
  |
MATATAG Competency
  |
Allowed Activity Type
  |
Difficulty / Item Count / Optional Topic
  |
Generate
  |
Teacher Preview + Edit
  |
Save / Publish
```

Do **not** provide a free-text "curriculum competency" box as the primary
workflow. That would let users accidentally create activities that are not
actually MATATAG-aligned.

## Privacy

Activity generation should contain curriculum information only. Do not send
student names, IDs, emails, detailed learner profiles, or other personal data to
Gemini just to create classroom activities.
