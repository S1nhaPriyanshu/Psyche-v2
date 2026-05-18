# =============================================================================
# Psyche v2 — Core Bot
# A privacy-first, high-reasoning Discord behavioral analysis bot.
# Powered by Google Gemini.
# =============================================================================

import asyncio
import aiosqlite
import logging
import os
import json
import time
from datetime import datetime

import aiohttp
import discord
from discord.ext import commands, tasks
from discord import ui
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

# =============================================================================
# 1. CONFIGURATION & ENVIRONMENT
# =============================================================================

load_dotenv()

_token = os.getenv('DISCORD_TOKEN')
DISCORD_TOKEN   = _token.strip() if _token else None
GEMINI_API_KEY  = os.getenv('GEMINI_API_KEY')
SENTINO_API_KEY = os.getenv('SENTINO_API_KEY')  # Optional: personality scoring

# Command channel: restrict commands to this channel (set in .env)
try:
    COMMAND_CHANNEL_ID = int(os.getenv('COMMAND_CHANNEL_ID', '0'))
except ValueError:
    COMMAND_CHANNEL_ID = 0
# ---------------------------------------------------------
SCAN_MODEL = os.getenv('SCAN_MODEL', 'llama-3.3-70b-versatile')
DOSSIER_MODEL = os.getenv('DOSSIER_MODEL', 'llama-3.3-70b-versatile')
# ---------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'psyche.db')

# Robust pathing for local assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_JSON = os.path.join(BASE_DIR, 'questions.json')

# Startup Validation
if not DISCORD_TOKEN: raise EnvironmentError("DISCORD_TOKEN missing.")
if not GEMINI_API_KEY:
    log.warning("GEMINI_API_KEY not set - Gemini fallback disabled (Groq will be used as primary).")

# =============================================================================
# 2. LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('psyche')
logging.getLogger('discord').setLevel(logging.WARNING)

# =============================================================================
# COST TRACKING (Token Savings Calculator)
# =============================================================================
# Estimates token savings from using structured JSON instead of raw transcript

COST_TRACKING = {
    "behavior_scan": {"full_transcript_tokens": 0, "structured_tokens": 0, "savings": 0},
    "generate_dossier": {"full_transcript_tokens": 0, "structured_tokens": 0, "savings": 0},
}

def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token."""
    return len(text) // 4

def log_cost_savings(command: str, full_tokens: int, structured_tokens: int):
    """Log token savings for a command and update totals."""
    savings = full_tokens - structured_tokens
    if command in COST_TRACKING:
        COST_TRACKING[command]["full_transcript_tokens"] += full_tokens
        COST_TRACKING[command]["structured_tokens"] += structured_tokens
        COST_TRACKING[command]["savings"] += savings
    log.info(f"[COST] {command}: {full_tokens} -> {structured_tokens} tokens (saved ~{savings})")

def get_cost_summary() -> str:
    """Get formatted cost summary for logging."""
    total_saved = sum(c["savings"] for c in COST_TRACKING.values())
    lines = ["=== COST TRACKING SUMMARY ==="]
    for cmd, data in COST_TRACKING.items():
        pct = (data["savings"] / data["full_transcript_tokens"] * 100) if data["full_transcript_tokens"] > 0 else 0
        lines.append(f"{cmd}: {data['savings']} tokens saved ({pct:.1f}% reduction)")
    lines.append(f"TOTAL SAVED: {total_saved} tokens")
    return "\n".join(lines)

# =============================================================================
# 3. GOOGLE GEMINI CONFIGURATION
# =============================================================================

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# System instruction forces the "Clinical" persona globally
SYSTEM_INSTRUCTION = (
    "You are Psyche, an advanced Forensic Psychology AI. Your task is to analyze "
    "Discord interactions and raw psychometric data. Look for cognitive dissonance, "
    "social archetypes, and linguistic patterns. NEVER provide a medical or psychiatric "
    "diagnosis. Frame all insights as 'behavioral conjecture' based on text patterns."
)

# =============================================================================
# MULTI-PROVIDER AI WATERFALL
# Groq → OpenRouter → OpenCode → Gemini
# Each provider is skipped if its API key is not set in .env
# =============================================================================

# Gemini model name fallback chain (tried on 404)
_GEMINI_FALLBACKS = [
    "{model}",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
]

# OpenAI-compatible provider waterfall (tried in order)
# Note: Only providers with valid API keys will work
_PROVIDER_WATERFALL = [
    {
        "name": "Groq",
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "models": [
            "llama-3.3-70b-versatile",     # Best quality, 128k - try this FIRST
            "llama-3.1-8b-instant",        # Ultra-fast fallback
            "mixtral-8x7b-32768",          # Wide context
        ],
    },
    {
        "name": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "models": [
            "inclusionai/ring-2.6-1t:free",          # 262K context + reasoning
            "google/gemma-4-31b-it:free",              # 262K Google's top free model
            "minimax/minimax-m2.5:free",              # 197K solid fallback
        ],
    },
    {
        "name": "OpenCode",
        "env_key": "OPENCODE_API_KEY",
        "base_url": "https://api.opencode.ai/v1/chat/completions",
        "models": [
            "ring-2.6-1t-free",              # 262K context
            "big-pickle",                     # 200K context
            "minimax-m2.5-free",             # 197K context
        ],
    },
]

async def _call_gemini(model: str, contents: str, config) -> str:
    """Try Gemini with model-name fallbacks on 404. Returns response text."""
    tried = set()
    for pattern in _GEMINI_FALLBACKS:
        candidate = pattern.format(model=model)
        if candidate in tried:
            continue
        tried.add(candidate)
        try:
            response = await client.models.generate_content(
                model=candidate,
                contents=contents,
                config=config,
            )
            if candidate != model:
                log.info("Gemini model fallback: '%s' → '%s'", model, candidate)
            return response.text
        except errors.APIError as e:
            if "404" in str(e) or "NOT_FOUND" in str(e):
                log.warning("Gemini model '%s' not found. Trying next...", candidate)
                continue
            raise  # Re-raise 429s to trigger provider waterfall
    raise errors.APIError(f"All Gemini model variants exhausted for '{model}'.")

async def _call_openai_compatible(provider: dict, contents: str, system_instruction: str) -> str:
    """
    Generic caller for any OpenAI-compatible REST endpoint.
    Tries each model in the provider's list, skips on 404/429.
    """
    api_key = os.getenv(provider["env_key"])
    if not api_key:
        raise RuntimeError(f"{provider['name']}: {provider['env_key']} not set. Skipping.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user",   "content": contents},
        ],
        "temperature": 0.7,
    }

    for model in provider["models"]:
        payload["model"] = model
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    provider["base_url"],
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    # Check for errors before raising
                    if resp.status == 413:
                        raise RuntimeError(f"413 Payload Too Large for model {model}")
                    if resp.status in (404, 429):
                        log.warning("%s: model '%s' returned %d. Trying next...",
                                    provider["name"], model, resp.status)
                        continue
                    if resp.status >= 400:
                        # Try to get error message from response
                        try:
                            err_data = await resp.json()
                            err_msg = err_data.get("error", {}).get("message", str(resp.status))
                        except Exception:
                            err_text = await resp.text()
                            err_msg = f"{resp.status}: {err_text[:100]}"
                        log.warning("%s: model '%s' returned error: %s", provider["name"], model, err_msg)
                        continue

                    # Try to parse JSON response
                    try:
                        data = await resp.json()
                    except Exception:
                        text = await resp.text()
                        log.warning("%s: model '%s' returned non-JSON: %s", provider["name"], model, text[:200])
                        continue

                    if "choices" in data and data["choices"]:
                        log.info("%s fallback succeeded with model '%s'.", provider["name"], model)
                        return data["choices"][0]["message"]["content"]
                    else:
                        log.warning("%s: model '%s' returned no choices", provider["name"], model)
                        continue

        except RuntimeError:
            raise
        except Exception as e:
            # Check if it's a 413 for this specific model
            if "413" in str(e) or "Payload Too Large" in str(e):
                raise RuntimeError(f"413 Payload Too Large: {e}")
            log.warning("%s model '%s' error: %s", provider["name"], model, e)
            continue

    raise RuntimeError(f"{provider['name']}: all models exhausted.")

# Truncation helper for large prompts
def _truncate_for_fallback(contents: str, max_len: int = 15000) -> str:
    """Truncates long prompts for smaller context windows."""
    if len(contents) <= max_len:
        return contents
    # Keep system instruction at end
    return contents[:max_len] + "\n\n[... content truncated for model limits ...]"

# =============================================================================
# SENTINO PERSONALITY API
# =============================================================================

async def get_sentino_scores(text: str) -> str | None:
    """
    Call Sentino API to get Big5 personality scores.
    Returns formatted string with scores, or None if API key not set / fails.
    """
    if not SENTINO_API_KEY:
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.sentino.org/api/score/text",
                headers={"Authorization": f"Token {SENTINO_TOKEN}"},
                json={"text": text, "inventories": ["big5"]},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    log.warning(f"Sentino API returned {resp.status}")
                    return None

                data = await resp.json()
                # Extract Big5 scores from response
                # Format: {"big5": {"openness": 0.78, "conscientiousness": 0.65, ...}}
                big5 = data.get("big5", {})
                if not big5:
                    return None

                scores = []
                for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
                    score = big5.get(trait)
                    if score is not None:
                        # Sentino returns 0-1, convert to 0-100 for readability
                        scores.append(f"{trait.capitalize()}: {int(score * 100)}")

                return " | ".join(scores)

    except Exception as e:
        log.warning(f"Sentino API call failed: {e}")
        return None

# =============================================================================
# GROQ BEHAVIORAL DATA EXTRACTION (Zero-RAM Solution)
# Uses llama-3.1-8b-instant for fast sentiment + topic extraction
# =============================================================================

async def extract_behavioral_data(messages: list[str], batch_size: int = 100) -> dict:
    """
    Batches messages and generates a behavioral analysis summary using Groq.
    Returns structured JSON with the combined analysis text.
    """
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        log.warning("GROQ_API_KEY not set - behavioral extraction disabled")
        return {"analysis": "GROQ_API_KEY not set", "error": "GROQ_API_KEY not set"}

    all_analyses = []

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        batch_text = "\n".join([f"{j+1}. {msg[:200]}" for j, msg in enumerate(batch[:50])])  # Limit to 50 per batch

        prompt = (
            "You are a behavioral data analyzer. Analyze this batch of messages and provide a structured analysis.\n\n"
            "Output ONLY valid JSON (no markdown, no explanation):\n"
            "{\n"
            '  "narrative_summary": "A 1-2 paragraph summary of the user\'s communication style, emotional tone, and behavior in this batch.",\n'
            '  "dominant_tone": "A short phrase describing the dominant tone",\n'
            '  "key_topics": ["array of 3-5 topic keywords"]\n'
            "}\n\n"
            f"Message Batch:\n{batch_text}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system", "content": "You are a JSON-only behavioral analyzer. Output only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        log.warning(f"Groq behavioral extraction returned {resp.status}")
                        continue

                    data = await resp.json()
                    if "choices" in data and data["choices"]:
                        try:
                            result = json.loads(data["choices"][0]["message"]["content"])
                            summary = result.get("narrative_summary", "No summary.")
                            tone = result.get("dominant_tone", "Unknown")
                            topics = result.get("key_topics", [])
                            all_analyses.append(f"--- Batch {i // batch_size + 1} (Tone: {tone}) ---\n{summary}\nTopics: {', '.join(topics)}")
                        except Exception as e:
                            log.warning(f"Failed to parse JSON from Groq: {e}")
                            all_analyses.append(f"--- Batch {i // batch_size + 1} ---\n{data['choices'][0]['message']['content']}")
        except Exception as e:
            log.warning(f"Batch {i // batch_size + 1} behavioral extraction failed: {e}")
            continue

    if not all_analyses:
        return {"analysis": "No valid analysis generated.", "error": "No valid batches"}

    return {
        "analysis": "\n\n".join(all_analyses),
        "error": None
    }



# =============================================================================
# STRUCTURED PROMPT CONTEXT BUILDER
# =============================================================================
async def build_structured_prompt_context(user_id: str, guild_id: str, max_prompt_tokens: int = 2000) -> dict:
    """
    Build a compact structured context dict from user data instead of raw transcripts.
    This dramatically reduces prompt size while preserving key analysis signals.

    Args:
        user_id: The Discord user ID to fetch data for
        guild_id: The Discord guild/server ID to filter by
        max_prompt_tokens: Cap for token count (default 2000)

    Returns:
        dict with: message_count, behavioral_data, quiz_results, recent_sample, token_count
    """
    global bot

    # 1. Fetch user messages from DB (filtered by guild)
    async with bot.db.execute(
        "SELECT content, timestamp FROM interaction_history WHERE user_id = ? AND guild_id = ? ORDER BY timestamp DESC LIMIT 2000",
        (user_id, guild_id)
    ) as cursor:
        rows = await cursor.fetchall()

    if not rows:
        return {"error": "No messages found", "message_count": 0}

    # Extract message strings for behavioral analysis
    messages = [row[0] for row in rows]
    message_count = len(rows)

    # 2. Run extract_behavioral_data to get sentiment/topics/emotions
    behavioral_data = await extract_behavioral_data(messages)

    # 3. Fetch quiz results
    async with bot.db.execute("SELECT quiz_type, raw_answers FROM quiz_results WHERE user_id = ?", (user_id,)) as cursor:
        quiz_rows = await cursor.fetchall()

    # Load questions for mapping
    quiz_results = []
    try:
        with open(QUESTIONS_JSON, 'r') as f:
            all_questions_data = json.load(f)
    except:
        all_questions_data = {}

    # Build compact quiz context
    for q_type, raw_json in quiz_rows:
        try:
            answers = json.loads(raw_json)
            quiz_data = all_questions_data.get(q_type, {})
            questions = quiz_data.get("questions", [])

            # Compact format: just question text and answer index
            compact_answers = []
            for idx, ans in enumerate(answers):
                if idx < len(questions):
                    q_text = questions[idx].get("q", "")[:80]  # Truncate long questions
                    compact_answers.append(f"{q_text}: {ans}")

            quiz_results.append({
                "type": q_type.upper(),
                "responses": compact_answers[:20],  # Limit to first 20 questions
                "total_questions": len(answers)
            })
        except Exception:
            continue

    # 4. Get recent sample (last 20 messages for conversational context)
    recent_sample = [row[0][:150] for row in rows[:20]]  # Truncate each to 150 chars

    # 5. Build structured context
    structured = {
        "message_count": message_count,
        "behavioral_data": {
            "analysis": behavioral_data.get("analysis", "")
        },
        "quiz_results": quiz_results,
        "recent_sample": recent_sample
    }

    # 6. Estimate token count (structured is MUCH smaller than raw transcript)
    structured_json = json.dumps(structured)
    token_count = estimate_tokens(structured_json)

    # 7. If over token limit, further compress
    if token_count > max_prompt_tokens:
        # Remove emotional arc (most verbose)
        structured["behavioral_data"]["emotional_arc"] = []
        # Trim recent sample
        structured["recent_sample"] = structured["recent_sample"][:10]
        # Recalculate
        structured_json = json.dumps(structured)
        token_count = estimate_tokens(structured_json)

    structured["token_count"] = token_count
    structured["original_transcript_estimate"] = estimate_tokens("\n".join(messages[:500]))

    log.info(f"Structured context: {token_count} tokens (vs ~{structured['original_transcript_estimate']} for raw transcript)")

    return structured

def format_structured_context(ctx_data: dict, command: str) -> str:
    """
    Format structured context into a readable prompt section.
    Different commands get different detail levels.
    """
    if "error" in ctx_data:
        return f"⚠️ Data unavailable: {ctx_data['error']}"

    lines = []
    lines.append("=== USER CONTEXT (STRUCTURED) ===")
    lines.append(f"Messages analyzed: {ctx_data.get('message_count', 0):,}")

    # Behavioral data
    bd = ctx_data.get("behavioral_data", {})
    lines.append(f"\n--- Behavioral Profile ---")
    lines.append(f"Overall sentiment: {bd.get('sentiment', 0):.2f} (-1=negative, 0=neutral, 1=positive)")
    lines.append(f"Emotional intensity: {bd.get('intensity', 0):.2f}")

    topics = bd.get("topics", [])
    if topics:
        lines.append(f"Key topics: {', '.join(topics[:6])}")

    # Emotional arc summary (for detailed analysis)
    emotional_arc = bd.get("emotional_arc", [])
    if emotional_arc and command in ["generate_dossier", "deep_scan"]:
        arc_summary = [f"{e.get('emotions', [])}" for e in emotional_arc[-3:]]
        lines.append(f"Recent emotional states: {' | '.join(arc_summary)}")

    # Quiz results
    quiz_results = ctx_data.get("quiz_results", [])
    if quiz_results:
        lines.append(f"\n--- Psychometric Assessments ---")
        for quiz in quiz_results:
            lines.append(f"{quiz['type']}: {quiz['total_questions']} questions")
            # Show first 3 responses as examples
            if quiz.get("responses"):
                lines.append(f"  Sample: {quiz['responses'][0][:80]}...")
    else:
        lines.append("\n--- Psychometric Assessments ---")
        lines.append("None completed yet.")

    # Recent sample (only for behavior_scan - gives conversational context)
    if command == "behavior_scan":
        recent = ctx_data.get("recent_sample", [])
        if recent:
            lines.append(f"\n--- Recent Messages (sample) ---")
            for msg in recent[:5]:
                lines.append(f"> {msg}")

    # Token savings info
    orig_tokens = ctx_data.get("original_transcript_estimate", 0)
    new_tokens = ctx_data.get("token_count", 0)
    if orig_tokens > 0:
        savings = ((orig_tokens - new_tokens) / orig_tokens) * 100
        lines.append(f"\n[Token reduction: {savings:.1f}% ({orig_tokens} → {new_tokens})]")

    return "\n".join(lines)

# =============================================================================
# EDGE CASE DETECTION FOR WILDCARD SUB-PROMPT MODE
# =============================================================================
def should_use_llm_for_edge_case(behavioral_data: dict) -> bool:
    """
    Lightweight edge case detection - returns True when LLM deep analysis is needed.
    Uses structured Groq extraction as default path, falls back to LLM for edge cases.

    Triggers LLM when:
    - Sentiment is ambiguous (between -0.2 and 0.2)
    - Intensity is very high (> 0.8)
    - Topics contain unusual/ambiguous terms
    - Emotional arc shows sudden mood swings
    """
    if not behavioral_data or behavioral_data.get("error"):
        return True  # Default to LLM if extraction failed

    # Check 1: Ambiguous sentiment (neutral zone)
    sentiment = behavioral_data.get("sentiment", 0)
    if sentiment is not None and -0.2 <= sentiment <= 0.2:
        log.debug(f"Edge case: ambiguous sentiment {sentiment}")
        return True

    # Check 2: Very high intensity
    # Calculate average intensity from emotional arc
    emotional_arc = behavioral_data.get("emotional_arc", [])
    if emotional_arc:
        intensities = [batch.get("intensity", 0) for batch in emotional_arc]
        avg_intensity = sum(intensities) / len(intensities) if intensities else 0
        if avg_intensity > 0.8:
            log.debug(f"Edge case: high intensity {avg_intensity}")
            return True

        # Check 3: Sudden mood swings in emotional arc
        if len(emotional_arc) >= 3:
            sentiment_swings = []
            for i in range(1, len(emotional_arc)):
                prev_sent = emotional_arc[i-1].get("sentiment", 0)
                curr_sent = emotional_arc[i].get("sentiment", 0)
                swing = abs(curr_sent - prev_sent)
                sentiment_swings.append(swing)

            # If any swing > 0.5, it's a sudden mood shift
            if any(s > 0.5 for s in sentiment_swings):
                log.debug(f"Edge case: mood swing detected {sentiment_swings}")
                return True

    # Check 4: Unusual/ambiguous topics
    topics = behavioral_data.get("topics", [])
    # Keywords that indicate ambiguous/unusual content
    ambiguous_keywords = {
        "confused", "weird", "strange", "unsure", "maybe", "perhaps",
        "actually", "honestly", "literally", "basically", "whatever",
        "idk", "idc", "don't know", "not sure", "uncertain"
    }
    for topic in topics:
        topic_lower = topic.lower()
        if any(kw in topic_lower for kw in ambiguous_keywords):
            log.debug(f"Edge case: ambiguous topic '{topic}'")
            return True

    # Check 5: Emotion tags indicate mixed/ambiguous state
    if emotional_arc:
        all_emotions = []
        for batch in emotional_arc:
            all_emotions.extend(batch.get("emotions", []))

        # If more than 3 distinct emotions detected, it's complex
        unique_emotions = set(all_emotions)
        if len(unique_emotions) >= 4:
            # Check for conflicting emotions
            positive = {"happy", "excited", "hopeful"}
            negative = {"sad", "angry", "frustrated", "anxious"}
            has_positive = bool(unique_emotions & positive)
            has_negative = bool(unique_emotions & negative)
            if has_positive and has_negative:
                log.debug(f"Edge case: conflicting emotions {unique_emotions}")
                return True

    # Default: use structured data (no LLM needed)
    return False


async def generate_with_fallback(model: str, contents: str, config):
    """
    2-tier AI generation waterfall:
      1. OpenAI-compatible: Groq -> OpenRouter -> OpenCode
      2. Gemini (final fallback)

    If 413 Payload Too Large, retries with truncated content.

    Returns the response text as a plain string.
    """
    system_instruction = (config.system_instruction
                          if hasattr(config, "system_instruction") else SYSTEM_INSTRUCTION)

    # Tier 1: OpenAI-compatible providers (Groq -> OpenRouter -> OpenCode)
    #4: OpenAI-compatible providers
    last_error = None
    content_tried_full = False

    for provider in _PROVIDER_WATERFALL:
        try:
            return await _call_openai_compatible(provider, contents, system_instruction)
        except RuntimeError as e:
            # Check if it's a 413 (Payload Too Large) and we haven't tried truncation yet
            if "413" in str(e) and not content_tried_full:
                log.warning(f"Provider {provider['name']}: payload too large. Trying with truncated content...")
                content_tried_full = True
                try:
                    truncated = _truncate_for_fallback(contents)
                    return await _call_openai_compatible(provider, truncated, system_instruction)
                except Exception:
                    # Truncated also failed, continue to next provider
                    pass
            log.warning("Provider skipped: %s", e)
            last_error = e
            continue
        except Exception as e:
            log.warning("%s failed entirely: %s", provider["name"], e)
            last_error = e
            continue

    # Tier 2: Gemini (final fallback)
    try:
        return await _call_gemini(model, contents, config)
    except errors.APIError as e:
        log.warning("Gemini failed: %s", e)
        last_error = e

    raise RuntimeError(f"All AI providers exhausted. Last error: {last_error}")


# =============================================================================
# 4. PHASE 2: DATABASE & PRIVACY ARCHITECTURE
# =============================================================================

async def init_db(db: aiosqlite.Connection):
    """
    Initializes the database schema using an existing async connection.
    """
    log.info("Initializing database schema...")
    
    # Enable WAL mode for better concurrency (AI reads while scraper writes)
    await db.execute("PRAGMA journal_mode=WAL;")
    
    # -- 1. Message History Table (with INDEX for performance) --
    await db.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   TEXT NOT NULL,
            guild_id  TEXT NOT NULL,
            content   TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_guild ON messages(user_id, guild_id)")

    # -- 2. (Deprecated Compound PK Table removed to prevent schema collision) --

    # -- 3. Quiz Sessions Table (PK: user_id) --
    await db.execute('''
        CREATE TABLE IF NOT EXISTS quiz_sessions (
            user_id   TEXT PRIMARY KEY,
            quiz_type TEXT,
            progress  INTEGER,
            answers   TEXT
        )
    ''')

    # -- 4. Quiz Results Table --
    await db.execute('''
        CREATE TABLE IF NOT EXISTS quiz_results (
            user_id     TEXT,
            quiz_type   TEXT,
            raw_answers TEXT,
            timestamp   TEXT
        )
    ''')

    # -- 5. Cooldowns Table --
    await db.execute('''
        CREATE TABLE IF NOT EXISTS cooldowns (
            user_id                  TEXT PRIMARY KEY,
            last_dossier_time        REAL,
            last_map_time            REAL,
            last_scan_time           REAL
        )
    ''')
    
    # Run safe migrations for cooldowns table in case it was created in an older version
    try:
        await db.execute("ALTER TABLE cooldowns ADD COLUMN last_map_time REAL")
    except aiosqlite.OperationalError:
        pass
    try:
        await db.execute("ALTER TABLE cooldowns ADD COLUMN last_scan_time REAL")
    except aiosqlite.OperationalError:
        pass

    # -- 6. Interaction History Table (with guild_id for multi-server support) --
    await db.execute('''
        CREATE TABLE IF NOT EXISTS interaction_history (
            message_id  TEXT PRIMARY KEY,
            user_id     TEXT,
            guild_id    TEXT,
            content     TEXT,
            reply_to_id TEXT,
            timestamp   TEXT
        )
    ''')
    await db.execute('CREATE INDEX IF NOT EXISTS idx_interaction_guild ON interaction_history(guild_id)')
    await db.execute('CREATE INDEX IF NOT EXISTS idx_interaction_user_guild ON interaction_history(user_id, guild_id)')

    # -- 6. Sync Checkpoints (Scraper Resume) --
    await db.execute('''
        CREATE TABLE IF NOT EXISTS sync_checkpoints (
            channel_id      TEXT PRIMARY KEY,
            last_message_id TEXT NOT NULL
        )
    ''')

    # -- 7. Scrape Checkpoints (User-specific incremental scraping) --
    await db.execute('''
        CREATE TABLE IF NOT EXISTS scrape_checkpoints (
            user_id         TEXT NOT NULL,
            channel_id      TEXT NOT NULL,
            last_message_id TEXT NOT NULL,
            last_timestamp  TEXT NOT NULL,
            PRIMARY KEY (user_id, channel_id)
        )
    ''')

    await db.commit()
    log.info("Database schema ready.")

async def get_scrape_checkpoint(db: aiosqlite.Connection, user_id: str, channel_id: str) -> str | None:
    """Get last message ID for user+channel combo. Returns None if no checkpoint."""
    async with db.execute(
        "SELECT last_message_id FROM scrape_checkpoints WHERE user_id = ? AND channel_id = ?",
        (user_id, channel_id)
    ) as cursor:
        row = await cursor.fetchone()
        return row[0] if row else None

async def set_scrape_checkpoint(db: aiosqlite.Connection, user_id: str, channel_id: str, message_id: str, timestamp: str):
    """Update checkpoint for user+channel combo."""
    await db.execute(
        "INSERT OR REPLACE INTO scrape_checkpoints (user_id, channel_id, last_message_id, last_timestamp) VALUES (?, ?, ?, ?)",
        (user_id, channel_id, message_id, timestamp)
    )

async def get_global_checkpoint(db: aiosqlite.Connection, channel_id: str) -> str | None:
    """Get last message ID for global scraper (all users)."""
    async with db.execute(
        "SELECT last_message_id FROM sync_checkpoints WHERE channel_id = ?",
        (channel_id,)
    ) as cursor:
        row = await cursor.fetchone()
        return row[0] if row else None

async def set_global_checkpoint(db: aiosqlite.Connection, channel_id: str, message_id: str):
    """Update global checkpoint for channel."""
    await db.execute(
        "INSERT OR REPLACE INTO sync_checkpoints (channel_id, last_message_id) VALUES (?, ?)",
        (channel_id, message_id)
    )

def is_opted_in(member: discord.Member) -> bool:
    """Checks for the role named 'psycheoptin' (case-insensitive)."""
    if not isinstance(member, discord.Member): return False
    return any(role.name.lower() == "psycheoptin" for role in member.roles)

def apply_disclaimer(embed):
    """Applies the mandatory clinical disclaimer to any Discord Embed."""
    disclaimer = (
        "🚨 **CLINICAL DISCLAIMER:** This report/assessment is generated by AI based on automated "
        "pattern recognition. It is NOT a medical diagnosis or clinical advice. "
        "If you are in distress, please contact a licensed mental health professional."
    )
    embed.add_field(name="⚠️ Ethical Notice", value=disclaimer, inline=False)
    embed.set_footer(text="Psyche v2 | Experimental Behavioral Modeling")
    return embed

def chunk_text(text: str, chunk_size: int = 3900) -> list:
    """
    Splits text into chunks of chunk_size characters.
    Tries to break at sentence boundaries for readability.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    while len(text) > chunk_size:
        # Try to find a sentence break before chunk_size
        chunk = text[:chunk_size]
        last_period = max(chunk.rfind('.'), chunk.rfind('\n'))
        last_bullet = max(chunk.rfind('•'), chunk.rfind('-'))

        # Break at the better boundary
        break_point = max(last_period, last_bullet, chunk_size // 2)

        if break_point < chunk_size // 4:
            break_point = chunk_size  # No good break point, cut hard

        chunks.append(text[:break_point + 1].strip())
        text = text[break_point + 1:].strip()

    if text:
        chunks.append(text)

    return chunks

async def send_chunked_response(ctx_or_channel, text: str, title: str, color=discord.Color.dark_purple()):
    """
    Sends a long response as multiple embeds to bypass Discord's 4K limit.
    """
    chunks = chunk_text(text, 3900)

    for i, chunk in enumerate(chunks):
        embed = discord.Embed(
            title=f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title,
            description=chunk,
            color=color
        )
        embed = apply_disclaimer(embed)
        await ctx_or_channel.send(embed=embed)

async def resilient_call(func, *args, retries=3, delay=2, **kwargs):
    """
    Retries a Discord API call on transient network failures.
    Designed for HuggingFace Spaces where outbound HTTPS intermittently drops.
    Uses exponential backoff: 2s, 4s, 8s.
    """
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except (aiohttp.ClientConnectorError, ConnectionResetError, OSError) as e:
            if attempt == retries - 1:
                log.error("Network call failed after %d attempts: %s", retries, e)
                raise
            wait = delay * (2 ** attempt)
            log.warning("Network blip (attempt %d/%d): %s. Retrying in %ds...", attempt + 1, retries, e, wait)
            await asyncio.sleep(wait)

async def purge_user_data(user_id: str):
    """
    The Scrub Protocol (User-Level).
    Wipes ALL of a user's data from every table.
    """
    log.info("🧹 User Scrub: Purging data for user %s", user_id)
    await bot.db.execute("DELETE FROM interaction_history WHERE user_id = ?", (user_id,))
    await bot.db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    await bot.db.execute("DELETE FROM quiz_results WHERE user_id = ?", (user_id,))
    await bot.db.execute("DELETE FROM quiz_sessions WHERE user_id = ?", (user_id,))
    await bot.db.execute("DELETE FROM cooldowns WHERE user_id = ?", (user_id,))
    await bot.db.commit()
    log.info("✅ User Scrub complete for %s", user_id)

# =============================================================================
# 5. PHASE 3: CLINICAL ASSESSMENT ENGINE (discord.ui.View)
# =============================================================================

class AssessmentView(discord.ui.View):
    def __init__(self, bot_instance, user_id, quiz_type, questions, progress=0, answers=None, instructions=""):
        super().__init__(timeout=600)  # 10-minute idle timeout
        self.bot = bot_instance
        self.user_id = int(user_id)
        self.quiz_type = quiz_type
        self.questions = questions  # List of strings
        self.progress = progress
        self.answers = answers or []
        self.instructions = instructions

    async def update_question(self, interaction: discord.Interaction):
        """Edits the existing message to the next question. Handles both Likert and A/B formats."""
        if self.progress >= len(self.questions):
            await self.finish_quiz(interaction)
            return

        q_data = self.questions[self.progress]
        q_text = q_data["q"]

        # Build the question prompt based on format
        description = f"**Question {self.progress + 1} of {len(self.questions)}**\n\n{q_text}"
        if "a" in q_data and "b" in q_data:
            description += f"\n\n**A)** {q_data['a']}\n**B)** {q_data['b']}"
            # Show A/B buttons, hide 1-5
            self.enable_ab_mode()
        else:
            self.enable_likert_mode()
            if self.instructions:
                description += f"\n\n{self.instructions}"

        embed = discord.Embed(
            title=f"Assessment: {self.quiz_type.upper()}",
            description=description,
            color=discord.Color.blue()
        )
        embed = apply_disclaimer(embed)

        try:
            if not interaction.response.is_done():
                await interaction.response.edit_message(embed=embed, view=self)
            else:
                await interaction.edit_original_response(embed=embed, view=self)
        except discord.NotFound:
            # Interaction expired or message deleted - restart quiz
            log.warning("Quiz interaction expired for user %s, session cleared", self.user_id)
            try:
                await self.bot.db.execute("DELETE FROM quiz_sessions WHERE user_id = ?", (str(self.user_id),))
                await self.bot.db.commit()
            except:
                pass
            # Try to send new message in DM
            try:
                user = self.bot.get_user(self.user_id)
                if user:
                    await user.send("⚠️ Your session expired. Use `!assessment mbti` to start a new quiz.")
            except:
                pass
        except Exception as e:
            log.error("Quiz update failed: %s", e)
            # Try to respond ephemerally if edit fails
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message("⚠️ Something went wrong. Please restart the quiz.", ephemeral=True)
                else:
                    await interaction.followup.send("⚠️ Something went wrong. Please restart the quiz.", ephemeral=True)
            except:
                pass

    def enable_ab_mode(self):
        """Switches UI to A/B buttons."""
        self.btn_a.disabled = False
        self.btn_b.disabled = False
        for btn in [self.c1, self.c2, self.c3, self.c4, self.c5]:
            btn.disabled = True

    def enable_likert_mode(self):
        """Switches UI to 1-5 Likert buttons."""
        self.btn_a.disabled = True
        self.btn_b.disabled = True
        for btn in [self.c1, self.c2, self.c3, self.c4, self.c5]:
            btn.disabled = False

    async def handle_choice(self, interaction: discord.Interaction, value: str):
        """Handles choice selection and saves progress."""
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("This session is not yours.", ephemeral=True)

        # Removed defer to process inline and avoid webhook DM interaction failures

        self.answers.append(value)
        self.progress += 1

        # Persistence
        try:
            await self.bot.db.execute(
                "INSERT OR REPLACE INTO quiz_sessions (user_id, quiz_type, progress, answers) VALUES (?, ?, ?, ?)",
                (str(self.user_id), self.quiz_type, self.progress, json.dumps(self.answers))
            )
            await self.bot.db.commit()
        except Exception as e:
            log.error("Quiz progress save failed: %s", e)
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message("❌ Failed to save progress. Please try again.", ephemeral=True)
                else:
                    await interaction.followup.send("❌ Failed to save progress. Please try again.", ephemeral=True)
            except:
                pass
            return

        try:
            await self.update_question(interaction)
        except Exception as e:
            log.error("Quiz update failed: %s", e)
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message("⚠️ Something went wrong. Please use `!assessment_resume` to continue.", ephemeral=True)
                else:
                    await interaction.followup.send("⚠️ Something went wrong. Please use `!assessment_resume` to continue.", ephemeral=True)
            except:
                pass

    # UI BUTTONS
    @discord.ui.button(label="A", style=discord.ButtonStyle.green, disabled=True)
    async def btn_a(self, interaction, button): await self.handle_choice(interaction, "A")
    
    @discord.ui.button(label="B", style=discord.ButtonStyle.green, disabled=True)
    async def btn_b(self, interaction, button): await self.handle_choice(interaction, "B")

    @discord.ui.button(label="1", style=discord.ButtonStyle.grey)
    async def c1(self, interaction, button): await self.handle_choice(interaction, "1")
    @discord.ui.button(label="2", style=discord.ButtonStyle.grey)
    async def c2(self, interaction, button): await self.handle_choice(interaction, "2")
    @discord.ui.button(label="3", style=discord.ButtonStyle.grey)
    async def c3(self, interaction, button): await self.handle_choice(interaction, "3")
    @discord.ui.button(label="4", style=discord.ButtonStyle.grey)
    async def c4(self, interaction, button): await self.handle_choice(interaction, "4")
    @discord.ui.button(label="5", style=discord.ButtonStyle.grey)
    async def c5(self, interaction, button): await self.handle_choice(interaction, "5")

    async def finish_quiz(self, interaction: discord.Interaction):
        """Zero-CPU Completion: No math, just data handoff."""
        timestamp = datetime.now().isoformat()
        raw_data = json.dumps(self.answers)

        try:
            async with self.bot.db.cursor() as cursor:
                # Move from active session to permanent results
                await cursor.execute(
                    "DELETE FROM quiz_results WHERE user_id = ? AND quiz_type = ?",
                    (str(self.user_id), self.quiz_type)
                )
                await cursor.execute(
                    "INSERT INTO quiz_results (user_id, quiz_type, raw_answers, timestamp) VALUES (?, ?, ?, ?)",
                    (str(self.user_id), self.quiz_type, raw_data, timestamp)
                )
                await cursor.execute("DELETE FROM quiz_sessions WHERE user_id = ?", (str(self.user_id),))

            await self.bot.db.commit()

            embed = discord.Embed(
                title="✅ Assessment Secured",
                description=f"Your raw **{self.quiz_type.upper()}** responses have been stored. Gemini will interpret these in your next Deep Synthesis.",
                color=discord.Color.green()
            )

            try:
                if not interaction.response.is_done():
                    await interaction.response.edit_message(embed=apply_disclaimer(embed), view=None)
                else:
                    await interaction.edit_original_response(embed=apply_disclaimer(embed), view=None)
            except:
                # Fallback: send DM
                user = self.bot.get_user(self.user_id)
                if user:
                    await user.send(embed=apply_disclaimer(embed))
        except discord.NotFound:
            # Interaction expired - data still saved, just can't show message
            log.warning("Quiz finish: interaction expired but data saved for user %s", self.user_id)
            # Data is saved, try to send DM
            try:
                user = self.bot.get_user(self.user_id)
                if user:
                    embed = discord.Embed(
                        title="✅ Assessment Secured",
                        description=f"Your **{self.quiz_type.upper()}** responses have been saved.",
                        color=discord.Color.green()
                    )
                    await user.send(embed=apply_disclaimer(embed))
            except Exception as e:
                log.error("Failed to DM user %s: %s", self.user_id, e)
        except Exception as e:
            log.error("Quiz finish failed: %s", e)
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message("❌ Failed to save quiz. Please try again.", ephemeral=True)
                else:
                    await interaction.followup.send("❌ Failed to save quiz. Please try again.", ephemeral=True)
            except:
                pass

    async def on_timeout(self):
        """Handle view timeout - notify user they can resume."""
        try:
            log.info("⏱️ AssessmentView timed out for user %s", self.user_id)
            user = self.bot.get_user(self.user_id)
            if user:
                try:
                    await user.send("⏱️ Your assessment session timed out. Use `!assessment_resume` to continue where you left off.")
                except discord.Forbidden:
                    log.warning("Could not send DM to user %s on timeout", self.user_id)
        except Exception as e:
            log.error("Timeout notification failed for user %s: %s", self.user_id, e)

# =============================================================================
# 5. PHASE 4: QUIZ DATA & ENGINE
# =============================================================================

# QUIZ SCORING LOGIC REMOVED (Migrated to Gemini Deep Synthesis Engine)

# =============================================================================
# 6. PHASE 3: ANALYSIS HELPERS & COOLDOWNS
# =============================================================================

import re as _re

# Pre-compiled cleaning patterns (module-level for performance)
_CUSTOM_EMOJI = _re.compile(r'<a?:\w+:\d+>')          # <:pepe:123456>
_USER_MENTION = _re.compile(r'<@!?\d+>')               # <@123456>
_CHANNEL_MENTION = _re.compile(r'<#\d+>')               # <#123456>
_ROLE_MENTION = _re.compile(r'<@&\d+>')                 # <@&123456>
_MULTI_WHITESPACE = _re.compile(r'\s{3,}')              # Collapse excessive whitespace
_REPEATED_CHARS = _re.compile(r'(.)\1{4,}')             # aaaaaaaa → aa

def _clean_message(content: str) -> str:
    """Strip Discord-specific noise from a message to save tokens."""
    text = content
    text = _CUSTOM_EMOJI.sub('', text)                   # Remove custom emojis
    text = _USER_MENTION.sub('', text)                    # Remove user mentions (resolved separately)
    text = _CHANNEL_MENTION.sub('[channel]', text)
    text = _ROLE_MENTION.sub('[role]', text)
    text = _REPEATED_CHARS.sub(r'\1\1', text)            # Collapse repeated chars
    text = _MULTI_WHITESPACE.sub(' ', text)              # Collapse whitespace
    return text.strip()

def _resolve_user(guild: discord.Guild, user_id: str) -> str:
    """Resolve a user ID to a display name, safely."""
    try:
        member = guild.get_member(int(user_id))
        return f"@{member.display_name}" if member else user_id
    except (ValueError, TypeError):
        return user_id

def format_transcript(rows, max_chars=150000, guild=None, reply_map=None, include_threads=False, thread_graph=None):
    """
    Formats raw DB rows into a readable script with deduplication.
    Default cap is 150k chars (~40k tokens) to fit all AI provider limits.

    Rows expected: (content, timestamp, reply_to_id, user_id)
    guild: discord.Guild — used to resolve user IDs to display names.
    reply_map: dict of {message_id: user_id} — used to resolve reply targets.
    include_threads: bool — if True, group messages by reply chains instead of chronologically.
    thread_graph: dict — the graph structure for thread grouping (from build_reply_graph).
    """
    # Handle thread-based formatting
    if include_threads and thread_graph:
        return format_threaded_transcript(rows, thread_graph, guild, max_chars)

    transcript = []
    total_chars = 0
    last_content = None
    dupe_count = 0

    for row in rows:
        content = _clean_message(row[0])
        timestamp = row[1]
        reply_to_id = row[2]
        user_id = row[3] if len(row) > 3 else None

        # Resolve author name
        author_name = _resolve_user(guild, user_id) if guild and user_id else (user_id or "User")

        # Resolve reply target
        reply_note = ""
        if reply_to_id and reply_map and reply_to_id in reply_map:
            target_id = reply_map[reply_to_id]
            target_name = _resolve_user(guild, target_id) if guild else target_id
            reply_note = f" (→ {target_name})"

        # Skip empty results after cleaning
        if not content or len(content) < 3:
            continue

        # Deduplicate consecutive identical messages
        if content == last_content:
            dupe_count += 1
            continue
        if dupe_count > 0:
            transcript.append(f"  [repeated {dupe_count} more time(s)]")
            dupe_count = 0
        last_content = content

        # Truncate extremely long copypastas
        if len(content) > 300:
            content = content[:300] + "...[truncated]"

        line = f"[{timestamp[:10]}] {author_name}{reply_note}: {content}"
        total_chars += len(line)

        # Optional token budget guard (disabled when max_chars=None)
        if max_chars and total_chars > max_chars:
            transcript.append(f"[...{len(rows) - len(transcript)} older messages omitted for brevity...]")
            break

        transcript.append(line)

    if dupe_count > 0:
        transcript.append(f"  [repeated {dupe_count} more time(s)]")

    return "\n".join(transcript)


# =============================================================================
# DFS-BASED CONVERSATION THREAD ANALYSIS
# =============================================================================

# Default depth limit to prevent infinite loops
thread_depth_limit = 10


async def build_reply_graph(user_id: str, guild_id: str) -> dict:
    """
    Builds a reply graph for a given user.
    Returns adjacency list: {message_id: [reply_message_ids]}
    Also includes reverse mapping for replies TO the user.
    """
    graph = {}  # {parent_id: [child_ids]}
    message_authors = {}  # {message_id: user_id}

    # Fetch messages where user is the author (filtered by guild)
    async with bot.db.execute(
        "SELECT message_id, user_id, reply_to_id FROM interaction_history WHERE user_id = ? AND guild_id = ?",
        (user_id, guild_id)
    ) as cursor:
        user_messages = await cursor.fetchall()

    # Store message authors and build graph
    for row in user_messages:
        msg_id, author_id, reply_to = row
        message_authors[msg_id] = author_id
        if reply_to:
            # Add to graph: parent -> child relationship
            if reply_to not in graph:
                graph[reply_to] = []
            graph[reply_to].append(msg_id)

    # Fetch messages where user is being replied TO (replies to user's messages)
    user_msg_ids = [row[0] for row in user_messages]
    if user_msg_ids:
        placeholders = ','.join('?' * len(user_msg_ids))
        async with bot.db.execute(
            f"SELECT message_id, user_id, reply_to_id FROM interaction_history WHERE reply_to_id IN ({placeholders}) AND guild_id = ?",
            tuple(user_msg_ids) + (guild_id,)
        ) as cursor:
            replies_to_user = await cursor.fetchall()

        for row in replies_to_user:
            msg_id, author_id, reply_to = row
            message_authors[msg_id] = author_id
            if reply_to:
                if reply_to not in graph:
                    graph[reply_to] = []
                graph[reply_to].append(msg_id)

    return {
        "graph": graph,
        "authors": message_authors
    }


async def extract_threads_with_db(graph: dict, root_message_id: str, depth_limit: int = 10) -> list:
    """
    Uses DFS to extract conversation threads with full message data from DB.
    Returns list of messages with: message_id, author_id, content, timestamp, reply_to_id, depth.
    """
    if isinstance(graph, dict) and "graph" in graph:
        adjacency = graph["graph"]
    else:
        adjacency = graph

    all_descendants = []
    visited = set()

    def collect_descendants(node: str, depth: int):
        if node in visited or depth > depth_limit:
            return
        visited.add(node)
        all_descendants.append(node)

        for child in adjacency.get(node, []):
            if child not in visited:
                collect_descendants(child, depth + 1)

    collect_descendants(root_message_id, 0)

    if not all_descendants:
        return []

    # Batch fetch all message details
    placeholders = ','.join('?' * len(all_descendants))
    async with bot.db.execute(
        f"SELECT message_id, user_id, content, timestamp, reply_to_id FROM interaction_history WHERE message_id IN ({placeholders})",
        tuple(all_descendants)
    ) as cursor:
        rows = await cursor.fetchall()

    # Build message lookup
    msg_data = {}
    for row in rows:
        msg_id, author_id, content, timestamp, reply_to = row
        msg_data[msg_id] = {
            "message_id": msg_id,
            "author_id": author_id,
            "content": content,
            "timestamp": timestamp,
            "reply_to_id": reply_to
        }

    # Build tree structure from root using DFS
    result = []
    node_visited = set()

    def traverse(node_id: str, depth: int):
        if node_id in node_visited or depth > depth_limit:
            return
        node_visited.add(node_id)

        msg = msg_data.get(node_id)
        if msg:
            result.append({
                "message_id": node_id,
                "author_id": msg.get("author_id"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", ""),
                "reply_to_id": msg.get("reply_to_id"),
                "depth": depth
            })

        for child_id in adjacency.get(node_id, []):
            traverse(child_id, depth + 1)

    traverse(root_message_id, 0)
    return result


async def find_top_threads_for_user(user_id: str, guild_id: str, top_n: int = 5) -> list:
    """
    Find the top N most active conversation threads for a user.
    Returns list of thread info with: root_message_id, total_messages, participants, topics.
    """
    async with bot.db.execute(
        "SELECT message_id FROM interaction_history WHERE user_id = ? AND guild_id = ?",
        (user_id, guild_id)
    ) as cursor:
        user_messages = await cursor.fetchall()

    if not user_messages:
        return []

    user_msg_ids = [row[0] for row in user_messages]
    thread_stats = []

    for msg_id in user_msg_ids:
        graph = await build_reply_graph(user_id, guild_id)
        thread = await extract_threads_with_db(graph, msg_id, thread_depth_limit)

        if len(thread) > 1:
            participants = set()
            topics = []
            for msg in thread:
                if msg.get("author_id"):
                    participants.add(msg["author_id"])
                content = msg.get("content", "")
                if content and len(topics) < 5:
                    words = content.split()[:3]
                    if words:
                        topics.append(" ".join(words))

            thread_stats.append({
                "root_message_id": msg_id,
                "total_messages": len(thread),
                "participants": list(participants),
                "participant_count": len(participants),
                "topics": topics
            })

    thread_stats.sort(key=lambda x: x["total_messages"], reverse=True)
    return thread_stats[:top_n]


def format_threaded_transcript(rows, graph: dict, guild=None, max_chars=150000) -> str:
    """
    Formats messages grouped by their reply chains instead of chronologically.
    """
    if isinstance(graph, dict) and "graph" in graph:
        adjacency = graph["graph"]
    else:
        adjacency = graph

    msg_lookup = {}
    for row in rows:
        msg_id = row[0] if row[0] else None
        if msg_id:
            msg_lookup[msg_id] = {
                "content": _clean_message(row[1]) if len(row) > 1 else "",
                "timestamp": row[2] if len(row) > 2 else "",
                "user_id": row[3] if len(row) > 3 else None,
                "reply_to": row[4] if len(row) > 4 else None
            }

    # Identify root messages
    all_msg_ids = set(msg_lookup.keys())
    root_candidates = []
    for msg_id, data in msg_lookup.items():
        reply_to = data.get("reply_to")
        if not reply_to or reply_to not in all_msg_ids:
            root_candidates.append(msg_id)

    root_candidates.sort(key=lambda m: msg_lookup.get(m, {}).get("timestamp", ""))

    transcript = []
    total_chars = 0

    for root_id in root_candidates:
        thread_lines = []
        visited = set()

        def traverse(node_id: str, depth: int):
            if node_id in visited or depth > thread_depth_limit:
                return
            visited.add(node_id)

            msg = msg_lookup.get(node_id)
            if not msg:
                return

            content = msg.get("content", "")
            if not content or len(content) < 3:
                return

            user_id = msg.get("user_id")
            author_name = _resolve_user(guild, user_id) if guild and user_id else (user_id or "User")
            timestamp = msg.get("timestamp", "")[:10]

            indent = "  " * depth
            line = f"{indent}[{timestamp}] {author_name}: {content[:200]}"
            thread_lines.append(line)

            for child_id in adjacency.get(node_id, []):
                traverse(child_id, depth + 1)

        traverse(root_id, 0)

        if thread_lines:
            transcript.append(f"--- Thread (root: {root_id[:8]}...) ---")
            for line in thread_lines:
                total_chars += len(line)
                if max_chars and total_chars > max_chars:
                    break
                transcript.append(line)
            transcript.append("")

    if total_chars > max_chars:
        transcript.append("[... transcript truncated for length limit ...]")

    return "\n".join(transcript)


# =============================================================================
# 6. BOT CLASS (Lifecycle Managed)
# =============================================================================

class PsycheBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db: aiosqlite.Connection = None

    async def setup_hook(self):
        """Persistent DB connection startup."""
        self.db = await aiosqlite.connect(DB_PATH)
        self.db.row_factory = aiosqlite.Row
        await init_db(self.db)

    async def close(self):
        """Graceful shutdown."""
        log.info("Shutting down Psyche...")
        if self.db:
            await self.db.close()
            self.db = None
            log.info("Database connection closed.")
        await super().close()

    async def start(self, token, *, reconnect=True):
        """
        Override to retry login on transient network failures.
        Retries happen INSIDE the async context so the HTTP session stays alive.
        """
        max_retries = 10
        for attempt in range(max_retries):
            try:
                log.info("Connecting to Discord (attempt %d/%d)...", attempt + 1, max_retries)
                await self.login(token)
                break
            except (aiohttp.ClientConnectorError, ConnectionResetError, OSError) as e:
                if attempt == max_retries - 1:
                    log.critical("Failed to login after %d attempts. Giving up.", max_retries)
                    raise
                wait = min(15 * (2 ** attempt), 300)
                log.warning(
                    "Login failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1, max_retries, e, wait
                )
                await asyncio.sleep(wait)
        await self.connect(reconnect=reconnect)

    async def on_ready(self):
        activity = discord.Activity(type=discord.ActivityType.watching, name="patterns | !help")
        await self.change_presence(status=discord.Status.online, activity=activity)
        scan_m = os.getenv('SCAN_MODEL', 'Unknown')
        dossier_m = os.getenv('DOSSIER_MODEL', 'Unknown')
        log.info(f"✨ Psyche v2 Online | Scan: {scan_m} | Dossier: {dossier_m}")

        # Auto-start daily scraper if enabled via env var
        auto_scrape = os.getenv('AUTO_SCRAPE', 'false').lower()
        if auto_scrape in ('true', '1', 'yes'):
            if start_daily_scrape():
                log.info("🚀 Auto-started daily scraper (AUTO_SCRAPE=true)")

# =============================================================================
# 7. EVENT GATES (Privacy & Scrub Protocol)
# =============================================================================

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True  # Required for scheduled tasks
bot = PsycheBot(
    command_prefix='!',
    intents=intents,
    help_command=None
)

# =============================================================================
# DAILY SCRAPER TASK (Background)
# =============================================================================

# Noise filter patterns
import re as _re_scraper
_LINK_ONLY_SCRAPER = _re_scraper.compile(r'^https?://\S+$')
_PURE_EMOJI_SCRAPER = _re_scraper.compile(r'^[\U0001F000-\U0001FFFF☀-⟿︀-️‍\s]+$')
_BOT_CMD_SCRAPER = _re_scraper.compile(r'^[!?./]\w+')
_MEDIA_ONLY_SCRAPER = _re_scraper.compile(r'^https?://\S+\.(png|jpg|jpeg|gif|webp|mp4|mov|webm)(\?\S*)?$', _re_scraper.IGNORECASE)
_NOISE_WORDS_SCRAPER = {'k', 'ok', 'okay', 'yes', 'no', 'yep', 'nah', 'nope', 'ya', 'ye',
                        'lol', 'lmao', 'lmfao', 'haha', 'hahaha', 'heh', 'xd',
                        'bruh', 'oof', 'rip', 'gg', 'ez', 'f', 'w', 'l', 'ty', 'np',
                        'ikr', 'idk', 'idc', 'smh', 'tbh', 'fr', 'ngl', 'omg', 'wtf',
                        'hmm', 'ah', 'oh', 'uh', 'mhm', 'yea', 'yeh', 'wow', 'dam',
                        'damn', 'nice', 'cool', 'sure', 'bet', 'kk', 'ig', 'fs'}

def _is_noise_scraper(content: str) -> bool:
    """Check if message has no behavioral value."""
    text = content.strip()
    if not text or len(text) < 3:
        return True
    if _BOT_CMD_SCRAPER.match(text) or _LINK_ONLY_SCRAPER.match(text) or _MEDIA_ONLY_SCRAPER.match(text):
        return True
    if _PURE_EMOJI_SCRAPER.match(text):
        return True
    if text.lower() in _NOISE_WORDS_SCRAPER and len(text) < 10:
        return True
    if len(set(text.lower().replace(' ', ''))) <= 2 and len(text) > 3:
        return True
    return False

async def _scrape_guild(guild: discord.Guild, report_channel: discord.TextChannel = None, incremental: bool = True):
    """Scrapes all messages from a guild and stores in DB.
    If incremental=True, only scrapes new messages since last checkpoint."""
    log.info(f"🕷️ Starting scrape for guild: {guild.name} (incremental={incremental})")

    all_channels = guild.text_channels + list(guild.threads)
    batch_rows = []
    total_scanned = 0
    total_stored = 0

    for channel in all_channels:
        if not hasattr(channel, 'history'):
            continue
        perms = channel.permissions_for(guild.me)
        if not perms.read_message_history or not perms.read_messages:
            continue

        channel_id_str = str(channel.id)

        # Get global checkpoint for this channel
        last_msg_id = await get_global_checkpoint(bot.db, channel_id_str) if incremental else None

        try:
            # Determine starting point
            if last_msg_id:
                try:
                    anchor_msg = await channel.fetch_message(int(last_msg_id))
                    message_iter = channel.history(limit=None, oldest_first=True, after=anchor_msg)
                except Exception:
                    # Any error (NotFound, Forbidden, HTTPException, rate limit) - do full scan
                    message_iter = channel.history(limit=None, oldest_first=True)
            else:
                message_iter = channel.history(limit=None, oldest_first=True)

            last_processed_id = last_msg_id

            async for message in message_iter:
                total_scanned += 1
                last_processed_id = str(message.id)

                # Skip bots
                if message.author.bot:
                    continue

                # Skip noise
                if _is_noise_scraper(message.content):
                    continue

                reply_id = str(message.reference.message_id) if message.reference else None

                batch_rows.append((
                    str(message.id),
                    str(message.author.id),
                    str(guild.id),
                    message.content,
                    reply_id,
                    message.created_at.isoformat()
                ))
                total_stored += 1

                # Batch commit every 500 messages
                if len(batch_rows) >= 500:
                    try:
                        await bot.db.executemany(
                            "INSERT OR IGNORE INTO interaction_history "
                            "(message_id, user_id, guild_id, content, reply_to_id, timestamp) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            batch_rows
                        )
                        await bot.db.commit()

                        # Progress update every 1000 messages
                        if report_channel and total_scanned % 1000 == 0:
                            try:
                                await report_channel.send(
                                    f"📡 Scraping... `{total_scanned:,}` messages scanned, `{total_stored:,}` stored"
                                )
                            except Exception as e:
                                log.warning(f"Failed to send progress message: {e}")
                    except Exception as e:
                        log.warning(f"Batch insert failed: {e}")

                    batch_rows.clear()

            # Update checkpoint after successfully processing channel
            if last_processed_id and last_processed_id != last_msg_id:
                await set_global_checkpoint(bot.db, channel_id_str, last_processed_id)

        except discord.Forbidden:
            continue
        except Exception as e:
            log.warning(f"Channel {channel.name} scrape error: {e}")
            continue

    # Final commit
    if batch_rows:
        try:
            await bot.db.executemany(
                "INSERT OR IGNORE INTO interaction_history "
                "(message_id, user_id, guild_id, content, reply_to_id, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                batch_rows
            )
            await bot.db.commit()
        except Exception as e:
            log.warning(f"Final batch insert failed: {e}")

    log.info(f"✅ Daily scrape complete for {guild.name}: {total_stored:,} messages stored")
    return total_scanned, total_stored

# Global task instance
_daily_scrape_task = None
_scrape_interval_hours = int(os.getenv('SCRAPE_INTERVAL_HOURS', '24'))

@tasks.loop(hours=_scrape_interval_hours)
async def daily_scrape():
    """Scheduled task: runs daily to scrape all server messages."""
    if not bot.is_ready():
        log.warning("Daily scrape skipped: bot not ready")
        return

    log.info("🕷️ Starting scheduled daily scrape...")

    # Find a channel to report progress to
    report_channel = None
    if COMMAND_CHANNEL_ID:
        for guild in bot.guilds:
            channel = guild.get_channel(COMMAND_CHANNEL_ID)
            if channel:
                report_channel = channel
                break

    if report_channel:
        try:
            start_msg = await report_channel.send("🕷️ **Daily scrape started...** This runs in the background.")
        except Exception as e:
            log.warning(f"Failed to send start message: {e}")
            start_msg = None
    else:
        start_msg = None

    scraped_total = 0
    stored_total = 0

    for guild in bot.guilds:
        try:
            sc, st = await _scrape_guild(guild, report_channel)
            scraped_total += sc
            stored_total += st
        except Exception as e:
            log.warning(f"Scrape failed for guild {guild.name}: {e}")

    log.info(f"✅ Daily scrape done: {scraped_total:,} scanned, {stored_total:,} stored")

    # Send completion message to Discord
    if report_channel:
        try:
            await report_channel.send(
                f"✅ **Daily scrape complete**\n"
                f"📊 Scanned: `{scraped_total:,}` messages\n"
                f"💾 Stored: `{stored_total:,}` interactions"
            )
        except Exception as e:
            log.warning(f"Failed to send completion message: {e}")

@daily_scrape.before_loop
async def before_daily_scrape():
    await bot.wait_until_ready()

def start_daily_scrape():
    """Start the daily scraper."""
    global _daily_scrape_task
    if not daily_scrape.is_running():
        daily_scrape.start()
        log.info("🚀 Daily scraper started")
        return True
    return False

def stop_daily_scrape():
    """Stop the daily scraper."""
    if daily_scrape.is_running():
        daily_scrape.cancel()
        log.info("🛑 Daily scraper stopped")
        return True
    return False

# =============================================================================
# GLOBAL COMMAND GATE (Channel Restriction + Owner Bypass)
# =============================================================================

async def command_channel_check(ctx: commands.Context) -> bool:
    """Restricts commands to COMMAND_CHANNEL_ID (if set), bypasses for owners."""
    if ctx.command and ctx.command.name == "help":
        return True
    # No channel restriction if not configured
    if not COMMAND_CHANNEL_ID:
        return True
    # Allow if command is in the designated channel
    if ctx.channel.id == COMMAND_CHANNEL_ID:
        return True
    # Block otherwise
    await ctx.send(f"⛔ **Channel Lock:** Commands only work in <#{COMMAND_CHANNEL_ID}>", ephemeral=True)
    return False

# Register global check for all commands
bot.add_check(command_channel_check)

@bot.event
async def on_command_error(ctx, error):
    """Global error handler to catch silent command failures."""
    if isinstance(error, commands.CommandNotFound):
        return
    log.error("Command Error in %s: %s", ctx.command.name if ctx.command else "Unknown", error)
    try:
        await ctx.send(f"⚠️ **Command Execution Failed:** {str(error)}")
    except:
        pass

@bot.event
async def on_message(message: discord.Message):
    """The Message Gate: Privacy-first logging."""
    if message.author.bot:
        return
        
    log.info(f"Message received from {message.author}: {message.content}")

    # Process commands immediately
    log.info("Attempting to process commands...")
    await bot.process_commands(message)

    # Skip behavioral logging for command invocations (no analytical value)
    if message.content.startswith('!'):
        return

    # Phase 2: Log only if opted-in AND in a Server (No DM logging)
    log.info(f"Checking opt-in for {message.author}...")
    if not isinstance(message.channel, discord.DMChannel) and is_opted_in(message.author):
        log.info(f"User {message.author} is opted in. Saving to database.")
        # Skip very short messages with no analytical value at ingest time
        if len(message.content.strip()) < 3:
            return
        try:
            await bot.db.execute(
                "INSERT INTO messages (user_id, guild_id, content) VALUES (?, ?, ?)",
                (str(message.author.id), str(message.guild.id), message.content)
            )
            await bot.db.commit()
        except Exception as e:
            log.error("DB Log Error: %s", e)

@bot.event
async def on_guild_remove(guild: discord.Guild):
    """The Scrub Protocol (Guild-Level): Wipes ALL data for this server."""
    log.info("🧹 Guild Scrub: Purging all data for %s (%s)", guild.name, guild.id)
    try:
        gid = str(guild.id)
        await bot.db.execute("DELETE FROM messages WHERE guild_id = ?", (gid,))
        # Wipe checkpoints for all channels in this guild
        for ch in guild.text_channels:
            await bot.db.execute("DELETE FROM sync_checkpoints WHERE channel_id = ?", (str(ch.id),))
        await bot.db.commit()
        log.info("✅ Guild Scrub complete for %s.", guild.id)
    except Exception as e:
        log.error("Guild Scrub Error: %s", e)

@bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    """Scrub Protocol (Role Removal): If PsycheOptIn is removed, purge user data."""
    had_role = any(r.name.lower() == "psycheoptin" for r in before.roles)
    has_role = any(r.name.lower() == "psycheoptin" for r in after.roles)
    if had_role and not has_role:
        log.info("🚨 PsycheOptIn removed from %s. Triggering user purge.", after.id)
        await purge_user_data(str(after.id))

# =============================================================================
# 8. CLINICAL ASSESSMENT COMMANDS
# =============================================================================

@bot.command(name='assessment')
async def assessment(ctx, quiz_type: str = None):
    """Starts a new assessment in DMs."""
    if not is_opted_in(ctx.author):
        return await ctx.reply("🔒 **Privacy Gate**: You need the `PsycheOptIn` role.")

    quiz_type = quiz_type.lower() if quiz_type else None
    if quiz_type == "big5":
        quiz_type = "ocean"

    if not quiz_type or quiz_type not in ["ocean", "mbti", "enneagram"]:
        return await ctx.send("❌ Use: `!assessment [ocean|big5|mbti|enneagram]`")

    # Load questions from local JSON with path hardening
    try:
        with open(QUESTIONS_JSON, 'r') as f:
            data = json.load(f)
        quiz_type_data = data.get(quiz_type, {})
        questions = quiz_type_data.get("questions", [])
        instructions = quiz_type_data.get("instructions", "")
    except FileNotFoundError:
        return await ctx.send("❌ **System Error**: `questions.json` missing.")
    except Exception as e:
        return await ctx.send(f"❌ **System Error**: {str(e)}")

    if not questions:
        return await ctx.send(f"❌ **Data Void**: No questions found for test type `{quiz_type}`.")

    # Check for existing session
    async with bot.db.execute("SELECT quiz_type FROM quiz_sessions WHERE user_id = ?", (str(ctx.author.id),)) as cursor:
        row = await cursor.fetchone()
        if row:
            existing_type = row[0]
            if existing_type == quiz_type:
                # Same type: delete to allow restart
                await bot.db.execute("DELETE FROM quiz_sessions WHERE user_id = ?", (str(ctx.author.id),))
                await bot.db.commit()
                log.info("Cleared existing %s session for user %s to start fresh", quiz_type, ctx.author.id)
            else:
                # Different type: block
                return await ctx.send(f"⚠️ You have an active **{existing_type.upper()}** session. Use `!assessment_resume` to continue.")

    view = AssessmentView(bot, ctx.author.id, quiz_type, questions, instructions=instructions)
    q_data = questions[0]
    description = f"**Question 1 of {len(questions)}**\n\n{q_data['q']}"
    if "a" in q_data and "b" in q_data:
        description += f"\n\n**A)** {q_data['a']}\n**B)** {q_data['b']}"
        view.enable_ab_mode()
    else:
        view.enable_likert_mode()
        if instructions:
            description += f"\n\n{instructions}"

    embed = discord.Embed(
        title=f"Starting {quiz_type.upper()} (Inst len: {len(instructions)})",
        description=description
    )

    try:
        await ctx.author.send(embed=apply_disclaimer(embed), view=view)
        await ctx.send(f"📩 Check your DMs, {ctx.author.mention}, to begin the assessment.")
    except discord.Forbidden:
        await ctx.send("❌ I couldn't DM you. Please open your DMs and try again.")

@bot.command(name='assessment_resume')
async def assessment_resume(ctx):
    """Resumes a saved assessment from the HF Bucket."""
    async with bot.db.execute(
        "SELECT quiz_type, progress, answers FROM quiz_sessions WHERE user_id = ?",
        (str(ctx.author.id),)
    ) as cursor:
        row = await cursor.fetchone()
        if not row:
            return await ctx.send("❌ No active assessment found to resume.")
        
        quiz_type, progress, answers_json = row
        answers = json.loads(answers_json)

    try:
        with open(QUESTIONS_JSON, 'r') as f:
            data = json.load(f)
        quiz_type_data = data.get(quiz_type, {})
        questions = quiz_type_data.get("questions", [])
        instructions = quiz_type_data.get("instructions", "")
    except Exception:
        return await ctx.send("❌ **System Error**: `questions.json` inaccessible.")

    if not questions:
        return await ctx.send("❌ **System Error**: Questions data unavailable.")

    if progress >= len(questions):
        # Session is corrupted (completed but not cleared, or questions.json changed)
        # Clear the corrupted session and guide user to restart
        await bot.db.execute("DELETE FROM quiz_sessions WHERE user_id = ?", (str(ctx.author.id),))
        await bot.db.commit()
        return await ctx.send("⚠️ **Session Expired**: Your assessment progress was lost (likely due to timeout or data update). Use `!assessment mbti` to start fresh.")

    view = AssessmentView(bot, ctx.author.id, quiz_type, questions, progress, answers, instructions=instructions)
    q_data = questions[progress]
    description = f"**Question {progress + 1} of {len(questions)}**\n\n{q_data['q']}"
    if "a" in q_data and "b" in q_data:
        description += f"\n\n**A)** {q_data['a']}\n**B)** {q_data['b']}"
        view.enable_ab_mode()
    else:
        view.enable_likert_mode()
        if instructions:
            description += f"\n\n{instructions}"

    embed = discord.Embed(
        title=f"Resuming {quiz_type.upper()}",
        description=description
    )
    try:
        await ctx.author.send(embed=apply_disclaimer(embed), view=view)
        await ctx.send("📩 Resuming in DMs...")
    except discord.Forbidden:
        await ctx.send("❌ I couldn't DM you. Please open your DMs and try again.")

# =============================================================================
# 8. INTERACTION SCRAPER COMMAND (Free-Tier Optimized)
# =============================================================================

@bot.command(name='map_interactions')
async def map_interactions(ctx, force: str = None):
    """Maps your interaction history. Use 'force' for full rescan."""

    # 1. Privacy & Server Gate
    if not is_opted_in(ctx.author):
        return await ctx.send("🔒 **Access Denied:** You must have the `PsycheOptIn` role to map your interactions.")

    if not ctx.guild:
        return await ctx.send("⚠️ This command must be run inside the private server, not in DMs.")

    # 2. Cooldown Check (24h, bypassed for Owner/Admins)
    user_id = str(ctx.author.id)
    current_time = time.time()
    is_vip = ctx.author.guild_permissions.administrator

    if not is_vip:
        async with bot.db.execute("SELECT last_map_time FROM cooldowns WHERE user_id = ?", (user_id,)) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                elapsed = current_time - row[0]
                if elapsed < 86400:  # 24 hours
                    hours_left = round((86400 - elapsed) / 3600, 1)
                    return await ctx.send(f"⏳ **Cooldown Active:** You can run this again in {hours_left} hours.")

    # 3. UI Feedback
    status_msg = await resilient_call(
        ctx.send,
        "🔍 **Initializing Total Interaction Scraper...**\n"
        "*This may take a while. The database is batching writes to conserve CPU.*"
    )
    
    total_mapped = 0
    total_scanned = 0
    total_filtered = 0
    batch_rows = []

    # --- Smart Noise Filter ---
    import re
    _LINK_ONLY = re.compile(r'^https?://\S+$')
    _PURE_EMOJI = re.compile(r'^[\U0001F000-\U0001FFFF\u2600-\u27FF\uFE00-\uFE0F\u200d\s]+$')
    _BOT_CMD = re.compile(r'^[!?./]\w+')
    _MEDIA_ONLY = re.compile(r'^https?://\S+\.(png|jpg|jpeg|gif|webp|mp4|mov|webm)(\?\S*)?$', re.IGNORECASE)
    _NOISE_WORDS = {'k', 'ok', 'okay', 'yes', 'no', 'yep', 'nah', 'nope', 'ya', 'ye',
                    'lol', 'lmao', 'lmfao', 'haha', 'hahaha', 'heh', 'xd',
                    'bruh', 'oof', 'rip', 'gg', 'ez', 'f', 'w', 'l', 'ty', 'np',
                    'ikr', 'idk', 'idc', 'smh', 'tbh', 'fr', 'ngl', 'omg', 'wtf',
                    'hmm', 'ah', 'oh', 'uh', 'mhm', 'yea', 'yeh', 'wow', 'dam',
                    'damn', 'nice', 'cool', 'sure', 'bet', 'kk', 'ig', 'fs'}

    def is_noise(content: str) -> bool:
        """Returns True if the message has no behavioral value."""
        text = content.strip()
        if not text:
            return True  # Empty/whitespace
        if len(text) < 3:
            return True  # Single chars / emotes
        if _BOT_CMD.match(text):
            return True  # Bot commands
        if _LINK_ONLY.match(text):
            return True  # Link-only, no commentary
        if _MEDIA_ONLY.match(text):
            return True  # Media file links with no commentary
        if _PURE_EMOJI.match(text):
            return True  # Pure emoji spam
        if text.lower() in _NOISE_WORDS:
            return True  # Single-word noise
        # Repetitive character spam (e.g., "aaaaaaa", "!!!!!!")
        if len(set(text.lower().replace(' ', ''))) <= 2 and len(text) > 3:
            return True
        return False

    # 3. The Iterative Crawl (Optimized for 2 vCPUs)
    all_channels = ctx.guild.text_channels + ctx.guild.voice_channels + list(ctx.guild.threads)

    # Check for force flag
    do_force = force and force.lower() in ('force', 'full', '-f')

    # Check for existing checkpoint - if found, do incremental scan (unless force)
    user_id_str = str(ctx.author.id)
    last_checkpoint = None if do_force else await get_scrape_checkpoint(bot.db, user_id_str, "global")

    if last_checkpoint:
        # Get current stored message count to calculate "new" messages
        async with bot.db.execute(
            "SELECT COUNT(*) FROM interaction_history WHERE user_id = ? AND guild_id = ?",
            (user_id_str, str(ctx.guild.id))
        ) as cursor:
            row = await cursor.fetchone()
            existing_count = row[0] if row else 0

    mode_msg = "🔄 **Incremental Update**" if last_checkpoint else "🔍 **Full Scan**"
    status_msg = await resilient_call(
        ctx.send,
        f"{mode_msg}\n"
        f"{'⏩ Skipping to last checkpoint...' if last_checkpoint else '*This may take a while.*'}"
    )

    for channel in all_channels:
        if not hasattr(channel, 'history'):
            continue

        perms = channel.permissions_for(ctx.guild.me)
        if not perms.read_message_history or not perms.read_messages:
            continue

        # Get channel-specific checkpoint (skip if force)
        channel_id_str = str(channel.id)
        last_msg_id = None if do_force else await get_scrape_checkpoint(bot.db, user_id_str, channel_id_str)

        try:
            channel_start = time.time()

            # Determine starting point - use checkpoint if exists
            if last_msg_id:
                # Get the message object to use as 'after' anchor
                try:
                    anchor_msg = await channel.fetch_message(int(last_msg_id))
                    message_iter = channel.history(limit=None, oldest_first=True, after=anchor_msg)
                except Exception:
                    # Any error (NotFound, Forbidden, HTTPException, rate limit) - do full scan
                    message_iter = channel.history(limit=None, oldest_first=True)
            else:
                message_iter = channel.history(limit=None, oldest_first=True)

            last_processed_id = last_msg_id  # Track last message processed in this channel

            async for message in message_iter:
                total_scanned += 1

                # Per-channel timeout: 300s max for large channels
                if time.time() - channel_start > 300:
                    log.warning("Channel %s exceeded 300s timeout. Moving on.", channel.name)
                    break

                # HEARTBEAT: Update UI every 1000 messages (non-critical, wrapped in try/except)
                if total_scanned % 1000 == 0:
                    try:
                        await status_msg.edit(
                            content=f"🔄 **Incremental Mapping...**\n"
                                    f"Messages scanned: `{total_scanned:,}`\n"
                                    f"Interactions found: `{total_mapped:,}`"
                        )
                    except Exception:
                        pass  # Non-critical UI update — continue scraping

                # Update checkpoint as we go
                last_processed_id = str(message.id)

                if message.author.id == ctx.author.id:
                    # Skip messages with no behavioral signal
                    if is_noise(message.content):
                        total_filtered += 1
                        continue

                    reply_id = str(message.reference.message_id) if message.reference else None
                    batch_rows.append((
                        str(message.id),
                        str(message.author.id),
                        str(ctx.guild.id),
                        message.content,
                        reply_id,
                        message.created_at.isoformat()
                    ))
                    total_mapped += 1

                    # Progress update: every 5000 messages scanned (outside batch logic)
                    if total_scanned % 5000 == 0:
                        try:
                            await status_msg.edit(
                                content=f"🔄 **Incremental Mapping...**\n"
                                        f"Messages scanned: `{total_scanned:,}`\n"
                                        f"Interactions found: `{total_mapped:,}`"
                            )
                        except Exception:
                            pass  # Non-critical UI update — continue scraping

                    if len(batch_rows) >= 200:
                        await bot.db.executemany(
                            "INSERT OR IGNORE INTO interaction_history "
                            "(message_id, user_id, guild_id, content, reply_to_id, timestamp) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            batch_rows
                        )
                        await bot.db.commit()
                        batch_rows.clear()
                        await asyncio.sleep(0.5)

        except discord.Forbidden:
            continue  # Silently skip hidden admin channels
        except Exception as e:
            log.warning("Error on channel %s: %s. Skipping.", channel.name, e)
            continue  # Skip this channel, try the next one
        finally:
            # Always update checkpoint after processing a channel (even on error)
            if last_processed_id and last_processed_id != last_msg_id:
                # Only update if we actually processed new messages
                await set_scrape_checkpoint(bot.db, user_id_str, channel_id_str, last_processed_id, "")

    # 4. Final Cleanup Commit (Catching the remainders)
    if batch_rows:
        await bot.db.executemany(
            "INSERT OR IGNORE INTO interaction_history "
            "(message_id, user_id, guild_id, content, reply_to_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            batch_rows
        )
        await bot.db.commit()

    # 5. Completion UI with Interactive Buttons
    class AnalysisView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=None)

        @discord.ui.button(label="🧬 Generate Dossier", style=discord.ButtonStyle.green)
        async def dossier(self, interaction: discord.Interaction, button: discord.ui.Button):
            if interaction.user.id != ctx.author.id: return
            await interaction.response.send_message("🧬 **Synthesis Initialized.** Check your DMs shortly.", ephemeral=True)
            await ctx.invoke(bot.get_command('generate_dossier'))

        @discord.ui.button(label="📡 Quick Behavior Scan", style=discord.ButtonStyle.secondary)
        async def scan(self, interaction: discord.Interaction, button: discord.ui.Button):
            if interaction.user.id != ctx.author.id: return
            await interaction.response.send_message("📡 **Scan Initialized.** Check your DMs shortly.", ephemeral=True)
            await ctx.invoke(bot.get_command('behavior_scan'))

    embed = discord.Embed(
        title="✅ Social Web Mapped",
        description=(
            f"Successfully extracted and secured **{total_mapped:,}** interactions for {ctx.author.mention}.\n"
            f"Noise filtered: `{total_filtered:,}` messages (bot cmds, emoji, links, filler)"
        ),
        color=discord.Color.brand_green()
    )
    embed.set_footer(text="RESTRICTED ACCESS | FOR FORENSIC USE ONLY")

    await resilient_call(status_msg.edit, content=None, embed=embed, view=AnalysisView())

    # Set cooldown (even on partial runs to prevent abuse)
    await bot.db.execute(
        "INSERT OR REPLACE INTO cooldowns (user_id, last_map_time) VALUES (?, ?)",
        (user_id, current_time)
    )
    await bot.db.commit()


# =============================================================================
# DFS-BASED THREAD ANALYSIS COMMAND
# =============================================================================
@bot.command(name='analyze_threads')
async def analyze_threads(ctx, target_id: str = None):
    """
    Analyzes conversation threads using DFS to map reply chains.
    Takes a target user (or defaults to self).
    Returns: participants, total messages, avg thread length, topics discussed.
    """
    # Privacy check
    if not is_opted_in(ctx.author):
        return await ctx.send("🔒 **Access Denied:** You must have the `PsycheOptIn` role to analyze your threads.")

    if not ctx.guild:
        return await ctx.send("⚠️ This command must be run inside a server.")

    # Resolve target ID
    if target_id is not None:
        clean = target_id.strip('<@!> ')
        try:
            resolved_id = str(int(clean))
        except ValueError:
            if ctx.message.mentions:
                resolved_id = str(ctx.message.mentions[0].id)
            else:
                return await ctx.send("⚠️ **Invalid Target:** Provide a user ID (e.g. `1234567890123456`) or a @mention.")
    else:
        resolved_id = str(ctx.author.id)

    # Check if target is opted in (unless it's self)
    if resolved_id != str(ctx.author.id):
        try:
            target_member = await ctx.guild.fetch_member(int(resolved_id))
            if not is_opted_in(target_member):
                return await ctx.send(f"🔒 **Access Denied:** Target user must have `PsycheOptIn` role.")
        except Exception:
            pass

    # Status message
    status_msg = await ctx.send(f"🔍 **Analyzing conversation threads for user {resolved_id[:8]}...**")

    try:
        # Get top 5 threads for the user
        top_threads = await find_top_threads_for_user(resolved_id, str(ctx.guild.id), top_n=5)

        if not top_threads:
            return await status_msg.edit(content="📭 **No conversation threads found** for this user.")

        # Build response embed
        embed = discord.Embed(
            title=f"🧵 Thread Analysis for User {resolved_id[:8]}...",
            description=f"Top {len(top_threads)} most active conversation threads:",
            color=discord.Color.purple()
        )

        for i, thread in enumerate(top_threads, 1):
            total_msgs = thread.get("total_messages", 0)
            participants = thread.get("participant_count", 0)
            topics = thread.get("topics", [])

            # Format topics
            topic_str = ", ".join(topics[:3]) if topics else "General"

            field_value = (
                f"**Messages:** {total_msgs}\n"
                f"**Participants:** {participants}\n"
                f"**Topics:** {topic_str}\n"
                f"**Root ID:** `{thread.get('root_message_id', 'N/A')[:12]}...`"
            )

            embed.add_field(name=f"Thread #{i} ({total_msgs} messages)", value=field_value, inline=False)

        embed = apply_disclaimer(embed)
        await status_msg.edit(content=None, embed=embed)

    except Exception as e:
        log.error(f"Thread analysis error: {e}")
        await status_msg.edit(content=f"⚠️ **Error analyzing threads:** {str(e)}")


# =============================================================================
# --- THE SCRUB PROTOCOL (Data Rights) ---
@bot.command(name='purge_my_data')
async def purge_my_data(ctx: commands.Context):
    """Allows the user to instantly delete their total history from the bot's DB."""
    await purge_user_data(str(ctx.author.id))
    await ctx.send("🗑️ **Scrub Protocol Executed:** Your interaction history has been permanently deleted from the database.")

# =============================================================================
# 9. PHASE 4: BEHAVIOR SCAN (Quick AI Snapshot)
# =============================================================================

@bot.command(name='behavior_scan')
async def behavior_scan(ctx):
    """Analyzes the last 500 messages for an immediate behavioral snapshot."""
    if not is_opted_in(ctx.author):
        return await ctx.send("🔒 You must have the `PsycheOptIn` role.")

    # Cooldown Check (1h, bypassed for Owner/Admins)
    user_id = str(ctx.author.id)
    current_time = time.time()
    is_vip = is_owner(ctx.author.id) or ctx.author.guild_permissions.administrator

    if not is_vip:
        async with bot.db.execute("SELECT last_scan_time FROM cooldowns WHERE user_id = ?", (user_id,)) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                elapsed = current_time - row[0]
                if elapsed < 3600:  # 1 hour
                    mins_left = round((3600 - elapsed) / 60, 0)
                    return await ctx.send(f"⏳ **Cooldown Active:** Please wait {mins_left} minutes before scanning again.")

    status_msg = await ctx.send("🧠 **Initiating Behavior Scan...** Accessing recent interaction matrix.")

    user_id_str = str(ctx.author.id)

    # 1. Fetch user's messages AND messages that reply TO user (context from others)
    guild_id_str = str(ctx.guild.id)
    async with bot.db.execute(
        "SELECT content, timestamp, reply_to_id, user_id FROM interaction_history WHERE user_id = ? AND guild_id = ? ORDER BY timestamp DESC LIMIT 500",
        (user_id_str, guild_id_str)
    ) as cursor:
        rows = await cursor.fetchall()

    # Also fetch messages where user was replied to (context from others - equal to user message count)
    async with bot.db.execute(
        "SELECT content, timestamp, reply_to_id, user_id FROM interaction_history WHERE reply_to_id = ? AND guild_id = ? ORDER BY timestamp DESC LIMIT ?",
        (user_id_str, guild_id_str, len(rows))
    ) as cursor:
        reply_context_rows = await cursor.fetchall()

    if len(rows) < 50:
        return await status_msg.edit(content="⚠️ **Insufficient Data:** I need at least 50 interactions to form a baseline.")

    # Build reply_map for reply context (include both user's messages and replies TO user)
    all_rows = rows + reply_context_rows
    reply_map = {str(row[2]): row[3] for row in all_rows if row[2]}
    # Sort by timestamp for proper chronological order
    all_rows_sorted = sorted(all_rows, key=lambda r: r[1])
    transcript = format_transcript(all_rows_sorted, guild=ctx.guild, reply_map=reply_map)

    # Estimate full transcript tokens for cost tracking
    full_transcript_tokens = estimate_tokens(transcript)

    # Extract structured behavioral data (2-3K tokens vs 100K+ for full transcript)
    await status_msg.edit(content="🧠 **Extracting behavioral patterns...**")
    message_texts = [row[0] for row in all_rows_sorted]
    behavioral_data = await extract_behavioral_data(message_texts, batch_size=50)
    analysis_text = behavioral_data.get("analysis", "No analysis generated.")

    # Get Sentino personality scores (pre-analysis context)
    sentino_scores = await get_sentino_scores(transcript[:3000])  # Limit text for API
    sentino_context = f"\n\n[REALITY CHECK: Sentino personality scores: {sentino_scores}]" if sentino_scores else ""

    recent_sample = [row[0][:150] for row in all_rows_sorted[-20:]]  # Last 20 messages
    sample_text = "\n".join(f"> {msg}" for msg in recent_sample)

    # 2. Async AI Generation using analysis text
    prompt = (
        "You are an elite forensic psychologist. Based on the following behavioral analysis of the user's messages, provide a concise, 3-paragraph summary covering:\n"
        "1. Current emotional tone. 2. Primary communication style. 3. Social role in the server.\n"
        "Focus on analyzing the conversational style and linguistic patterns observed rather than just reading off scores.\n\n"
        f"Total Messages Analyzed: {len(all_rows_sorted)}\n\n"
        f"{sentino_context}\n\n"
        f"BEHAVIORAL ANALYSIS:\n{analysis_text}\n\n"
        f"RECENT MESSAGES SAMPLE:\n{sample_text}"
    )

    try:
        result_text = await generate_with_fallback(
            model=SCAN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            )
        )

        embed = discord.Embed(
            title="🔍 Behavioral Snapshot",
            description=result_text,
            color=discord.Color.teal()
        )
        embed = apply_disclaimer(embed)

        await ctx.author.send(embed=embed)
        await status_msg.edit(content="✅ **Scan Complete.** The results have been sent to your DMs.")

        # Set cooldown on success
        await bot.db.execute(
            "INSERT OR REPLACE INTO cooldowns (user_id, last_scan_time) VALUES (?, ?)",
            (user_id, current_time)
        )
        await bot.db.commit()

    except discord.Forbidden:
        await status_msg.edit(content="❌ I couldn't DM you. Please enable DMs for this server.")
    except errors.APIError as e:
        if "429" in str(e) or "ResourceExhausted" in str(e):
            await status_msg.edit(content="⚠️ **Rate Limit Hit:** The AI engine is currently overloaded. Please try again in a few moments.")
        else:
            await status_msg.edit(content=f"⚠️ **AI Engine API Error:** {str(e)}")
    except Exception as e:
        await status_msg.edit(content=f"⚠️ **AI Engine Error:** {str(e)}")

# =============================================================================
# 10. PHASE 5: DEEP SYNTHESIS DOSSIER
# =============================================================================

@bot.command(name='generate_dossier')
async def generate_dossier(ctx):
    """The Flagship Command: Synthesizes ALL history + Raw Assessment Data."""
    if not is_opted_in(ctx.author):
        return await ctx.send("🔒 You must have the `PsycheOptIn` role.")

    user_id = str(ctx.author.id)
    current_time = time.time()

    # 1. Strict 7-Day Cooldown Check (Bypassed for Owner/Admins)
    is_vip = is_owner(ctx.author.id) or ctx.author.guild_permissions.administrator
    
    async with bot.db.execute("SELECT last_dossier_time FROM cooldowns WHERE user_id = ?", (user_id,)) as cursor:
        row = await cursor.fetchone()
        if row and row[0] is not None and not is_vip:
            elapsed = current_time - row[0]
            if elapsed < 604800:  # 7 days in seconds
                days_left = round((604800 - elapsed) / 86400, 1)
                return await ctx.send(f"⏳ **Cooldown Active:** Deep Synthesis requires massive computation. Please wait {days_left} days.")

    status_msg = await ctx.send("🧬 **Initiating Deep Synthesis (2-Phase)...** Phase 1: Compress interaction data. Phase 2: Generate psychological dossier. *This may take a minute.*")

    # Pre-warm the DM channel BEFORE the long Gemini call
    # This keeps the connection alive and reduces stale-session failures
    try:
        dm_channel = await ctx.author.create_dm()
    except Exception:
        dm_channel = None

    # 2. Fetch History (capped at 40k for safe payload - OpenCode/OpenRouter now have 262K context models)
    guild_id_str = str(ctx.guild.id)
    async with bot.db.execute(
        "SELECT content, timestamp, reply_to_id, user_id FROM interaction_history WHERE user_id = ? AND guild_id = ? ORDER BY timestamp DESC LIMIT 40000",
        (user_id, guild_id_str)
    ) as cursor:
        chat_rows = await cursor.fetchall()

    # Also fetch messages that reply TO user (context from others - equal to user message count)
    async with bot.db.execute(
        "SELECT content, timestamp, reply_to_id, user_id FROM interaction_history WHERE reply_to_id = ? AND guild_id = ? ORDER BY timestamp DESC LIMIT ?",
        (user_id, guild_id_str, len(chat_rows))
    ) as cursor:
        reply_context_rows = await cursor.fetchall()

    if len(chat_rows) < 100:
        return await status_msg.edit(content="⚠️ **Insufficient Data:** Run `!map_interactions` to build your social web first.")

    # Combine user's messages with context from others
    all_rows = chat_rows + reply_context_rows
    all_rows_sorted = sorted(all_rows, key=lambda r: r[1])
    reply_map = {str(row[2]): row[3] for row in all_rows if row[2]}

    # 3. Fetch Raw Quiz Data and Map to Actual Questions
    async with bot.db.execute("SELECT quiz_type, raw_answers FROM quiz_results WHERE user_id = ?", (user_id,)) as cursor:
        quiz_rows = await cursor.fetchall()

    try:
        with open(QUESTIONS_JSON, 'r') as f:
            all_questions_data = json.load(f)
    except Exception:
        all_questions_data = {}

    quiz_context = ""
    for q_type, raw_json in quiz_rows:
        try:
            answers = json.loads(raw_json)
            quiz_data = all_questions_data.get(q_type, {})
            questions = quiz_data.get("questions", [])
            
            quiz_context += f"--- {q_type.upper()} ASSESSMENT ---\n"
            if questions:
                for idx, ans in enumerate(answers):
                    if idx < len(questions):
                        q_text = questions[idx].get("q", "Unknown Question")
                        quiz_context += f"Q: {q_text} | Answered: {ans}\n"
            else:
                quiz_context += f"Raw Likert Scale Array: {answers}\n"
            quiz_context += "\n"
        except Exception:
            continue

    if not quiz_context:
        quiz_context = "No psychometric assessments completed. Rely strictly on behavioral data."

    # Get Sentino personality scores (pre-analysis context)
    combined_text = " ".join([row[0] for row in chat_rows[:1000]])
    sentino_scores = await get_sentino_scores(combined_text[:3000])
    sentino_context = f"\n[SENTINO REALITY CHECK: {sentino_scores}]" if sentino_scores else ""

    # 4. Extract structured behavioral data (replaces summarization pass - massive token savings)
    status_msg_edit = await status_msg.edit(content="📝 **Phase 1: Extracting behavioral patterns...**")

    transcript = format_transcript(all_rows_sorted, guild=ctx.guild, reply_map=reply_map)
    full_transcript_tokens = estimate_tokens(transcript[:50000])  # Estimate from truncated transcript

    # Extract behavioral data using Groq (2-3K tokens vs 40K+ for full transcript)
    message_texts = [row[0] for row in all_rows_sorted]
    behavioral_data = await extract_behavioral_data(message_texts, batch_size=100)
    analysis_text = behavioral_data.get("analysis", "No analysis generated.")

    # Update status for Phase 2
    status_msg = await status_msg.edit(content="🧬 **Phase 2: Generating psychological dossier...**")

    # 5. The Master Prompt
    prompt = (
        "Generate a 'Deep Psychological Synthesis Dossier' for this user.\n"
        "You are an elite behavioral profiler. You have access to their raw psychometric test results and their behavioral analysis data.\n\n"
        "GOALS:\n"
        "1. Identify 'Cognitive Dissonance'—where does their actual chat behavior contradict their self-reported test answers?\n"
        "2. Analyze 'Linguistic Variance'—how does their tone shift when addressing different people or topics?\n"
        "3. Provide a full-fledged analysis combining both datasets to determine their true psychological state.\n"
        "4. Format the response with the following REQUIRED sections:\n"
        "   - **The Public Mask** (How they present themselves to others)\n"
        "   - **The Private Reality** (What the psychometric data reveals vs. behavior)\n"
        "   - **The Social Archetype** (Their core role and impact on the server)\n"
        "Make it profound, clinical, and highly detailed (1000+ words).\n\n"
        f"Total Messages Analyzed: {len(all_rows_sorted)}\n\n"
        f"=== PSYCHOMETRIC TEST DATA ===\n{quiz_context}{sentino_context}\n"
        f"=== BEHAVIORAL ANALYSIS ===\n{analysis_text}"
    )

    try:
        result_text = await asyncio.wait_for(
            generate_with_fallback(
                model=DOSSIER_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            ),
            timeout=120.0
        )

        # Try to send as multiple embeds (chunked response)
        delivered = False
        try:
            await send_chunked_response(
                ctx.author,
                result_text,
                "🧬 Deep Synthesis Dossier",
                discord.Color.dark_purple()
            )
            delivered = True
        except Exception as e:
            log.warning(f"Chunked response failed, trying single embed: {e}")
            # Fallback to single embed
            embed = discord.Embed(
                title="🧬 Deep Synthesis Dossier",
                description=result_text[:4000],
                color=discord.Color.dark_purple()
            )
            embed = apply_disclaimer(embed)

            for dm_attempt in range(3):
                try:
                    await ctx.author.send(embed=embed)
                    delivered = True
                    break
                except (aiohttp.ClientConnectorError, ConnectionResetError, OSError, discord.DiscordServerError) as e:
                    if dm_attempt == 2:
                        break
                    log.warning(f"DM delivery attempt {dm_attempt + 1} failed: {e}. Retrying in 5s...")
                    await asyncio.sleep(5)

        if not delivered:
            # Fallback: send in the channel instead
            log.warning("DM delivery failed after attempts. Falling back to channel.")
            embed = discord.Embed(
                title="🧬 Deep Synthesis Dossier",
                description=result_text[:4000],
                color=discord.Color.dark_purple()
            )
            embed = apply_disclaimer(embed)
            await ctx.send(embed=embed)
        
        # Set the Cooldown ONLY on success
        await bot.db.execute(
            "INSERT OR REPLACE INTO cooldowns (user_id, last_dossier_time) VALUES (?, ?)",
            (user_id, current_time)
        )
        await bot.db.commit()

        if delivered:
            await status_msg.edit(content="✅ **Synthesis Complete.** The secure dossier has been delivered to your DMs.")
        else:
            await status_msg.edit(content="✅ **Synthesis Complete.** DM delivery failed — results posted above.")

    except discord.Forbidden:
        await status_msg.edit(content="❌ **Privacy Error:** I couldn't deliver the dossier because your DMs are closed. Please enable DMs for this server and try again.")
    except asyncio.TimeoutError:
        await status_msg.edit(content="⚠️ **AI Engine Timeout:** The data volume was too large for the current model allocation. Try again later.")
    except errors.APIError as e:
        if "429" in str(e) or "ResourceExhausted" in str(e):
            await status_msg.edit(content="⚠️ **Rate Limit Hit:** The Deep Synthesis engine is overloaded. Please try again later.")
        else:
            await status_msg.edit(content=f"⚠️ **Synthesis API Error:** {str(e)}")
    except Exception as e:
        await status_msg.edit(content=f"⚠️ **Synthesis Error:** {str(e)}")

# DEPRECATED QUIZ SYSTEM REMOVED (Replaced by AssessmentView and !assessment)

# =============================================================================
# 10. SYSTEM COMMANDS
# =============================================================================

@bot.command(name='ping')
async def ping(ctx: commands.Context):
    """Connection health check."""
    latency = round(bot.latency * 1000)
    db_status = "Connected (/data/psyche.db)" if bot.db else "Disconnected"
    scan_m = os.getenv('SCAN_MODEL', 'gemini-3.1-flash-lite')
    engine_str = "Gemini 3.1 Flash-Lite" if "flash-lite" in scan_m.lower() else scan_m

    msg = (f"Pong! 🏓\n"
           f"Latency: {latency}ms\n"
           f"Database: {db_status}\n"
           f"Engine: {engine_str} Active")
    await resilient_call(ctx.send, msg)

# =============================================================================
# 11. PHASE 6: THE CREATOR'S SKELETON KEY
# =============================================================================

@bot.command(name="help")
async def help_command(ctx):
    embed = discord.Embed(
        title="🧠 Psyche v2 | Forensic Operational Manual",
        description="Privacy-first behavioral analysis. Your data stays yours.",
        color=0x2f3136
    )

    # ─────────────────────────────────────────────────────────
    # ASSESSMENTS (Psychometric Tests)
    # ─────────────────────────────────────────────────────────
    embed.add_field(
        name="🔬 Assessment",
        value="`!assessment [ocean|big5|mbti|enneagram]`\n"
               "Start a psychometric test delivered via DM. Takes ~5 minutes.\n"
               "• **Ocean/Big5** — Personality traits (openness, conscientiousness, etc.)\n"
               "• **MBTI** — Cognitive preferences and communication style\n"
               "• **Enneagram** — Core motivations and emotional patterns\n"
               "⏳ 7-day cooldown between assessments",
        inline=False
    )

    embed.add_field(
        name="🔁 Resume",
        value="`!assessment_resume`\n"
               "Continue a paused assessment from where you left off.\n"
               "⏸️ Automatically saves your progress",
        inline=False
    )

    # ─────────────────────────────────────────────────────────
    # DATA & ANALYSIS COMMANDS
    # ─────────────────────────────────────────────────────────
    embed.add_field(
        name="📡 Map Interactions",
        value="`!map_interactions` — Scan & save your messages\n"
               "`!map_interactions force` — Full rescan from beginning\n"
               "🔄 Incremental mode: Only scans NEW messages since last scan\n"
               "⚡ Up to 50x faster on repeat runs\n"
               "⏳ 24-hour cooldown (bypassed for admins)",
        inline=False
    )

    embed.add_field(
        name="🧵 Analyze Threads",
        value="`!analyze_threads` or `!analyze_threads @user`\n"
               "Uses DFS to map your actual conversation reply chains.\n"
               "📊 Shows: top threads, participants, topics discussed.\n"
               "🔬 Reveals who you actually talk to, not just who's online.",
        inline=False
    )

    embed.add_field(
        name="🔍 Behavior Scan",
        value="`!behavior_scan`\n"
               "Quick AI snapshot of your recent messages.\n"
               "⚡ Uses lightweight extraction for instant results.\n"
               "📋 Shows: emotional tone, communication style, social role.\n"
               "⏳ 1-hour cooldown (bypassed for admins)",
        inline=False
    )

    embed.add_field(
        name="🧬 Deep Synthesis",
        value="`!generate_dossier`\n"
               "Full psychological profile (1000+ words).\n"
               "🔬 Combines: chat history + psychometric data + sentiment analysis.\n"
               "📊 Identifies: cognitive dissonance, linguistic variance, social archetype.\n"
               "⏳ 7-day cooldown",
        inline=False
    )

    # ─────────────────────────────────────────────────────────
    # UTILITY & SECURITY
    # ─────────────────────────────────────────────────────────
    embed.add_field(
        name="🏓 Ping",
        value="`!ping`\n"
               "Check bot latency and system status.\n"
               "📡 Shows: response time, database connection, active model.",
        inline=False
    )

    embed.add_field(
        name="🛡️ Purge My Data",
        value="`!purge_my_data`\n"
               "Permanently delete ALL your stored data.\n"
               "⚠️ This cannot be undone — your messages, quiz results, and profiles are wiped instantly.",
        inline=False
    )

    # ─────────────────────────────────────────────────────────
    # REQUIREMENTS
    # ─────────────────────────────────────────────────────────
    embed.add_field(
        name="📋 Requirements",
        value="**PsycheOptIn Role** — Required for all analysis commands.\n"
               "Get the role from your server admin to unlock behavioral features.\n"
               "Remove the role anytime — your data auto-purges.",
        inline=False
    )

    # ─────────────────────────────────────────────────────────
    # DISCLAIMER
    # ─────────────────────────────────────────────────────────
    await ctx.send(embed=apply_disclaimer(embed))

if __name__ == '__main__':
    bot.run(DISCORD_TOKEN, reconnect=True, log_handler=None)
