# Psyche v2 — Technical Documentation

## Overview

Psyche v2 is a privacy-first Discord behavioral analysis bot that combines psychometric testing with AI-powered chat analysis to generate psychological profiles. It operates on a strict "data minimization" principle — users opt-in, own their data, and can purge it anytime.

---

## Architecture

### Tech Stack

| Component | Technology |
| --- | --- |
| Bot Framework | discord.py (async) |
| Database | SQLite (aiosqlite) |
| Primary AI | Google Gemini |
| Fast AI | Groq (llama-3.1-8b-instant) |
| Fallback AI | OpenRouter, OpenCode |
| Personality API | Sentino (Big5 scoring) |

### Data Flow

```
User Message → Opt-In Check → SQLite DB
                    ↓
Analysis Request → Fetch from DB → Groq Extraction → Structured JSON
                                        ↓
                              Edge Case Detection
                                /              \
                         Fast Path          Full LLM
                         (no cost)         (deep analysis)
                              ↓                   ↓
                         Results via DM    Results via DM
```

---

## Core Components

### 1. The Opt-In System

```python
def is_opted_in(member) -> bool:
    # Case-insensitive check for 'psycheoptin' role
    return any(role.name.lower() == "psycheoptin" for role in member.roles)
```

**How it works:**
- Requires the `PsycheOptIn` role on Discord server
- Every analysis command checks this before proceeding
- If role is removed → `on_member_update` triggers automatic data purge
- Privacy-first: no data stored for non-opted users

---

### 2. Assessment System (Psychometric Tests)

**Commands:** `!assessment [ocean|mbti|enneagram]`

**Process:**
1. User selects test type → questions loaded from `questions.json`
2. `AssessmentView` creates interactive DM with buttons (1-5 Likert or A/B)
3. Progress saved to `quiz_sessions` table after each answer
4. On completion → raw answers stored in `quiz_results` table
5. Future analysis combines quiz data with chat patterns

**Database Schema:**
```sql
quiz_sessions (
    user_id TEXT PRIMARY KEY,
    quiz_type TEXT,
    progress INTEGER,
    answers TEXT
)

quiz_results (
    user_id TEXT,
    quiz_type TEXT,
    raw_answers TEXT,
    timestamp TEXT
)
```

---

### 3. Message Mapping System

**Command:** `!map_interactions`

**Process:**
1. Iterates through ALL server text channels + threads
2. Filters noise (bot commands, emoji spam, single words)
3. Stores in `interaction_history` table with reply context
4. Batch commits every 200 messages for performance

**Noise Filter:**
- Bot commands (`!cmd`, `?cmd`)
- Link-only messages
- Pure emoji messages
- Single-word filler (ok, lol, yeah)
- Repetitive characters (aaaaaa)

**Database Schema:**
```sql
interaction_history (
    message_id TEXT PRIMARY KEY,
    user_id TEXT,
    content TEXT,
    reply_to_id TEXT,
    timestamp TEXT
)
```

---

### 4. Behavioral Data Extraction (Groq)

**Function:** `extract_behavioral_data(messages)`

This is the key innovation — instead of sending 25K+ tokens to expensive LLMs, we use fast Groq extraction to get structured signals:

**Input:** Raw messages (batched, 100 per batch)

**Output:**
```json
{
  "sentiment": 0.35,
  "topics": ["gaming", "music", "philosophy"],
  "emotional_arc": [
    {"batch": 1, "sentiment": 0.2, "intensity": 0.6, "emotions": ["happy", "neutral"]},
    {"batch": 2, "sentiment": 0.5, "intensity": 0.4, "emotions": ["excited"]}
  ],
  "message_batches": 5
}
```

**Why this matters:**
- Tokens: ~25,000 → ~2,000 (90% reduction)
- Cost: ~$0.03 → ~$0.002 (15x cheaper)
- Speed: ~3 seconds vs ~30 seconds

---

### 5. Edge Case Detection

**Function:** `should_use_llm_for_edge_case(behavioral_data)`

Decides whether to invoke expensive full LLM analysis:

**Triggers full LLM when:**
- Sentiment ambiguous (-0.2 to 0.2)
- High intensity (> 0.8)
- Sudden mood swings detected
- Ambiguous topics detected
- Conflicting emotions (both positive AND negative)

**Fast Path:** No edge cases → build structured profile from Groq data → send via DM (no LLM)

**Deep Path:** Edge cases → multi-batch summarization → full psychological analysis

---

### 6. Thread Analysis (DFS)

**Command:** `!analyze_threads`

Maps actual conversation reply chains instead of flat transcripts:

**Process:**
1. Build reply graph: `{message_id: [reply_ids]}`
2. DFS traversal from user's messages
3. Find top 5 most active threads
4. Extract: participants, total messages, topics

**Why it matters:** Shows WHO user actually talks to, not just who's online. Reveals conversation depth with each person.

---

### 7. Analysis Commands

| Command | Input | Processing | Output |
| --- | --- | --- | --- |
| `!behavior_scan` | Last 500 messages | Groq extraction only | Quick snapshot |
| `!analyze_threads` | All user messages | DFS graph analysis | Thread analysis |
| `!generate_dossier` | All messages + quiz data | Groq + LLM (if edge) | Full profile (1000+ words) |

---

## Database Schema

```sql
-- Message history (opt-in users only)
messages (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    guild_id TEXT,
    content TEXT,
    timestamp DATETIME
)
CREATE INDEX idx_messages_user_guild ON messages(user_id, guild_id)

-- Interaction history (includes reply context)
interaction_history (
    message_id TEXT PRIMARY KEY,
    user_id TEXT,
    content TEXT,
    reply_to_id TEXT,
    timestamp TEXT
)

-- Quiz data
quiz_sessions (...)
quiz_results (...)

-- Cooldowns (prevents abuse)
cooldowns (
    user_id TEXT PRIMARY KEY,
    last_dossier_time REAL,
    last_map_time REAL,
    last_scan_time REAL
)

-- Sync checkpoints (scraper resume)
sync_checkpoints (
    channel_id TEXT PRIMARY KEY,
    last_message_id TEXT
)
```

---

## AI Waterfall

The bot tries providers in order until one works:

1. Groq (`llama-3.3-70b-versatile`) → 128K context
   ↓ (if no key or error)
2. OpenRouter (`inclusionai/ring-2.6-1t:free`) → 262K context
   ↓
3. OpenCode (`ring-2.6-1t-free`) → 262K context
   ↓
4. Gemini (`gemini-1.5-flash`) → final fallback

This ensures reliability — if one provider fails, it gracefully falls back.

---

## Privacy & Security

| Feature | Implementation |
| --- | --- |
| Data Ownership | User can delete anytime (`!purge_my_data`) |
| Auto-Purge | Removing role triggers data wipe |
| No Medical Claims | All output framed as "behavioral conjecture" |
| Local Storage | All data in local SQLite, not cloud |
| Channel Restriction | Commands can be limited to specific channel via `COMMAND_CHANNEL_ID` |
| DM-Only Results | Sensitive analysis delivered via DM |
| Multi-Owner Support | `OWNER_IDS` env var for multiple admins |

---

## Command Summary

| Command | Access | Description |
| --- | --- | --- |
| `!assessment` | Members | Start psychometric test (DM) |
| `!assessment_resume` | Members | Continue paused test |
| `!map_interactions` | Members | Scan and store your messages |
| `!analyze_threads` | Members | DFS thread analysis |
| `!behavior_scan` | Members | Quick 500-msg snapshot |
| `!generate_dossier` | Members | Full psychological profile |
| `!ping` | Anyone | System health check |
| `!purge_my_data` | Members | Delete all your data |
| `!help` | Anyone | Show this help |

*All analysis commands require PsycheOptIn role.*

---

## Cost Optimization

**Before:**
- Full transcript → 25K tokens → LLM → $0.03+

**After:**
- Groq extraction → 2K tokens → $0.0001
- Structured JSON → 2K tokens → LLM → $0.002
- Total: ~15x cheaper

**Cost tracking logs savings:**
```
[COST] behavior_scan: 25000 -> 2000 tokens (saved ~23000)
[COST] generate_dossier: 125000 -> 3000 tokens (saved ~122000)
```

---

## Summary

Psyche v2 combines psychometric testing with AI behavioral analysis while maintaining strict privacy controls. The architecture uses Groq for fast extraction and only invokes expensive LLMs for complex edge cases, making it viable for low-resource hosting. All data stays local, users control their data, and the system gracefully handles provider failures through the AI waterfall.
