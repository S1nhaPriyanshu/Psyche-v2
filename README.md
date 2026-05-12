---
title: Psyche's Brain
emoji: 🧠
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 Psyche — AI Behavioral Psychologist

Psyche is a **privacy-first, high-reasoning Discord bot** that uses Google Gemini 3.1 Pro to perform deep behavioral and psychological analysis of opted-in users based on their communication patterns and self-assessment quiz results.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Hosting** | Hugging Face Spaces (Docker) |
| **Intelligence** | Google Gemini 3.1 Pro (`gemini-2.5-pro-preview-05-06`) |
| **Database** | SQLite (persisted via HF Storage Bucket at `/data/psyche.db`) |
| **Bot Framework** | discord.py 2.x |
| **Heartbeat** | Native aiohttp web server on port 7860 |

---

## 🛡️ Privacy Architecture

- **PsycheOptIn Role Gating:** The bot is completely blind to any user who does not have the `PsycheOptIn` role. No messages are logged, no analysis is run.
- **Scrub Protocol:** If the bot is removed from a server, all data associated with that Guild ID is immediately and permanently deleted.
- **DM-First Delivery:** All analysis results and quiz sessions are delivered via private DM.
- **Creator Lockdown:** The `!query` command only functions for the hardcoded `OWNER_ID` and only inside a Private DM.

---

## 🎭 Commands

### User Commands
| Command | Description |
|---|---|
| `!help` | Displays this interactive guide as a rich embed |
| `!ping` | Returns bot latency and system status |
| `!take_test` | Initiates an MBTI or OCEAN personality quiz via DM |
| `!quiz resume` | Resumes a paused quiz session |
| `!analyze_me` | Runs a behavioral analysis on your chat history |
| `!ultimate_analysis` | Synthesizes quiz results + chat history into a master profile |

### Creator-Only (DM only)
| Command | Description |
|---|---|
| `!query <user_id> <question>` | Asks the AI a specific question about a user's data |

---

## ⚙️ Required Secrets (HF Spaces Settings → Variables and Secrets)

| Secret Name | Description |
|---|---|
| `DISCORD_TOKEN` | Your bot token from the Discord Developer Portal |
| `GEMINI_API_KEY` | Your Google AI Studio API key |
| `OWNER_ID` | Your Discord User ID (numeric string) |
| `GEMINI_MODEL` | *(Optional)* Model override. Defaults to `gemini-2.5-pro-preview-05-06` |

---

*Powered by Gemini 3.1 Pro · Built with privacy-first architecture.*
