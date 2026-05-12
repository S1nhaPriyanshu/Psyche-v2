# =============================================================================
# Psyche v2 — Core Bot
# A privacy-first, high-reasoning Discord behavioral analysis bot.
# Powered by Google Gemini 3.1 Pro.
# =============================================================================

import os
import certifi

# --- 1. SSL CERTIFICATE PATCH (CRITICAL ORDER) ---
# MUST be executed BEFORE any network-reliant library (discord, aiohttp, google) 
# is even imported to ensure the environment variables are locked in.
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import asyncio
import aiosqlite         # Phase 2: Async SQLite
import logging
import signal
import json
from datetime import datetime

# Now safe to import network-reliant libraries
import discord
from discord.ext import commands
from discord import ui
from aiohttp import web
from dotenv import load_dotenv
import google.generativeai as genai
from questions import ASSESSMENTS

# =============================================================================
# 1. CONFIGURATION & ENVIRONMENT
# =============================================================================

load_dotenv()

DISCORD_TOKEN   = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY  = os.getenv('GEMINI_API_KEY')
OWNER_ID        = os.getenv('OWNER_ID')
MODEL_ID        = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro-preview-05-06')
DB_PATH         = '/data/psyche.db'

# Startup Validation
if not DISCORD_TOKEN: raise EnvironmentError("DISCORD_TOKEN missing.")
if not GEMINI_API_KEY: raise EnvironmentError("GEMINI_API_KEY missing.")

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
# 3. GOOGLE GEMINI CONFIGURATION
# =============================================================================

genai.configure(api_key=GEMINI_API_KEY, transport='rest')
model = genai.GenerativeModel(MODEL_ID)
DISCLAIMER = "\n\n---\n⚠️ *Analysis for research/entertainment purposes only.*"

# =============================================================================
# 4. PHASE 2: DATABASE & PRIVACY ARCHITECTURE
# =============================================================================

async def init_db(db: aiosqlite.Connection):
    """
    Initializes the database schema using an existing async connection.
    """
    log.info("Initializing database schema...")
    
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

    # -- 2. Quiz Results Table (Compound PK) --
    await db.execute('''
        CREATE TABLE IF NOT EXISTS quiz_results (
            user_id      TEXT NOT NULL,
            guild_id     TEXT NOT NULL,
            quiz_type    TEXT NOT NULL,
            result_summary TEXT,
            raw_answers  TEXT,
            completed_at DATETIME,
            PRIMARY KEY (user_id, guild_id, quiz_type)
        )
    ''')

    # -- 3. Quiz Sessions Table (PK: user_id) --
    await db.execute('''
        CREATE TABLE IF NOT EXISTS quiz_sessions (
            user_id          TEXT PRIMARY KEY,
            quiz_type        TEXT NOT NULL,
            current_question INTEGER DEFAULT 0,
            answers          TEXT DEFAULT '[]',
            started_at       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # -- 4. Cooldowns Table (Compound PK) --
    await db.execute('''
        CREATE TABLE IF NOT EXISTS cooldowns (
            user_id   TEXT NOT NULL,
            command   TEXT NOT NULL,
            last_used DATETIME NOT NULL,
            PRIMARY KEY (user_id, command)
        )
    ''')

    # -- 5. Interaction History Table (Single-Server Deep Scraper) --
    await db.execute('''
        CREATE TABLE IF NOT EXISTS interaction_history (
            message_id  TEXT PRIMARY KEY,
            user_id     TEXT,
            content     TEXT,
            reply_to_id TEXT,
            timestamp   TEXT
        )
    ''')

    # -- 6. Sync Checkpoints (Scraper Resume) --
    await db.execute('''
        CREATE TABLE IF NOT EXISTS sync_checkpoints (
            channel_id      TEXT PRIMARY KEY,
            last_message_id TEXT NOT NULL
        )
    ''')

    await db.commit()
    log.info("Database schema ready.")

def is_opted_in(member: discord.Member) -> bool:
    """Checks for the role named 'psycheoptin' (case-insensitive)."""
    if not isinstance(member, discord.Member): return False
    return any(role.name.lower() == "psycheoptin" for role in member.roles)

def is_owner(user_id: int) -> bool:
    """Compares against the OWNER_ID env var."""
    return str(user_id) == OWNER_ID

def apply_disclaimer(embed: discord.Embed):
    """
    Applies the mandatory clinical disclaimer to any Discord Embed.
    """
    embed.set_footer(text="⚠️ Disclaimer: AI-generated behavioral conjecture. Not a clinical assessment.")
    return embed

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

class AssessmentView(ui.View):
    """
    Singleton-message assessment UI.
    One message edits itself for every question. Likert 1-5 buttons.
    Commits to DB after every answer to survive HF restarts.
    """
    def __init__(self, user_id: str, quiz_type: str, progress: int, answers: list, dm_channel):
        super().__init__(timeout=600)  # 10 minute idle timeout
        self.user_id = user_id
        self.quiz_type = quiz_type
        self.progress = progress
        self.answers = answers
        self.dm_channel = dm_channel
        self.items_list = ASSESSMENTS[quiz_type]["items"]
        self.total = len(self.items_list)
        self.message = None  # The singleton message reference

    def build_embed(self):
        """Builds the embed for the current question."""
        q_text = self.items_list[self.progress]
        embed = discord.Embed(
            title=f"{ASSESSMENTS[self.quiz_type]['name']}",
            description=f"**Q{self.progress + 1}/{self.total}**\n\n“{q_text}”",
            color=discord.Color.from_str("#9B59B6")
        )
        embed.add_field(
            name="Scale",
            value="1️⃣ Strongly Disagree • 2️⃣ Disagree • 3️⃣ Neutral • 4️⃣ Agree • 5️⃣ Strongly Agree",
            inline=False
        )
        bar_len = 20
        filled = int((self.progress / self.total) * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = int((self.progress / self.total) * 100)
        embed.add_field(name="Progress", value=f"`[{bar}]` {pct}%", inline=False)
        apply_disclaimer(embed)
        return embed

    async def record_answer(self, value: int, interaction: discord.Interaction):
        """Core handler: saves answer, advances or finishes."""
        # Prevent other users from clicking
        if str(interaction.user.id) != self.user_id:
            return await interaction.response.send_message("This isn't your assessment.", ephemeral=True)

        self.answers.append(value)
        self.progress += 1

        # Persist to DB after every answer (survives HF restarts)
        await bot.db.execute(
            "INSERT OR REPLACE INTO quiz_sessions (user_id, quiz_type, current_question, answers) VALUES (?, ?, ?, ?)",
            (self.user_id, self.quiz_type, self.progress, json.dumps(self.answers))
        )
        await bot.db.commit()

        if self.progress >= self.total:
            await self.finish_quiz(interaction)
        else:
            await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def finish_quiz(self, interaction: discord.Interaction):
        """
        Zero-CPU Finalization:
        Moves raw answers list to quiz_results as JSON. No scoring.
        """
        # Save raw data to results
        await bot.db.execute(
            "INSERT OR REPLACE INTO quiz_results "
            "(user_id, guild_id, quiz_type, result_summary, raw_answers, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self.user_id, "0", self.quiz_type,
             f"Raw {self.total}-item response captured.",
             json.dumps(self.answers), datetime.now().isoformat())
        )
        # Wipe session
        await bot.db.execute("DELETE FROM quiz_sessions WHERE user_id = ?", (self.user_id,))
        await bot.db.commit()

        embed = discord.Embed(
            title="✅ Assessment Complete",
            description=f"**{ASSESSMENTS[self.quiz_type]['name']}**\n\n"
                        f"All {self.total} responses have been securely recorded.\n"
                        f"Use `!analyze_me` or `!ultimate_analysis` to generate your AI profile.",
            color=discord.Color.green()
        )
        apply_disclaimer(embed)
        self.stop()
        await interaction.response.edit_message(embed=embed, view=None)

    async def on_timeout(self):
        """Disable buttons on idle timeout. Progress is already saved."""
        if self.message:
            embed = discord.Embed(
                title="⏰ Session Paused",
                description=f"You were idle for too long. Your progress ({self.progress}/{self.total}) is saved.\n"
                            f"Use `!assessment_resume` to continue.",
                color=discord.Color.orange()
            )
            apply_disclaimer(embed)
            try:
                await self.message.edit(embed=embed, view=None)
            except:
                pass

    # --- Likert Buttons ---
    @ui.button(label="1", style=discord.ButtonStyle.secondary)
    async def btn_1(self, interaction: discord.Interaction, button: ui.Button):
        await self.record_answer(1, interaction)

    @ui.button(label="2", style=discord.ButtonStyle.secondary)
    async def btn_2(self, interaction: discord.Interaction, button: ui.Button):
        await self.record_answer(2, interaction)

    @ui.button(label="3", style=discord.ButtonStyle.primary)
    async def btn_3(self, interaction: discord.Interaction, button: ui.Button):
        await self.record_answer(3, interaction)

    @ui.button(label="4", style=discord.ButtonStyle.secondary)
    async def btn_4(self, interaction: discord.Interaction, button: ui.Button):
        await self.record_answer(4, interaction)

    @ui.button(label="5", style=discord.ButtonStyle.secondary)
    async def btn_5(self, interaction: discord.Interaction, button: ui.Button):
        await self.record_answer(5, interaction)

# =============================================================================
# 5. PHASE 4: QUIZ DATA & ENGINE
# =============================================================================

QUIZ_DATA = {
    "mbti": {
        "name": "MBTI (16 Personalities)",
        "instructions": "For each question, reply with **A** or **B**. Pick the one that feels more natural to you.",
        "questions": [
            {"q": "At a party, do you:", "a": "Interact with many, including strangers (E)", "b": "Interact with a few, known to you (I)", "dim": "EI"},
            {"q": "Are you more:", "a": "Realistic than speculative (S)", "b": "Speculative than realistic (N)", "dim": "SN"},
            {"q": "Is it worse to:", "a": "Have your head in the clouds (S)", "b": "Be in a rut (N)", "dim": "SN"},
            {"q": "Are you more impressed by:", "a": "Principles (T)", "b": "Emotions (F)", "dim": "TF"},
            {"q": "Are you more drawn toward the:", "a": "Convincing (T)", "b": "Touching (F)", "dim": "TF"},
            {"q": "Do you prefer to work:", "a": "To deadlines (J)", "b": "Just 'whenever' (P)", "dim": "JP"},
            {"q": "Do you tend to choose:", "a": "Rather carefully (J)", "b": "Somewhat impulsively (P)", "dim": "JP"},
            {"q": "In your social groups, are you:", "a": "The first to hear news (E)", "b": "The last to hear news (I)", "dim": "EI"},
            {"q": "Do you prefer:", "a": "Clear boundaries (S)", "b": "Possibilities (N)", "dim": "SN"},
            {"q": "Are you more:", "a": "Practical (S)", "b": "Conceptual (N)", "dim": "SN"},
            {"q": "Which is a higher compliment:", "a": "A consistent person (T)", "b": "A devoted person (F)", "dim": "TF"},
            {"q": "In making decisions, do you rely more on:", "a": "Data (T)", "b": "Inner values (F)", "dim": "TF"},
            {"q": "Are you more comfortable with:", "a": "Written plans (J)", "b": "Spontaneous options (P)", "dim": "JP"},
            {"q": "Do you prefer things to be:", "a": "Settled and decided (J)", "b": "Unsettled and open (P)", "dim": "JP"},
            {"q": "Do you consider yourself:", "a": "An outgoing person (E)", "b": "A private person (I)", "dim": "EI"},
            {"q": "Do you prefer to focus on:", "a": "What is (S)", "b": "What could be (N)", "dim": "SN"},
            {"q": "Do you value more in yourself:", "a": "Reason (T)", "b": "Compassion (F)", "dim": "TF"},
            {"q": "Is it your way to:", "a": "Make things happen (J)", "b": "Let things happen (P)", "dim": "JP"},
            {"q": "Do you prefer to:", "a": "Talk more than listen (E)", "b": "Listen more than talk (I)", "dim": "EI"},
            {"q": "Are you more comfortable with:", "a": "Concrete facts (S)", "b": "Abstract theories (N)", "dim": "SN"}
        ]
    },
    "ocean": {
        "name": "Big Five (OCEAN)",
        "instructions": "Reply with a number from **1 to 5**:\n1: Strongly Disagree\n2: Disagree\n3: Neutral\n4: Agree\n5: Strongly Agree",
        "questions": [
            {"q": "I see myself as someone who is curious about many different things.", "dim": "O"},
            {"q": "I see myself as someone who is thorough in my work.", "dim": "C"},
            {"q": "I see myself as someone who is talkative.", "dim": "E"},
            {"q": "I see myself as someone who is helpful and unselfish with others.", "dim": "A"},
            {"q": "I see myself as someone who worries a lot.", "dim": "N"},
            {"q": "I see myself as someone who has an active imagination.", "dim": "O"},
            {"q": "I see myself as someone who tends to be disorganized.", "dim": "C", "rev": True},
            {"q": "I see myself as someone who is full of energy.", "dim": "E"},
            {"q": "I see myself as someone who has a forgiving nature.", "dim": "A"},
            {"q": "I see myself as someone who is relaxed, handles stress well.", "dim": "N", "rev": True},
            {"q": "I see myself as someone who values artistic, aesthetic experiences.", "dim": "O"},
            {"q": "I see myself as someone who is dependable.", "dim": "C"},
            {"q": "I see myself as someone who is outgoing, sociable.", "dim": "E"},
            {"q": "I see myself as someone who is generally trusting.", "dim": "A"},
            {"q": "I see myself as someone who gets nervous easily.", "dim": "N"},
            {"q": "I see myself as someone who is ingenious, a deep thinker.", "dim": "O"},
            {"q": "I see myself as someone who can be somewhat careless.", "dim": "C", "rev": True},
            {"q": "I see myself as someone who is reserved.", "dim": "E", "rev": True},
            {"q": "I see myself as someone who is considerate and kind to almost everyone.", "dim": "A"},
            {"q": "I see myself as someone who stays calm in tense situations.", "dim": "N", "rev": True},
            {"q": "I see myself as someone who prefers work that is routine.", "dim": "O", "rev": True},
            {"q": "I see myself as someone who follows through with plans.", "dim": "C"},
            {"q": "I see myself as someone who is sometimes shy, inhibited.", "dim": "E", "rev": True},
            {"q": "I see myself as someone who is sometimes rude to others.", "dim": "A", "rev": True},
            {"q": "I see myself as someone who is depressed, blue.", "dim": "N"}
        ]
    }
}

async def calculate_scores(quiz_type, answers):
    """Calculates final scores or types from raw answers."""
    if quiz_type == "mbti":
        dims = {"EI": 0, "SN": 0, "TF": 0, "JP": 0}
        questions = QUIZ_DATA["mbti"]["questions"]
        for i, ans in enumerate(answers):
            dim = questions[i]["dim"]
            dims[dim] += 1 if ans == "A" else -1
        
        mbti_type = (
            ("E" if dims["EI"] >= 0 else "I") +
            ("S" if dims["SN"] >= 0 else "N") +
            ("T" if dims["TF"] >= 0 else "F") +
            ("J" if dims["JP"] >= 0 else "P")
        )
        return f"MBTI Type: {mbti_type}"
    
    elif quiz_type == "ocean":
        scores = {"O": 0, "C": 0, "E": 0, "A": 0, "N": 0}
        questions = QUIZ_DATA["ocean"]["questions"]
        for i, ans in enumerate(answers):
            val = int(ans)
            q = questions[i]
            if q.get("rev"): val = 6 - val # Reverse scoring
            scores[q["dim"]] += val
        
        # Calculate percentage (assuming 5 Qs per dim, max 25 pts)
        results = [f"{k}: {v}/25" for k, v in scores.items()]
        return "OCEAN Scores: " + ", ".join(results)

# =============================================================================
# 6. PHASE 3: ANALYSIS HELPERS & COOLDOWNS
# =============================================================================

async def is_on_cooldown(user_id: str, command: str, seconds: int):
    """
    Checks the persistent database for a cooldown.
    Returns (True, time_remaining_seconds) or False.
    """
    # Owner bypasses all cooldowns
    if OWNER_ID and str(user_id) == OWNER_ID:
        return False

    async with bot.db.execute(
        "SELECT last_used FROM cooldowns WHERE user_id = ? AND command = ?",
        (user_id, command)
    ) as cursor:
        row = await cursor.fetchone()
        if row:
            last_used = datetime.fromisoformat(row[0])
            elapsed = (datetime.now() - last_used).total_seconds()
            if elapsed < seconds:
                return True, int(seconds - elapsed)
    return False

async def set_cooldown(user_id: str, command: str):
    """Updates the persistent cooldown timestamp."""
    await bot.db.execute(
        "INSERT OR REPLACE INTO cooldowns (user_id, command, last_used) VALUES (?, ?, ?)",
        (user_id, command, datetime.now().isoformat())
    )
    await bot.db.commit()

def format_transcript(rows):
    """
    Formats DB rows into a clean dialogue transcript for Gemini.
    Input rows: (content, timestamp)
    Output: "[HH:MM] User: 'Content'"
    """
    transcript = []
    for row in rows:
        try:
            ts = datetime.fromisoformat(row[1]).strftime("%H:%M")
        except:
            ts = "??:??"
        transcript.append(f"[{ts}] User: \"{row[0]}\"")
    return "\n".join(transcript)

async def deliver_dossier(ctx, title, content):
    """
    Helper to send a report to DMs with a fail-safe for closed DMs.
    """
    full_report = f"{title}\n\n{content}{DISCLAIMER}"
    
    # Split content if it exceeds 2000 chars (Discord limit)
    chunks = [full_report[i:i+1900] for i in range(0, len(full_report), 1900)]
    
    try:
        for chunk in chunks:
            await ctx.author.send(chunk)
        return True
    except discord.Forbidden:
        return False

# =============================================================================
# 6. HEARTBEAT WEB SERVER
# =============================================================================

class HeartbeatServer:
    def __init__(self, port=7860):
        self.port = port
        self.runner = None

    async def start(self):
        app = web.Application()
        app.router.add_get('/', lambda r: web.Response(text="Psyche is Awake"))
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, '0.0.0.0', self.port).start()
        log.info("Heartbeat online on port %s", self.port)

    async def stop(self):
        if self.runner: await self.runner.cleanup()

# =============================================================================
# 6. BOT CLASS (Lifecycle Managed)
# =============================================================================

class PsycheBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_server = HeartbeatServer(port=7860)
        self.db: aiosqlite.Connection = None

    async def setup_hook(self):
        """Persistent DB connection and Heartbeat startup."""
        # 1. Start Heartbeat
        await self.web_server.start()
        
        # 2. Establish Persistent DB Connection
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.db = await aiosqlite.connect(DB_PATH)
        self.db.row_factory = aiosqlite.Row  # Access columns by name
        
        # 3. Initialize Schema
        await init_db(self.db)

    async def close(self):
        """Graceful shutdown of DB and Web Server."""
        log.info("Closing bot resources...")
        await self.web_server.stop()
        if self.db:
            await self.db.close()
            log.info("Database connection closed.")
        await super().close()

    async def on_ready(self):
        activity = discord.Activity(type=discord.ActivityType.watching, name="patterns | !help")
        await self.change_presence(status=discord.Status.online, activity=activity)
        log.info("✨ Psyche v2 Online | Engine: %s", MODEL_ID)

# =============================================================================
# 7. EVENT GATES (Privacy & Scrub Protocol)
# =============================================================================

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = PsycheBot(command_prefix='!', intents=intents, help_command=None)

@bot.event
async def on_message(message: discord.Message):
    """The Message Gate: Privacy-first logging."""
    if message.author.bot or isinstance(message.channel, discord.DMChannel):
        return

    # Phase 2: Log only if opted-in
    if is_opted_in(message.author):
        try:
            await bot.db.execute(
                "INSERT INTO messages (user_id, guild_id, content) VALUES (?, ?, ?)",
                (str(message.author.id), str(message.guild.id), message.content)
            )
            await bot.db.commit()
        except Exception as e:
            log.error("DB Log Error: %s", e)

    await bot.process_commands(message)

@bot.event
async def on_guild_remove(guild: discord.Guild):
    """The Scrub Protocol (Guild-Level): Wipes ALL data for this server."""
    log.info("🧹 Guild Scrub: Purging all data for %s (%s)", guild.name, guild.id)
    try:
        gid = str(guild.id)
        await bot.db.execute("DELETE FROM messages WHERE guild_id = ?", (gid,))
        await bot.db.execute("DELETE FROM quiz_results WHERE guild_id = ?", (gid,))
        await bot.db.execute("DELETE FROM interaction_history WHERE guild_id = ?", (gid,))
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
async def assessment(ctx: commands.Context, test_type: str = None):
    """Start a clinical-grade personality assessment (!assessment ipip|oejti|enneagram)."""
    if not is_opted_in(ctx.author):
        return await ctx.reply("🔒 **Privacy Gate**: You need the `PsycheOptIn` role.")

    valid_types = list(ASSESSMENTS.keys())
    if not test_type or test_type.lower() not in valid_types:
        menu = "\n".join([f"`{k}` — {v['name']} ({len(v['items'])} items)" for k, v in ASSESSMENTS.items()])
        return await ctx.reply(f"❓ **Choose an assessment:**\n{menu}\n\nUsage: `!assessment ipip`")

    test_type = test_type.lower()
    user_id = str(ctx.author.id)

    # Check for existing session
    async with bot.db.execute("SELECT quiz_type FROM quiz_sessions WHERE user_id = ?", (user_id,)) as cursor:
        existing = await cursor.fetchone()
        if existing:
            return await ctx.reply(f"⚠️ You have an active `{existing[0]}` session. Use `!assessment_resume` or `!quiz cancel`.")

    # Initialize session in DB
    await bot.db.execute(
        "INSERT OR REPLACE INTO quiz_sessions (user_id, quiz_type, current_question, answers) VALUES (?, ?, ?, ?)",
        (user_id, test_type, 0, "[]")
    )
    await bot.db.commit()

    # Build the View and send to DMs
    try:
        view = AssessmentView(user_id, test_type, 0, [], ctx.author.dm_channel or await ctx.author.create_dm())
        
        # Start embed with disclaimer
        start_embed = discord.Embed(
            title=f"🧪 {ASSESSMENTS[test_type]['name']}",
            description=f"{ASSESSMENTS[test_type]['description']}\n\n"
                        f"**{len(ASSESSMENTS[test_type]['items'])} questions** • Likert Scale (1-5)\n"
                        f"Your progress is auto-saved after every answer.",
            color=discord.Color.from_str("#9B59B6")
        )
        apply_disclaimer(start_embed)
        await ctx.author.send(embed=start_embed)

        msg = await ctx.author.send(embed=view.build_embed(), view=view)
        view.message = msg
        await ctx.reply("📩 Check your DMs to begin the assessment!")
    except discord.Forbidden:
        await ctx.reply("❌ I can't DM you! Please open your privacy settings.")

@bot.command(name='assessment_resume')
async def assessment_resume(ctx: commands.Context):
    """Resumes a paused clinical assessment from the exact question."""
    user_id = str(ctx.author.id)

    async with bot.db.execute(
        "SELECT quiz_type, current_question, answers FROM quiz_sessions WHERE user_id = ?", (user_id,)
    ) as cursor:
        row = await cursor.fetchone()
        if not row:
            return await ctx.reply("❌ No active assessment session found.")

    quiz_type, progress, answers_json = row[0], row[1], row[2]
    answers = json.loads(answers_json)

    try:
        dm = ctx.author.dm_channel or await ctx.author.create_dm()
        view = AssessmentView(user_id, quiz_type, progress, answers, dm)

        resume_embed = discord.Embed(
            title="▶️ Resuming Assessment",
            description=f"**{ASSESSMENTS[quiz_type]['name']}**\nPicking up at question {progress + 1}/{len(ASSESSMENTS[quiz_type]['items'])}.",
            color=discord.Color.from_str("#9B59B6")
        )
        apply_disclaimer(resume_embed)
        await ctx.author.send(embed=resume_embed)

        msg = await ctx.author.send(embed=view.build_embed(), view=view)
        view.message = msg
        await ctx.reply("📩 Resuming in DMs...")
    except discord.Forbidden:
        await ctx.reply("❌ I can't DM you! Please open your privacy settings.")

# =============================================================================
# 8. INTERACTION SCRAPER COMMAND (Free-Tier Optimized)
# =============================================================================

@bot.command(name='map_interactions')
async def map_interactions(ctx: commands.Context):
    """Maps the total interaction history of the server securely."""
    
    # 1. Privacy & Server Gate
    if not is_opted_in(ctx.author):
        return await ctx.send("🔒 **Access Denied:** You must have the `PsycheOptIn` role to map your interactions.")
    
    if not ctx.guild:
        return await ctx.send("⚠️ This command must be run inside the private server, not in DMs.")

    # 2. UI Feedback
    status_msg = await ctx.send(
        "🔍 **Initializing Total Interaction Scraper...**\n"
        "*This may take a while. The database is batching writes to conserve CPU.*"
    )
    
    total_mapped = 0
    batch_rows = []

    # 3. The Iterative Crawl (Optimized for 2 vCPUs)
    for channel in ctx.guild.text_channels:
        # Skip channels the bot cannot read
        if not channel.permissions_for(ctx.guild.me).read_message_history:
            continue
            
        try:
            # oldest_first=True builds the timeline chronologically
            async for message in channel.history(limit=None, oldest_first=True):
                
                # Only save if the user sent it
                if message.author.id == ctx.author.id:
                    reply_id = str(message.reference.message_id) if message.reference else None
                    
                    # Add to RAM batch instead of hitting the DB immediately
                    batch_rows.append((
                        str(message.id), 
                        str(message.author.id), 
                        message.content, 
                        reply_id, 
                        message.created_at.isoformat()
                    ))
                    total_mapped += 1

                    # --- FREE TIER CPU PROTECTION ---
                    # Only write to the hard drive every 200 messages
                    if len(batch_rows) >= 200:
                        await bot.db.executemany(
                            "INSERT OR IGNORE INTO interaction_history "
                            "(message_id, user_id, content, reply_to_id, timestamp) "
                            "VALUES (?, ?, ?, ?, ?)", 
                            batch_rows
                        )
                        await bot.db.commit()
                        batch_rows.clear()
                        
                        # Update the Discord UI every 200 messages
                        await status_msg.edit(
                            content=f"🔍 **Mapping in progress...**\n"
                                    f"Interactions mapped: `{total_mapped:,}`"
                        )
                        
                        # Yield back to the async loop so the bot doesn't freeze
                        await asyncio.sleep(0.5)

        except discord.Forbidden:
            continue  # Silently skip hidden admin channels

    # 4. Final Cleanup Commit (Catching the remainders)
    if batch_rows:
        await bot.db.executemany(
            "INSERT OR IGNORE INTO interaction_history "
            "(message_id, user_id, content, reply_to_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?)", 
            batch_rows
        )
        await bot.db.commit()

    # 5. Completion UI
    embed = discord.Embed(
        title="✅ Social Web Mapped",
        description=f"Successfully extracted and secured **{total_mapped:,}** interactions for {ctx.author.mention}.",
        color=discord.Color.brand_green()
    )
    embed.set_footer(
        text="⚠️ DISCLAIMER: This data is securely stored for AI behavioral conjecture only. "
             "Seek professional help for mental health concerns."
    )
    await status_msg.edit(content=None, embed=embed)

# --- THE SCRUB PROTOCOL (Data Rights) ---
@bot.command(name='purge_my_data')
async def purge_my_data(ctx: commands.Context):
    """Allows the user to instantly delete their total history from the bot's DB."""
    await purge_user_data(str(ctx.author.id))
    await ctx.send("🗑️ **Scrub Protocol Executed:** Your interaction history has been permanently deleted from the database.")

# =============================================================================
# 9. CORE ANALYSIS COMMANDS (Gemini Integration)
# =============================================================================

@bot.command(name='analyze_me')
async def analyze_me(ctx: commands.Context):
    """Behavioral profile based on last 500 messages."""
    if not is_opted_in(ctx.author):
        return await ctx.reply("❌ **Privacy Error**: You must have the `PsycheOptIn` role to use this.")

    # 1. Persistent Cooldown Check (10 Minutes)
    cooldown = await is_on_cooldown(str(ctx.author.id), "analyze_me", 600)
    if cooldown:
        return await ctx.reply(f"⏳ **Cooldown**: Please wait {round(cooldown[1]/60)} minutes.")

    # 2. Fetch Data
    async with bot.db.execute(
        "SELECT content, timestamp FROM messages WHERE user_id = ? AND guild_id = ? ORDER BY timestamp DESC LIMIT 500",
        (str(ctx.author.id), str(ctx.guild.id))
    ) as cursor:
        rows = await cursor.fetchall()

    if len(rows) < 50:
        return await ctx.reply(f"❌ **Data Gap**: I need 50+ messages for a valid profile (you have {len(rows)}).")

    # 3. Process with Gemini
    async with ctx.typing():
        transcript = format_transcript(rows)
        prompt = (
            "Act as a world-class behavioral psychologist. Analyze the following chat transcript "
            "for linguistic style, emotional tone, and social dynamics. Provide a concise, "
            "insightful profile (approx 300 words).\n\n"
            f"Transcript:\n{transcript}"
        )
        
        try:
            # Run blocking AI call in a thread
            response = await asyncio.to_thread(model.generate_content, prompt)
            
            # 4. Delivery
            success = await deliver_dossier(ctx, f"📊 **Behavioral Profile: {ctx.author.name}**", response.text)
            
            if success:
                await ctx.reply("✅ **Analysis complete.** Check your DMs for the dossier.")
                await set_cooldown(str(ctx.author.id), "analyze_me")
            else:
                await ctx.reply("❌ **Delivery Error**: I can't DM you! Please open your privacy settings.")
        
        except Exception as e:
            log.error("AI Error: %s", e)
            await ctx.reply("⚠️ **Engine Overload**: Gemini is struggling to process this. Try again in a moment.")

@bot.command(name='ultimate_analysis')
async def ultimate_analysis(ctx: commands.Context):
    """The Milestone Report: Chat history + Quiz synthesis."""
    if not is_opted_in(ctx.author):
        return await ctx.reply("❌ **Privacy Error**: You must have the `PsycheOptIn` role.")

    # 1. Persistent Cooldown Check (7 Days)
    cooldown = await is_on_cooldown(str(ctx.author.id), "ultimate_analysis", 604800)
    if cooldown:
        return await ctx.reply(f"⏳ **Cooldown**: Major Dossiers take time. Next available: {round(cooldown[1]/3600)} hours.")

    # 2. Fetch Data (1,000 messages + Quizzes)
    async with bot.db.execute(
        "SELECT content, timestamp FROM messages WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1000",
        (str(ctx.author.id),)
    ) as cursor:
        rows = await cursor.fetchall()
        
    async with bot.db.execute(
        "SELECT quiz_type, result_summary FROM quiz_results WHERE user_id = ?",
        (str(ctx.author.id),)
    ) as cursor:
        quizzes = await cursor.fetchall()

    if len(rows) < 50 or not quizzes:
        return await ctx.reply("❌ **Missing Inputs**: You need 50+ messages AND at least one completed quiz (!take_test).")

    # 3. Process with Gemini
    async with ctx.typing():
        transcript = format_transcript(rows)
        quiz_data = "\n".join([f"[{q[0].upper()}] {q[1]}" for q in quizzes])
        
        prompt = (
            "Act as a Forensic Psychologist. Synthesize this user's chat history and personality test results "
            "into a Master Profile. Explore contradictions between their self-reported quiz data and their "
            "actual behavior in chat. Aim for a deep, 1000-word synthesis.\n\n"
            f"Chat History:\n{transcript}\n\nQuiz Data:\n{quiz_data}"
        )
        
        try:
            response = await asyncio.to_thread(model.generate_content, prompt)
            
            # 4. Delivery
            success = await deliver_dossier(ctx, f"🏆 **ULTIMATE PSYCHOLOGICAL DOSSIER: {ctx.author.name}**", response.text)
            
            if success:
                await ctx.reply("✅ **Synthesis complete.** Your Master Dossier has arrived in your DMs.")
                await set_cooldown(str(ctx.author.id), "ultimate_analysis")
            else:
                await ctx.reply("❌ **Delivery Error**: I can't DM you! Please check your privacy settings.")

        except Exception as e:
            log.error("AI Error: %s", e)
            await ctx.reply("⚠️ **Synthesis Failed**: The AI encountered a complex conflict. Try again later.")

# =============================================================================
# 9. PHASE 4: QUIZ COMMANDS & LOOP
# =============================================================================

async def run_quiz_loop(ctx, quiz_type, start_index=0, existing_answers=None):
    """The interactive DM-based quiz loop."""
    user_id = str(ctx.author.id)
    data = QUIZ_DATA[quiz_type]
    answers = existing_answers or []
    
    await ctx.author.send(f"🏁 **Starting {data['name']}**\n{data['instructions']}\nType `cancel` at any time to abort.")

    for i in range(start_index, len(data["questions"])):
        q = data["questions"][i]
        prompt_text = f"**Question {i+1}/{len(data['questions'])}**\n{q['q']}"
        if quiz_type == "mbti":
            prompt_text += f"\n**A)** {q['a']}\n**B)** {q['b']}"
        
        await ctx.author.send(prompt_text)

        def check(m):
            if m.author.id != ctx.author.id or not isinstance(m.channel, discord.DMChannel):
                return False
            val = m.content.upper().strip()
            if val == "CANCEL": return True
            if quiz_type == "mbti": return val in ["A", "B"]
            if quiz_type == "ocean": return val in ["1", "2", "3", "4", "5"]
            return False

        try:
            msg = await bot.wait_for('message', timeout=300.0, check=check)
            val = msg.content.upper().strip()
            
            if val == "CANCEL":
                await ctx.author.send("❌ Quiz cancelled. Use `!quiz resume` later to pick up where you left off.")
                return

            answers.append(val)
            # Persistence
            await bot.db.execute(
                "INSERT OR REPLACE INTO quiz_sessions (user_id, quiz_type, current_question, answers) VALUES (?, ?, ?, ?)",
                (user_id, quiz_type, i + 1, json.dumps(answers))
            )
            await bot.db.commit()

        except asyncio.TimeoutError:
            await ctx.author.send("⏰ **Timeout**: You took too long. I've saved your progress. Use `!quiz resume` when you're back!")
            return

    # Completion
    await ctx.author.send("✅ **Quiz Complete!** Generating your psychological profile...")
    
    scores = await calculate_scores(quiz_type, answers)
    
    # Gemini Synthesis
    prompt = (
        f"The user scored {scores} on the {data['name']} test. "
        "Write a personalized 200-word summary of these results in a professional "
        "psychological tone. Use insight and depth."
    )
    
    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        summary = response.text
        
        # Save Results
        await bot.db.execute(
            "INSERT OR REPLACE INTO quiz_results (user_id, guild_id, quiz_type, result_summary, raw_answers, completed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, str(ctx.guild.id) if ctx.guild else "0", quiz_type, summary, json.dumps(answers), datetime.now().isoformat())
        )
        # Wipe Session
        await bot.db.execute("DELETE FROM quiz_sessions WHERE user_id = ?", (user_id,))
        await bot.db.commit()

        await ctx.author.send(f"📊 **Your Results Summary:**\n\n{summary}{DISCLAIMER}")
    except Exception as e:
        log.error("Quiz Gemini Error: %s", e)
        await ctx.author.send("⚠️ Synthesis failed, but your raw scores were saved.")

@bot.command(name='take_test')
async def take_test(ctx, test_type: str = None):
    """Start a personality assessment (!take_test mbti|ocean)."""
    if not is_opted_in(ctx.author):
        return await ctx.reply("❌ Privacy gate: You need the `PsycheOptIn` role.")
    
    if not test_type or test_type.lower() not in QUIZ_DATA:
        return await ctx.reply("❓ Please specify: `!take_test mbti` or `!take_test ocean`.")

    user_id = str(ctx.author.id)
    async with bot.db.execute("SELECT quiz_type FROM quiz_sessions WHERE user_id = ?", (user_id,)) as cursor:
        if await cursor.fetchone():
            return await ctx.reply("⚠️ You have an active session! Use `!quiz resume` or `!quiz cancel`.")

    try:
        await ctx.author.send("🧠 **Initializing Psyche Assessment Module...**")
        await ctx.reply("📩 Check your DMs to begin!")
        await run_quiz_loop(ctx, test_type.lower())
    except discord.Forbidden:
        await ctx.reply("❌ I can't DM you! Please open your privacy settings.")

@bot.group(name='quiz', invoke_without_command=True)
async def quiz(ctx):
    """Quiz management commands (!quiz resume|cancel)."""
    await ctx.reply("Usage: `!quiz resume` or `!quiz cancel`")

@quiz.command(name='resume')
async def quiz_resume(ctx):
    """Resumes an in-progress quiz."""
    async with bot.db.execute("SELECT quiz_type, current_question, answers FROM quiz_sessions WHERE user_id = ?", (str(ctx.author.id),)) as cursor:
        row = await cursor.fetchone()
        if not row:
            return await ctx.reply("❌ No active session found.")
        
        await ctx.reply("📩 Resuming in DMs...")
        await run_quiz_loop(ctx, row[0], start_index=row[1], existing_answers=json.loads(row[2]))

@quiz.command(name='cancel')
async def quiz_cancel(ctx):
    """Wipes an in-progress quiz session."""
    await bot.db.execute("DELETE FROM quiz_sessions WHERE user_id = ?", (str(ctx.author.id),))
    await bot.db.commit()
    await ctx.reply("🗑️ Active session wiped.")

# =============================================================================
# 10. SYSTEM COMMANDS
# =============================================================================

@bot.command(name='ping')
async def ping(ctx: commands.Context):
    """
    Senior DevOps Ping:
    Returns latency and verifies if the persistent /data volume is writable.
    """
    latency = round(bot.latency * 1000)
    db_writable = False
    
    # Verify DB Writability
    try:
        await bot.db.execute("CREATE TABLE IF NOT EXISTS _ping (id INTEGER PRIMARY KEY)")
        await bot.db.execute("INSERT INTO _ping (id) VALUES (?)", (int(datetime.now().timestamp()),))
        await bot.db.commit()
        db_writable = True
    except Exception as e:
        log.error("DB Write Check Failed: %s", e)

    embed = discord.Embed(
        title="🛰️ System Status",
        color=discord.Color.green() if db_writable else discord.Color.red()
    )
    embed.add_field(name="Gateway Latency", value=f"`{latency}ms`", inline=True)
    embed.add_field(name="Database (/data)", value="`Writable` ✅" if db_writable else "`Read-Only` ❌", inline=True)
    embed.add_field(name="Uptime Heartbeat", value="`Listening` 🎧", inline=True)
    
    apply_disclaimer(embed)
    await ctx.send(embed=embed)

if __name__ == '__main__':
    bot.run(DISCORD_TOKEN, reconnect=True, log_handler=None)
