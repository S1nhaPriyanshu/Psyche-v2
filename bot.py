import certifi
import os

# Set global environment variables immediately
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['WEBSOCKET_CLIENT_CA_BUNDLE'] = certifi.where() # Extra for 2026 stability

# =============================================================================
# Psyche v2 — Core Bot
# A privacy-first, high-reasoning Discord behavioral analysis bot.
# Powered by Google Gemini 3.1 Pro.
# =============================================================================

import asyncio
import aiosqlite         # Phase 2: Async SQLite
import logging
import signal
import json
import time
from datetime import datetime
import aiohttp
import ssl

# The Connector Refactor: Force all aiohttp clients to use the Certifi context
_orig_TCPConnector = aiohttp.TCPConnector
def _patched_TCPConnector(*args, **kwargs):
    if 'ssl' not in kwargs or kwargs['ssl'] is None or kwargs['ssl'] is True:
        kwargs['ssl'] = ssl.create_default_context(cafile=certifi.where())
    return _orig_TCPConnector(*args, **kwargs)
aiohttp.TCPConnector = _patched_TCPConnector

# Now safe to import network-reliant libraries
import discord
from discord.ext import commands
from discord import ui
from aiohttp import web
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
try:
    OWNER_ID = int(os.getenv('OWNER_ID', '0'))
except ValueError:
    OWNER_ID = 0
SCAN_MODEL      = os.getenv('SCAN_MODEL', 'gemini-3.1-flash-lite')
DOSSIER_MODEL   = os.getenv('DOSSIER_MODEL', 'gemini-3.1-pro')
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

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# System instruction forces the "Clinical" persona globally
SYSTEM_INSTRUCTION = (
    "You are Psyche, an advanced Forensic Psychology AI. Your task is to analyze "
    "Discord interactions and raw psychometric data. Look for cognitive dissonance, "
    "social archetypes, and linguistic patterns. NEVER provide a medical or psychiatric "
    "diagnosis. Frame all insights as 'behavioral conjecture' based on text patterns."
)

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
            user_id          TEXT PRIMARY KEY,
            last_dossier_time REAL
        )
    ''')

    # -- 6. Interaction History Table (Single-Server Deep Scraper) --
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
    return user_id == OWNER_ID

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
    def __init__(self, bot_instance, user_id, quiz_type, questions, progress=0, answers=None):
        super().__init__(timeout=600)  # 10-minute idle timeout
        self.bot = bot_instance
        self.user_id = int(user_id)
        self.quiz_type = quiz_type
        self.questions = questions  # List of strings
        self.progress = progress
        self.answers = answers or []

    async def update_question(self, interaction: discord.Interaction):
        """Edits the existing message to the next question. Minimizes Discord API calls."""
        if self.progress >= len(self.questions):
            await self.finish_quiz(interaction)
            return

        q_text = self.questions[self.progress]
        embed = discord.Embed(
            title=f"Assessment: {self.quiz_type.upper()}",
            description=f"**Question {self.progress + 1} of {len(self.questions)}**\n\n{q_text}",
            color=discord.Color.blue()
        )
        embed = apply_disclaimer(embed)
        await interaction.response.edit_message(embed=embed, view=self)

    async def handle_choice(self, interaction: discord.Interaction, value: int):
        """Handles button clicks and saves raw data to HF Bucket."""
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("This session is not yours.", ephemeral=True)

        self.answers.append(value)
        self.progress += 1

        # Save Raw Progress to DB (Zero-CPU Strategy)
        await self.bot.db.execute(
            "INSERT OR REPLACE INTO quiz_sessions (user_id, quiz_type, progress, answers) VALUES (?, ?, ?, ?)",
            (str(self.user_id), self.quiz_type, self.progress, json.dumps(self.answers))
        )
        await self.bot.db.commit()

        await self.update_question(interaction)

    # Likert Scale Buttons (1-5)
    @discord.ui.button(label="1", style=discord.ButtonStyle.grey)
    async def c1(self, interaction, button): await self.handle_choice(interaction, 1)
    @discord.ui.button(label="2", style=discord.ButtonStyle.grey)
    async def c2(self, interaction, button): await self.handle_choice(interaction, 2)
    @discord.ui.button(label="3", style=discord.ButtonStyle.grey)
    async def c3(self, interaction, button): await self.handle_choice(interaction, 3)
    @discord.ui.button(label="4", style=discord.ButtonStyle.grey)
    async def c4(self, interaction, button): await self.handle_choice(interaction, 4)
    @discord.ui.button(label="5", style=discord.ButtonStyle.grey)
    async def c5(self, interaction, button): await self.handle_choice(interaction, 5)

    async def finish_quiz(self, interaction: discord.Interaction):
        """Zero-CPU Completion: No math, just data handoff."""
        timestamp = datetime.now().isoformat()
        raw_data = json.dumps(self.answers)

        async with self.bot.db.cursor() as cursor:
            # Move from active session to permanent results
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
        await interaction.response.edit_message(embed=apply_disclaimer(embed), view=None)

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

def format_transcript(rows):
    """Formats raw DB rows into a readable script without bloating memory."""
    transcript = []
    for row in rows:
        content = row[0]
        timestamp = row[1]
        is_reply = " (Reply)" if row[2] else ""
        
        # Truncate extremely long copypastas to save tokens
        if len(content) > 400:
            content = content[:400] + "...[truncated]"
            
        transcript.append(f"[{timestamp[:10]}] User{is_reply}: {content}")
    return "\n".join(transcript)

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
        scan_m = os.getenv('SCAN_MODEL', 'Unknown')
        dossier_m = os.getenv('DOSSIER_MODEL', 'Unknown')
        log.info(f"✨ Psyche v2 Online | Scan: {scan_m} | Dossier: {dossier_m}")

# =============================================================================
# 7. EVENT GATES (Privacy & Scrub Protocol)
# =============================================================================

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = PsycheBot(
    command_prefix=commands.when_mentioned_or('!'), 
    case_insensitive=True, 
    intents=intents, 
    help_command=None
)

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

    # Phase 2: Log only if opted-in AND in a Server (No DM logging)
    log.info(f"Checking opt-in for {message.author}...")
    if not isinstance(message.channel, discord.DMChannel) and is_opted_in(message.author):
        log.info(f"User {message.author} is opted in. Saving to database.")
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
async def assessment(ctx, quiz_type: str = None):
    """Starts a new assessment in DMs."""
    if not is_opted_in(ctx.author):
        return await ctx.reply("🔒 **Privacy Gate**: You need the `PsycheOptIn` role.")

    if not quiz_type or quiz_type.lower() not in ["ocean", "mbti", "enneagram"]:
        return await ctx.send("❌ Use: `!assessment [ocean|mbti|enneagram]`")

    quiz_type = quiz_type.lower()

    # Load questions from local JSON
    with open('questions.json', 'r') as f:
        data = json.load(f)
    questions = data.get(quiz_type, [])

    # Check for existing session
    async with bot.db.execute("SELECT user_id FROM quiz_sessions WHERE user_id = ?", (str(ctx.author.id),)) as cursor:
        if await cursor.fetchone():
            return await ctx.send("⚠️ You have an active session. Use `!assessment_resume` to continue.")

    view = AssessmentView(bot, ctx.author.id, quiz_type, questions)
    embed = discord.Embed(
        title=f"Starting {quiz_type.upper()}",
        description=f"**Question 1:** {questions[0]}"
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

    with open('questions.json', 'r') as f:
        data = json.load(f)
    questions = data.get(quiz_type, [])

    view = AssessmentView(bot, ctx.author.id, quiz_type, questions, progress, answers)
    embed = discord.Embed(
        title=f"Resuming {quiz_type.upper()}",
        description=f"**Question {progress + 1}:** {questions[progress]}"
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
# 9. PHASE 4: BEHAVIOR SCAN (Quick AI Snapshot)
# =============================================================================

@bot.command(name='behavior_scan')
async def behavior_scan(ctx):
    """Analyzes the last 500 messages for an immediate behavioral snapshot."""
    if not is_opted_in(ctx.author):
        return await ctx.send("🔒 You must have the `PsycheOptIn` role.")

    status_msg = await ctx.send("🧠 **Initiating Behavior Scan...** Accessing recent interaction matrix.")

    # 1. Fetch recent data
    async with bot.db.execute(
        "SELECT content, timestamp, reply_to_id FROM interaction_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 500",
        (str(ctx.author.id),)
    ) as cursor:
        rows = await cursor.fetchall()

    if len(rows) < 50:
        return await status_msg.edit(content="⚠️ **Insufficient Data:** I need at least 50 interactions to form a baseline.")

    transcript = format_transcript(rows[::-1])  # Reverse to chronological order

    # 2. Async AI Generation (Zero-CPU blocking)
    prompt = (
        f"Analyze this recent chat transcript. Provide a concise, 3-paragraph snapshot covering: "
        f"1. Current emotional tone. 2. Primary communication style. 3. Social role in the server.\n\n"
        f"Transcript:\n{transcript}"
    )

    try:
        response = await client.models.generate_content(
            model=SCAN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            )
        )
        
        embed = discord.Embed(
            title="🔍 Behavioral Snapshot", 
            description=response.text, 
            color=discord.Color.teal()
        )
        embed = apply_disclaimer(embed)

        await ctx.author.send(embed=embed)
        await status_msg.edit(content="✅ **Scan Complete.** The results have been sent to your DMs.")

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

    # 1. Strict 7-Day Cooldown Check
    async with bot.db.execute("SELECT last_dossier_time FROM cooldowns WHERE user_id = ?", (user_id,)) as cursor:
        row = await cursor.fetchone()
        if row:
            elapsed = current_time - row[0]
            if elapsed < 604800:  # 7 days in seconds
                days_left = round((604800 - elapsed) / 86400, 1)
                return await ctx.send(f"⏳ **Cooldown Active:** Deep Synthesis requires massive computation. Please wait {days_left} days.")

    status_msg = await ctx.send("🧬 **Initiating Deep Synthesis...** Aggregating total social web and psychometric data. *This may take a minute.*")

    # 2. Fetch Total History
    async with bot.db.execute(
        "SELECT content, timestamp, reply_to_id FROM interaction_history WHERE user_id = ? ORDER BY timestamp ASC",
        (user_id,)
    ) as cursor:
        chat_rows = await cursor.fetchall()
    
    if len(chat_rows) < 100:
        return await status_msg.edit(content="⚠️ **Insufficient Data:** Run `!map_interactions` to build your social web first.")

    # 3. Fetch Raw Quiz Data
    async with bot.db.execute("SELECT quiz_type, raw_answers FROM quiz_results WHERE user_id = ?", (user_id,)) as cursor:
        quiz_rows = await cursor.fetchall()

    quiz_context = ""
    for q_type, raw_json in quiz_rows:
        quiz_context += f"- {q_type.upper()} Raw Likert Scale Array: {raw_json}\n"

    if not quiz_context:
        quiz_context = "No psychometric assessments completed. Rely strictly on behavioral data."

    # 4. The Master Prompt
    transcript = format_transcript(chat_rows)
    prompt = (
        "Generate a 'Psychological Synthesis Dossier' for this user.\n"
        "GOALS:\n"
        "1. Reconcile their self-reported psychometric data (Likert arrays) against their actual behavioral text.\n"
        "2. Identify 'Cognitive Dissonance'—where do they act differently than they test?\n"
        "3. Define their overarching server archetype.\n"
        "Format with clear Markdown headers and bold text. Make it profound, clinical, and detailed (1000+ words).\n\n"
        f"=== RAW PSYCHOMETRIC DATA ===\n{quiz_context}\n\n"
        f"=== TOTAL INTERACTION WEB ===\n{transcript}"
    )

    try:
        response = await asyncio.wait_for(
            client.models.generate_content(
                model=DOSSIER_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            ), 
            timeout=90.0
        )
        
        embed = discord.Embed(
            title="🧬 Deep Synthesis Dossier", 
            description=response.text[:4000],  # Discord embed limit is 4096
            color=discord.Color.dark_purple()
        )
        embed = apply_disclaimer(embed)

        await ctx.author.send(embed=embed)
        
        # Set the Cooldown ONLY on success
        await bot.db.execute(
            "INSERT OR REPLACE INTO cooldowns (user_id, last_dossier_time) VALUES (?, ?)",
            (user_id, current_time)
        )
        await bot.db.commit()

        await status_msg.edit(content="✅ **Synthesis Complete.** The secure dossier has been delivered to your DMs.")

    except asyncio.TimeoutError:
        await status_msg.edit(content="⚠️ **AI Engine Timeout:** The data volume was too large for the current model allocation. Try again later.")
    except errors.APIError as e:
        if "429" in str(e) or "ResourceExhausted" in str(e):
            await status_msg.edit(content="⚠️ **Rate Limit Hit:** The Deep Synthesis engine is overloaded. Please try again later.")
        else:
            await status_msg.edit(content=f"⚠️ **Synthesis API Error:** {str(e)}")
    except Exception as e:
        await status_msg.edit(content=f"⚠️ **Synthesis Error:** {str(e)}")

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
    """Connection health check."""
    latency = round(bot.latency * 1000)
    db_status = "Connected (/data/psyche.db)" if bot.db else "Disconnected"
    scan_m = os.getenv('SCAN_MODEL', 'gemini-3.1-flash-lite')
    engine_str = "Gemini 3.1 Flash-Lite" if "flash-lite" in scan_m.lower() else scan_m

    msg = (f"Pong! 🏓\n"
           f"Latency: {latency}ms\n"
           f"Database: {db_status}\n"
           f"Engine: {engine_str} Active")
    await ctx.send(msg)

# =============================================================================
# 11. PHASE 6: THE CREATOR'S SKELETON KEY
# =============================================================================

@bot.command(name="system_query", hidden=True)
async def system_query(ctx, target_id: str, *, query: str):
    """
    Owner-only command for deep-dive behavioral forensics.
    Locked to OWNER_ID and DM-only.
    """
    
    # 1. HARD SECURITY LOCK
    # Only triggers if the author is the owner; silent fail otherwise.
    if ctx.author.id != OWNER_ID:
        return

    # 2. PRIVACY LOCK
    # Ensure this is never run in a public channel where others can see the target_id
    if not isinstance(ctx.channel, discord.DMChannel):
        try:
            await ctx.message.delete()  # Wipe the evidence of the command
        except:
            pass
        return

    # 3. INITIALIZATION
    status_msg = await ctx.send(f"🛡️ **Access Granted.** Querying psychological footprint for UID: `{target_id}`...")

    # 4. DATA AGGREGATION (Pulling from HF Bucket)
    try:
        # Fetch Total History
        async with bot.db.execute(
            "SELECT content, timestamp, reply_to_id FROM interaction_history WHERE user_id = ? ORDER BY timestamp ASC",
            (target_id,)
        ) as cursor:
            chat_rows = await cursor.fetchall()

        # Fetch All Quiz Results
        async with bot.db.execute(
            "SELECT quiz_type, raw_answers FROM quiz_results WHERE user_id = ?",
            (target_id,)
        ) as cursor:
            quiz_rows = await cursor.fetchall()

        if not chat_rows and not quiz_rows:
            return await status_msg.edit(content=f"❌ **Data Void:** No records found for User ID `{target_id}`.")

        # 5. PREPARING DATA FOR GEMINI
        transcript = format_transcript(chat_rows)
        quiz_data = "\n".join([f"{r[0].upper()}: {r[1]}" for r in quiz_rows])

        admin_prompt = (
            f"SYSTEM ADMINISTRATION OVERRIDE: Forensic Analysis Required.\n"
            f"TARGET USER ID: {target_id}\n\n"
            f"CREATOR'S SPECIFIC INQUIRY: {query}\n\n"
            f"DATA CONTEXT:\n"
            f"--- QUIZ DATA ---\n{quiz_data}\n\n"
            f"--- CHAT LOGS ---\n{transcript}\n\n"
            f"INSTRUCTION: Provide a cold, clinical, and high-fidelity answer to the creator's inquiry "
            f"based strictly on the behavioral evidence provided above."
        )

        # 6. ASYNC AI GENERATION
        response = await asyncio.wait_for(
            client.models.generate_content(
                model=DOSSIER_MODEL,
                contents=admin_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            ),
            timeout=120.0  # Extended timeout for massive admin queries
        )

        # 7. FORMATTING THE ADMIN REPORT
        embed = discord.Embed(
            title="🛡️ Administrative Intelligence Report",
            description=response.text[:4000],
            color=discord.Color.red()  # Red indicates 'Admin Level'
        )
        embed.add_field(name="Target User", value=f"<@{target_id}>", inline=True)
        embed.add_field(name="Data Points", value=f"{len(chat_rows)} interactions", inline=True)
        
        # We still apply the disclaimer to maintain professional standards
        embed = apply_disclaimer(embed)

        await ctx.send(embed=embed)
        await status_msg.edit(content="✅ **Query Resolved.** Footprint analysis complete.")

    except asyncio.TimeoutError:
        await status_msg.edit(content="⚠️ **System Timeout:** The target's data footprint is too massive for a single pass.")
    except errors.APIError as e:
        if "429" in str(e) or "ResourceExhausted" in str(e):
            await status_msg.edit(content="⚠️ **Rate Limit Hit:** Administrative interface quota exceeded.")
        else:
            await status_msg.edit(content=f"⚠️ **Internal API Error:** {str(e)}")
    except Exception as e:
        await status_msg.edit(content=f"⚠️ **Internal Error:** {str(e)}")

# =============================================================================
# 12. UTILITY COMMANDS
# =============================================================================

@bot.command(name="help")
async def help_command(ctx):
    embed = discord.Embed(
        title="🧠 Psyche v2 | Forensic Protocol",
        description="Authorized command modules for psychological reconstruction.",
        color=0x2f3136
    )
    embed.add_field(name="📡 Utility", value="`!ping` - Connection health check\n`!purge_my_data` - Privacy wipe", inline=False)
    embed.add_field(name="🕵️ Forensics", value="`!map_interactions` - Global server history scrape\n`!behavior_scan @user` - AI linguistic snapshot (Flash)", inline=False)
    embed.add_field(name="🔬 Clinical", value="`!assessment ocean` - Launch OCEAN test\n`!assessment_resume` - Continue test\n`!generate_dossier` - Full profile synthesis (Pro)", inline=False)
    await ctx.send(embed=embed)

if __name__ == '__main__':
    bot.run(DISCORD_TOKEN, reconnect=True, log_handler=None)
