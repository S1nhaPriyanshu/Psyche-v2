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
from aiohttp import web
from dotenv import load_dotenv
import google.generativeai as genai

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
    
    await db.commit()
    log.info("Database schema ready.")

def is_opted_in(member: discord.Member) -> bool:
    """Checks for the role named 'psycheoptin' (case-insensitive)."""
    if not isinstance(member, discord.Member): return False
    return any(role.name.lower() == "psycheoptin" for role in member.roles)

def is_owner(user_id: int) -> bool:
    """Compares against the OWNER_ID env var."""
    return str(user_id) == OWNER_ID

# =============================================================================
# 5. PHASE 3: ANALYSIS HELPERS & COOLDOWNS
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
    """The Scrub Protocol: Auto-purge on server leave."""
    log.info("🧹 Scrub Protocol: Purging data for guild %s (%s)", guild.name, guild.id)
    try:
        await bot.db.execute("DELETE FROM messages WHERE guild_id = ?", (str(guild.id),))
        await bot.db.execute("DELETE FROM quiz_results WHERE guild_id = ?", (str(guild.id),))
        await bot.db.commit()
        log.info("✅ Scrub Complete.")
    except Exception as e:
        log.error("Scrub Protocol Error: %s", e)

# =============================================================================
# 8. CORE ANALYSIS COMMANDS (Gemini Integration)
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
# 9. SYSTEM COMMANDS
# =============================================================================

@bot.command(name='ping')
async def ping(ctx):
    """System heartbeat check."""
    await ctx.send(f"🏓 Pong! Latency: {round(bot.latency * 1000)}ms")

if __name__ == '__main__':
    bot.run(DISCORD_TOKEN, reconnect=True, log_handler=None)
