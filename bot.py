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
# 5. HEARTBEAT WEB SERVER
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
# 8. SYSTEM COMMANDS
# =============================================================================

@bot.command(name='ping')
async def ping(ctx):
    """System heartbeat check."""
    await ctx.send(f"🏓 Pong! Latency: {round(bot.latency * 1000)}ms")

if __name__ == '__main__':
    bot.run(DISCORD_TOKEN, reconnect=True, log_handler=None)
