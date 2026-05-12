# =============================================================================
# questions.py — Clinical Assessment Question Banks
# Loaded ONCE at import time into RAM. No file I/O per button click.
# =============================================================================
# Format: Each test is a dict with "name", "description", and "items" (list of strings).
# All items use a 1-5 Likert scale (Strongly Disagree → Strongly Agree).
# =============================================================================

ASSESSMENTS = {
    "ipip": {
        "name": "IPIP-NEO Personality Inventory",
        "description": "A 60-item inventory measuring the Big Five personality domains and their facets.",
        "items": [
            # --- Neuroticism (N) ---
            "I often feel blue.",
            "I dislike myself.",
            "I am often down in the dumps.",
            "I have frequent mood swings.",
            "I panic easily.",
            "I rarely get irritated.",  # R
            "I seldom feel blue.",  # R
            "I feel comfortable with myself.",  # R
            "I am not easily bothered by things.",  # R
            "I am very pleased with myself.",  # R
            "I am relaxed most of the time.",  # R
            "I seldom get mad.",  # R
            # --- Extraversion (E) ---
            "I feel comfortable around people.",
            "I make friends easily.",
            "I am skilled in handling social situations.",
            "I am the life of the party.",
            "I know how to captivate people.",
            "I have little to say.",  # R
            "I keep in the background.",  # R
            "I would describe my experiences as somewhat dull.",  # R
            "I don't like to draw attention to myself.",  # R
            "I don't talk a lot.",  # R
            "I am quiet around strangers.",  # R
            "I find it difficult to approach others.",  # R
            # --- Openness (O) ---
            "I believe in the importance of art.",
            "I have a vivid imagination.",
            "I tend to vote for liberal political candidates.",
            "I carry the conversation to a higher level.",
            "I enjoy hearing new ideas.",
            "I am not interested in abstract ideas.",  # R
            "I do not like art.",  # R
            "I avoid philosophical discussions.",  # R
            "I do not enjoy going to art museums.",  # R
            "I tend to vote for conservative political candidates.",  # R
            "I am not interested in theoretical discussions.",  # R
            "I have difficulty understanding abstract ideas.",  # R
            # --- Agreeableness (A) ---
            "I have a good word for everyone.",
            "I believe that others have good intentions.",
            "I respect others.",
            "I accept people as they are.",
            "I make people feel at ease.",
            "I have a sharp tongue.",  # R
            "I cut others to pieces.",  # R
            "I suspect hidden motives in others.",  # R
            "I get back at others.",  # R
            "I insult people.",  # R
            "I believe that I am better than others.",  # R
            "I contradict others.",  # R
            # --- Conscientiousness (C) ---
            "I am always prepared.",
            "I pay attention to details.",
            "I get chores done right away.",
            "I carry out my plans.",
            "I make plans and stick to them.",
            "I waste my time.",  # R
            "I find it difficult to get down to work.",  # R
            "I do just enough work to get by.",  # R
            "I don't see things through.",  # R
            "I shirk my duties.",  # R
            "I mess things up.",  # R
            "I leave things unfinished.",  # R
        ]
    },

    "oejti": {
        "name": "Open Extended Jungian Type Index",
        "description": "A 64-item Jungian typology assessment measuring cognitive preferences.",
        "items": [
            # --- E/I Dimension ---
            "I am the life of the party.",
            "I feel energized by being around other people.",
            "I enjoy meeting new people.",
            "I prefer large social gatherings to quiet evenings.",
            "I am talkative.",
            "I feel drained after social events.",  # R
            "I enjoy solitude.",  # R
            "I prefer one-on-one conversations.",  # R
            "I think before I speak.",  # R
            "I prefer working alone.",  # R
            "I keep my thoughts to myself.",  # R
            "I need time alone to recharge.",  # R
            "I enjoy being the center of attention.",
            "I start conversations.",
            "I feel comfortable in groups.",
            "I prefer quiet activities.",  # R
            # --- S/N Dimension ---
            "I focus on the details.",
            "I prefer concrete facts over abstract theories.",
            "I trust my five senses.",
            "I am practical.",
            "I like clear instructions.",
            "I enjoy theoretical ideas.",  # R
            "I get lost in my imagination.",  # R
            "I see the big picture easily.",  # R
            "I trust my hunches.",  # R
            "I enjoy speculating about the future.",  # R
            "I prefer innovation over tradition.",  # R
            "I notice patterns others miss.",  # R
            "I pay attention to what is real and actual.",
            "I like working with established methods.",
            "I prefer step-by-step instructions.",
            "I am drawn to new possibilities.",  # R
            # --- T/F Dimension ---
            "I make decisions based on logic.",
            "I value fairness over harmony.",
            "I am more analytical than sentimental.",
            "I prefer truth over tact.",
            "I find it easy to critique others' work.",
            "I consider others' feelings before deciding.",  # R
            "I value harmony in relationships.",  # R
            "I am sympathetic.",  # R
            "I am guided by my heart.",  # R
            "I take others' feelings personally.",  # R
            "I avoid conflict at all costs.",  # R
            "I feel others' pain deeply.",  # R
            "I stay calm during arguments.",
            "I am objective in my judgments.",
            "I prioritize efficiency over feelings.",
            "I am moved by stories of suffering.",  # R
            # --- J/P Dimension ---
            "I like to have things decided.",
            "I follow a schedule.",
            "I prefer order and structure.",
            "I plan ahead.",
            "I like to finish one task before starting another.",
            "I am spontaneous.",  # R
            "I keep my options open.",  # R
            "I am flexible with plans.",  # R
            "I prefer to go with the flow.",  # R
            "I work in bursts of energy.",  # R
            "I sometimes leave things to the last minute.",  # R
            "I enjoy surprises.",  # R
            "I am punctual.",
            "I keep my workspace organized.",
            "I feel stressed when things are unfinished.",
            "I prefer to adapt rather than plan.",  # R
        ]
    },

    "enneagram": {
        "name": "Enneagram Personality Assessment",
        "description": "A 54-item assessment measuring nine core personality types and motivations.",
        "items": [
            # --- Type 1: The Reformer ---
            "I have a strong sense of right and wrong.",
            "I hold myself to high standards.",
            "I get frustrated when things aren't done correctly.",
            "I believe in doing the right thing, even when it's hard.",
            "I am my own harshest critic.",
            "I notice mistakes and imperfections easily.",
            # --- Type 2: The Helper ---
            "I feel most fulfilled when I am helping others.",
            "I often put others' needs before my own.",
            "I am attuned to other people's emotions.",
            "I want to be needed by those I care about.",
            "I sometimes feel unappreciated for my efforts.",
            "I find it hard to say no when someone asks for help.",
            # --- Type 3: The Achiever ---
            "I am highly motivated by success.",
            "I adapt my persona to fit different social settings.",
            "I set ambitious goals and work hard to achieve them.",
            "I feel uncomfortable with failure.",
            "I care about how others perceive me.",
            "I measure my self-worth by my accomplishments.",
            # --- Type 4: The Individualist ---
            "I feel fundamentally different from other people.",
            "I am drawn to beauty, art, and deep emotions.",
            "I tend to dwell on what is missing in my life.",
            "I experience emotions more intensely than most.",
            "I value authenticity above all else.",
            "I sometimes feel misunderstood by others.",
            # --- Type 5: The Investigator ---
            "I prefer to observe rather than participate.",
            "I need a lot of time alone to think.",
            "I am protective of my time and energy.",
            "I feel more comfortable with ideas than with people.",
            "I seek to understand how things work.",
            "I minimize my needs to maintain independence.",
            # --- Type 6: The Loyalist ---
            "I am always thinking about what could go wrong.",
            "I value loyalty and commitment in relationships.",
            "I seek guidance from trusted authorities.",
            "I often second-guess my decisions.",
            "I prepare for worst-case scenarios.",
            "I feel anxious when I don't have a plan.",
            # --- Type 7: The Enthusiast ---
            "I am always looking for new experiences.",
            "I have a hard time sitting still.",
            "I avoid negative emotions by staying busy.",
            "I am an optimist at heart.",
            "I get bored easily with routine.",
            "I tend to overcommit to plans and activities.",
            # --- Type 8: The Challenger ---
            "I speak my mind, even if it makes others uncomfortable.",
            "I take charge in group settings.",
            "I dislike feeling controlled by others.",
            "I protect the people I care about fiercely.",
            "I respect strength and directness.",
            "I have a hard time showing vulnerability.",
            # --- Type 9: The Peacemaker ---
            "I go along with others to keep the peace.",
            "I have a hard time saying what I really want.",
            "I avoid conflict whenever possible.",
            "I feel comfortable going with the flow.",
            "I tend to merge with other people's agendas.",
            "I find it hard to get motivated sometimes.",
        ]
    }
}
