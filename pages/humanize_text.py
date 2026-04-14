import random
import re
import ssl
import warnings
import nltk
import spacy
import streamlit as st
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

warnings.filterwarnings("ignore", category=FutureWarning)

########################################
# Download needed NLTK resources
########################################
def download_nltk_resources():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = ['punkt', 'averaged_perceptron_tagger',
                 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger_eng']
    for r in resources:
        nltk.download(r, quiet=True)

download_nltk_resources()

########################################
# Prepare spaCy pipeline
########################################
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("spaCy en_core_web_sm model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

########################################
# Citation Regex
########################################
CITATION_REGEX = re.compile(
    r"\(\s*[A-Za-z&\-,\.\s]+(?:et al\.\s*)?,\s*\d{4}(?:,\s*(?:pp?\.\s*\d+(?:-\d+)?))?\s*\)"
)

########################################
# Helper: Word & Sentence Counts
########################################
def count_words(text):
    return len(word_tokenize(text))

def count_sentences(text):
    return len(sent_tokenize(text))

########################################
# Code Block Detection & Protection
########################################
# Regex patterns to detect code blocks (CSS, HTML, JS, Python, etc.)
CODE_BLOCK_PATTERNS = [
    # Fenced code blocks ```...```
    re.compile(r"```[\s\S]*?```", re.MULTILINE),
    # Inline code `...`
    re.compile(r"`[^`]+`"),
    # CSS-like patterns: selectors with { ... }
    re.compile(r"[a-zA-Z0-9\-_\.#\s,:>~\+\[\]=\*]+\{[^}]*\}", re.MULTILINE),
    # HTML tags with attributes
    re.compile(r"<[a-zA-Z][a-zA-Z0-9]*(?:\s+[a-zA-Z\-]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+))?)*\s*/?>", re.MULTILINE),
    # Closing HTML tags
    re.compile(r"</[a-zA-Z][a-zA-Z0-9]*\s*>"),
    # CSS @rules (media queries, keyframes, imports, etc.)
    re.compile(r"@[a-zA-Z\-]+\s*[^;{]*(?:\{[\s\S]*?\}|;)", re.MULTILINE),
]

# Heuristic: if a text block has many of these indicators, it's likely code
CODE_INDICATORS = [
    r"\{", r"\}", r";", r":", r"//", r"/\*", r"\*/",
    r"<[a-zA-Z]", r"</", r"=>", r"===", r"!==",
    r"function\s*\(", r"const\s+", r"let\s+", r"var\s+",
    r"import\s+", r"export\s+", r"class\s+\w+\s*\{",
    r"\.\w+\s*\{",  # CSS class selector
    r"#\w+\s*\{",   # CSS ID selector
    r"px\b", r"em\b", r"rem\b", r"%\s*;",  # CSS units
    r"background[-:]", r"color\s*:", r"margin\s*:", r"padding\s*:",
    r"display\s*:", r"font[-:]", r"border[-:]",
]


def is_code_block(text):
    """Detect if a text block is likely code (CSS, HTML, JS, Python, etc.)."""
    stripped = text.strip()
    if not stripped:
        return False

    # Quick check: fenced code blocks
    if stripped.startswith("```") and stripped.endswith("```"):
        return True

    # Count code indicators
    indicator_count = 0
    for pattern in CODE_INDICATORS:
        if re.search(pattern, stripped):
            indicator_count += 1

    # If 4+ code indicators found, treat as code
    if indicator_count >= 4:
        return True

    # Check ratio of special characters to words (code has more symbols)
    words = len(stripped.split())
    special_chars = len(re.findall(r"[{};:<>/=()]", stripped))
    if words > 0 and special_chars / words > 0.5:
        return True

    return False


def extract_code_blocks(text):
    """Extract code blocks and replace with placeholders."""
    code_map = {}
    result = text

    # First handle fenced code blocks (```...```)
    fenced_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    matches = list(fenced_pattern.finditer(result))
    for i, match in enumerate(reversed(matches)):  # reverse to preserve positions
        idx = len(matches) - 1 - i
        placeholder = f"[[CODE_BLOCK_{idx}]]"
        code_map[placeholder] = match.group(0)
        result = result[:match.start()] + placeholder + result[match.end():]

    return result, code_map


def restore_code_blocks(text, code_map):
    """Restore code blocks from placeholders."""
    result = text
    for placeholder, code in code_map.items():
        result = result.replace(placeholder, code)
    return result


########################################
# Grammar Correction via Anthropic API
########################################
########################################
# Shared grammar prompt (used by all engines)
########################################
GRAMMAR_PROMPT = (
    "You are a copy editor reviewing text that has been deliberately humanized "
    "to sound natural and conversational. Your ONLY job is to fix genuine "
    "grammatical errors while leaving the human style completely intact.\n\n"
    "STRICT RULES — violating these ruins the humanization:\n"
    "1. DO NOT change contractions (keep don't, it's, they're, etc.)\n"
    "2. DO NOT remove informal starters like 'And', 'But', 'So', 'Look,' etc.\n"
    "3. DO NOT replace casual words with formal ones (keep 'lots of', 'a bunch of', 'super', etc.)\n"
    "4. DO NOT remove em-dashes (—), parenthetical asides, or rhetorical questions\n"
    "5. DO NOT remove personal voice phrases like 'Honestly, I think' or 'If you ask me'\n"
    "6. DO NOT change sentence structure, clause order, or paragraph flow\n"
    "7. DO NOT add or remove any sentences, ideas, or content\n"
    "8. DO keep comma splices if they feel intentional and conversational\n"
    "9. ONLY fix: clear spelling mistakes, wrong verb tense, broken subject-verb agreement, "
    "missing/extra articles (a/an/the), and obvious punctuation errors.\n\n"
    "Return ONLY the corrected text. No explanations. No markdown. No quotes:\n\n"
)


def _split_prose_parts(text):
    """Split text into (prose, code) parts for safe grammar processing."""
    return re.split(r"(```[\s\S]*?```)", text)


def _is_skippable(part):
    """Return True if this part should NOT be grammar-corrected."""
    if not part.strip():
        return True
    if part.strip().startswith("```") and part.strip().endswith("```"):
        return True
    if is_code_block(part):
        return True
    return False


########################################
# Option 1 — LanguageTool (FREE, no key)
########################################
def fix_grammar_languagetool(text):
    """
    Fix grammar using the free LanguageTool public API.
    No API key or signup required.
    Rate limit: ~20 requests/min (plenty for normal use).
    """
    import requests as req_lib

    parts = _split_prose_parts(text)
    corrected_parts = []

    for part in parts:
        if _is_skippable(part):
            corrected_parts.append(part)
            continue

        try:
            response = req_lib.post(
                "https://api.languagetool.org/v2/check",
                data={"text": part, "language": "en-US"},
                timeout=30,
            )
            if response.status_code != 200:
                corrected_parts.append(part)
                continue

            matches = response.json().get("matches", [])
            if not matches:
                corrected_parts.append(part)
                continue

            # Apply replacements from end → start so offsets stay valid
            corrected = part
            for match in reversed(matches):
                replacements = match.get("replacements", [])
                if not replacements:
                    continue
                rule_id = match.get("rule", {}).get("id", "")
                # Skip style/tone rules — only apply grammar/spelling fixes
                rule_issue_type = match.get("rule", {}).get("issueType", "")
                if rule_issue_type in ("style", "typographical"):
                    continue
                # Skip whitespace rules that would remove intentional dashes
                if "WHITESPACE" in rule_id or "PUNCTUATION" in rule_id.upper():
                    continue

                offset = match["offset"]
                length = match["length"]
                best_replacement = replacements[0]["value"]
                corrected = corrected[:offset] + best_replacement + corrected[offset + length:]

            corrected_parts.append(corrected)

        except Exception:
            corrected_parts.append(part)

    return "".join(corrected_parts)


########################################
# Option 2 — Groq (FREE tier, needs key)
########################################
def fix_grammar_groq(text, api_key):
    """
    Fix grammar using Groq's free API (Llama 3.3 70B).
    Get a free key at: https://console.groq.com
    Free tier: ~14,400 requests/day.
    """
    import requests as req_lib

    parts = _split_prose_parts(text)
    corrected_parts = []

    for part in parts:
        if _is_skippable(part):
            corrected_parts.append(part)
            continue

        try:
            response = req_lib.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    "messages": [
                        {
                            "role": "system",
                            "content": GRAMMAR_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": part,
                        },
                    ],
                },
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                corrected_text = data["choices"][0]["message"]["content"].strip()
                corrected_parts.append(corrected_text if corrected_text else part)
            else:
                corrected_parts.append(part)

        except Exception:
            corrected_parts.append(part)

    return "".join(corrected_parts)


########################################
# Option 3 — Claude / Anthropic (PAID)
########################################
def fix_grammar_with_api(text, api_key):
    """
    Fix grammar using Anthropic Claude API (paid).
    Kept for backward compatibility.
    """
    import requests as req_lib

    parts = _split_prose_parts(text)
    corrected_parts = []

    for part in parts:
        if _is_skippable(part):
            corrected_parts.append(part)
            continue

        try:
            response = req_lib.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4096,
                    "messages": [
                        {
                            "role": "user",
                            "content": GRAMMAR_PROMPT + part,
                        }
                    ],
                },
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                corrected_text = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        corrected_text += block.get("text", "")
                corrected_parts.append(corrected_text.strip() if corrected_text.strip() else part)
            else:
                corrected_parts.append(part)
        except Exception:
            corrected_parts.append(part)

    return "".join(corrected_parts)


########################################
# Step 1: Extract & Restore Citations
########################################
def extract_citations(text):
    refs = CITATION_REGEX.findall(text)
    placeholder_map = {}
    replaced_text = text
    for i, r in enumerate(refs, start=1):
        placeholder = f"[[REF_{i}]]"
        placeholder_map[placeholder] = r
        replaced_text = replaced_text.replace(r, placeholder, 1)
    return replaced_text, placeholder_map

PLACEHOLDER_REGEX = re.compile(r"\[\s*\[\s*REF_(\d+)\s*\]\s*\]")


def restore_citations(text, placeholder_map):

    def replace_placeholder(match):
        # match.group(1) contains the numeric index captured from the placeholder
        idx = match.group(1)
        key = f"[[REF_{idx}]]"
        return placeholder_map.get(key, match.group(0))

    restored = PLACEHOLDER_REGEX.sub(replace_placeholder, text)
    return restored


########################################
# Step 2: Advanced Humanization Engine
# Designed to defeat AI content detectors by
# introducing natural human writing patterns.
########################################

# --- AI-typical phrases → natural human alternatives ---
# AI models heavily favour these formal/stiff constructions.
# Replacing them with casual equivalents is the single most
# effective signal change for AI-detection tools.
AI_PHRASE_REPLACEMENTS = {
    # Formal → casual connectors
    "Additionally,": "Also,",
    "Furthermore,": "Plus,",
    "Moreover,": "On top of that,",
    "Consequently,": "So,",
    "Nevertheless,": "Still,",
    "Nonetheless,": "Even so,",
    "In contrast,": "But then again,",
    "Hence,": "So,",
    "Therefore,": "That means",
    "Subsequently,": "After that,",
    "In conclusion,": "Bottom line:",
    "It is important to note that": "Worth knowing:",
    "It should be noted that": "Keep in mind,",
    "It is worth mentioning that": "One thing to note:",
    "In order to": "to",
    "due to the fact that": "because",
    "in the event that": "if",
    "a wide range of": "lots of",
    "a variety of": "different",
    "a number of": "several",
    "a significant amount of": "a lot of",
    "at the present time": "right now",
    "at this point in time": "right now",
    "for the purpose of": "to",
    "in light of the fact that": "since",
    "on the other hand": "then again",
    "as a matter of fact": "actually",
    "in the near future": "soon",
    "in terms of": "when it comes to",
    "with regard to": "about",
    "with respect to": "about",
    "in regard to": "about",
    "is able to": "can",
    "has the ability to": "can",
    "make a decision": "decide",
    "take into consideration": "consider",
    "give consideration to": "think about",
    "a considerable amount": "a lot",
    "on a regular basis": "regularly",
    "in a timely manner": "quickly",
    "whether or not": "whether",
    "each and every": "every",
    "first and foremost": "first off,",
    # AI-typical starters
    "This comprehensive": "This",
    "This innovative": "This",
    "This cutting-edge": "This",
    "It's important to understand that": "Here's the thing:",
    "It is essential to": "You need to",
    "It is crucial to": "You really should",
    "plays a crucial role": "matters a lot",
    "plays a vital role": "is pretty important",
    "provides a comprehensive": "gives you a full",
    "offers a comprehensive": "gives you a thorough",
    "a comprehensive overview": "the full picture",
    "designed to help": "built to help",
    "aims to provide": "gives you",
    "specifically designed": "built",
    "utilizing": "using",
    "utilize": "use",
    "utilization": "use",
    "implementing": "setting up",
    "implement": "set up",
    "facilitate": "help with",
    "facilitates": "helps with",
    "leverage": "use",
    "leveraging": "using",
    "optimal": "best",
    "optimizing": "improving",
    "optimize": "improve",
    "streamline": "simplify",
    "streamlined": "simplified",
    "robust": "solid",
    "scalable": "flexible",
    "seamless": "smooth",
    "seamlessly": "smoothly",
    "enhance": "boost",
    "enhancing": "boosting",
    "ensures": "makes sure",
    "ensure": "make sure",
    "empowers": "lets",
    "empower": "let",
    "prioritize": "focus on",
    "prioritizing": "focusing on",
    # Additional AI-heavy phrases
    "In today's fast-paced world": "These days",
    "In today's world": "These days",
    "In the modern world": "Nowadays",
    "In recent times": "Lately",
    "In the current era": "Right now",
    "With the rise of": "As",
    "With the advent of": "Now that we have",
    "In the age of": "Now that",
    "As technology continues to evolve": "As tech keeps changing",
    "As the world becomes more": "Now that things are getting more",
    "It goes without saying that": "Obviously,",
    "Needless to say,": "Clearly,",
    "Without a doubt,": "No question,",
    "There is no denying that": "Clearly,",
    "One cannot deny that": "It's hard to argue that",
    "It is widely acknowledged that": "Most people agree that",
    "It is generally accepted that": "Most people would say that",
    "It is worth noting that": "Worth mentioning:",
    "It is interesting to note that": "Interestingly,",
    "Studies have indicated that": "Research suggests that",
    "Research has demonstrated that": "Studies show that",
    "According to recent studies": "Recent research shows",
    "A growing body of research": "More and more research",
    "Experts in the field": "People who know this stuff",
    "Industry experts": "Experts in the space",
    "In conclusion,": "So,",
    "To conclude,": "To wrap up,",
    "In summary,": "To sum it all up,",
    "To summarize,": "In short,",
    "As a result,": "So,",
    "As a consequence,": "Because of that,",
    "This leads to": "This means",
    "This results in": "This causes",
    "This allows for": "This lets you",
    "This enables": "This lets",
    "This provides": "This gives you",
    "plays a key role": "is a big part of",
    "of paramount importance": "really important",
    "of utmost importance": "super important",
    "of great importance": "really important",
    "highly effective": "very effective",
    "highly efficient": "very efficient",
    "highly beneficial": "really useful",
    "greatly enhances": "really boosts",
    "significantly improves": "makes a real difference to",
    "positively impacts": "helps",
    "negatively impacts": "hurts",
    "has a profound impact": "has a huge effect",
    "has a significant impact": "makes a big difference",
}

# --- Formal → contracted forms (humans use contractions!) ---
HUMAN_CONTRACTIONS = {
    "it is": "it's",
    "It is": "It's",
    "do not": "don't",
    "Do not": "Don't",
    "does not": "doesn't",
    "Does not": "Doesn't",
    "did not": "didn't",
    "Did not": "Didn't",
    "will not": "won't",
    "Will not": "Won't",
    "can not": "can't",
    "Can not": "Can't",
    "cannot": "can't",
    "Cannot": "Can't",
    "could not": "couldn't",
    "Could not": "Couldn't",
    "should not": "shouldn't",
    "Should not": "Shouldn't",
    "would not": "wouldn't",
    "Would not": "Wouldn't",
    "is not": "isn't",
    "Is not": "Isn't",
    "are not": "aren't",
    "Are not": "Aren't",
    "has not": "hasn't",
    "Has not": "Hasn't",
    "have not": "haven't",
    "Have not": "Haven't",
    "had not": "hadn't",
    "Had not": "Hadn't",
    "we are": "we're",
    "We are": "We're",
    "they are": "they're",
    "They are": "They're",
    "you are": "you're",
    "You are": "You're",
    "I am": "I'm",
    "that is": "that's",
    "That is": "That's",
    "there is": "there's",
    "There is": "There's",
    "we have": "we've",
    "We have": "We've",
    "they have": "they've",
    "They have": "They've",
    "you have": "you've",
    "You have": "You've",
    "I have": "I've",
    "it will": "it'll",
    "It will": "It'll",
    "you will": "you'll",
    "You will": "You'll",
    "we will": "we'll",
    "We will": "We'll",
    "they will": "they'll",
    "They will": "They'll",
    "let us": "let's",
    "Let us": "Let's",
    "who is": "who's",
    "Who is": "Who's",
    "what is": "what's",
    "What is": "What's",
}

# --- Natural human sentence starters (casual, not academic) ---
HUMAN_TRANSITIONS = [
    "Look,",
    "Here's the deal:",
    "The thing is,",
    "Honestly,",
    "Truth is,",
    "Real talk:",
    "So basically,",
    "Just so you know,",
    "Quick note:",
    "Worth noting:",
    "To be fair,",
    "Full disclosure:",
    "Heads up:",
    "Fair warning:",
    "One more thing:",
    "And yes,",
    "Oh, and",
    "Also worth mentioning:",
    "Not gonna lie,",
    "For what it's worth,",
    "That said,",
    "Long story short,",
    "Big picture:",
    "Here's what matters:",
    "Key takeaway:",
    "Pro tip:",
    "Bottom line:",
]

# --- Filler / hedge words humans naturally sprinkle in ---
HEDGE_INSERTIONS = [
    "pretty much",
    "basically",
    "honestly",
    "arguably",
    "likely",
    "typically",
    "generally",
    "usually",
    "in most cases",
    "more or less",
    "roughly",
    "kind of",
    "sort of",
    "effectively",
]

# --- Sentence-ending softeners ---
SENTENCE_SOFTENERS = [
    " — which is a big deal.",
    " — and that matters.",
    " — so keep that in mind.",
    " — worth paying attention to.",
    " (seriously).",
    " — no joke.",
    " — and it shows.",
    " — plain and simple.",
    ".",  # keep original
    ".",
    ".",  # bias toward no change
    ".",
]

# --- AI-Overused Words (STRONGEST detection signals) ---
# Research shows AI detectors heavily weight these specific words.
# Replacing them is the single most effective technique.
AI_OVERUSED_WORDS = {
    "delve": ["explore", "dig into", "look at", "examine", "get into"],
    "delves": ["explores", "digs into", "looks at", "examines"],
    "delving": ["exploring", "digging into", "looking at", "examining"],
    "comprehensive": ["thorough", "complete", "full", "detailed", "in-depth"],
    "crucial": ["key", "important", "vital", "essential", "major"],
    "innovative": ["new", "fresh", "creative", "original", "novel"],
    "landscape": ["scene", "field", "area", "space", "world"],
    "paradigm": ["model", "approach", "framework", "way of thinking"],
    "multifaceted": ["complex", "varied", "diverse", "layered"],
    "nuanced": ["subtle", "detailed", "fine-grained", "complex"],
    "pivotal": ["key", "critical", "central", "decisive"],
    "groundbreaking": ["revolutionary", "game-changing", "pioneering", "major"],
    "transformative": ["life-changing", "game-changing", "radical", "major"],
    "holistic": ["complete", "whole", "all-around", "full-picture"],
    "synergy": ["teamwork", "collaboration", "combined effort"],
    "navigate": ["handle", "deal with", "work through", "figure out"],
    "foster": ["encourage", "support", "promote", "build", "grow"],
    "fostering": ["encouraging", "supporting", "promoting", "building"],
    "underscore": ["highlight", "stress", "show", "point out"],
    "underscores": ["highlights", "stresses", "shows", "points out"],
    "encompass": ["include", "cover", "span", "take in"],
    "encompasses": ["includes", "covers", "spans"],
    "realm": ["area", "field", "domain", "world", "space"],
    "intricate": ["complex", "detailed", "elaborate", "involved"],
    "interplay": ["interaction", "connection", "relationship", "dynamic"],
    "tapestry": ["mix", "blend", "collection", "mosaic"],
    "testament": ["proof", "evidence", "sign", "demonstration"],
    "embark": ["start", "begin", "set out", "kick off"],
    "embarking": ["starting", "beginning", "setting out"],
    "endeavor": ["effort", "attempt", "project", "work"],
    "commendable": ["impressive", "admirable", "great", "solid"],
    "noteworthy": ["notable", "interesting", "worth noting", "significant"],
    "meticulous": ["careful", "thorough", "precise", "detailed"],
    "meticulously": ["carefully", "thoroughly", "precisely"],
    "enlightening": ["informative", "eye-opening", "insightful", "revealing"],
    "aforementioned": ["mentioned", "above", "previous", "noted"],
    "harnessing": ["using", "tapping into", "making use of"],
    "harness": ["use", "tap into", "make use of"],
    "spearhead": ["lead", "drive", "head up", "champion"],
    "spearheading": ["leading", "driving", "heading up"],
    "bolster": ["strengthen", "support", "boost", "reinforce"],
    "bolstering": ["strengthening", "supporting", "boosting"],
    "augment": ["add to", "boost", "increase", "supplement"],
    "aligns": ["fits", "matches", "goes with", "works with"],
    "underpin": ["support", "back up", "form the basis of"],
    "underpins": ["supports", "backs up"],
    "necessitate": ["require", "call for", "need", "demand"],
    "necessitates": ["requires", "calls for", "needs"],
    "proliferation": ["spread", "growth", "increase", "rise"],
    "ubiquitous": ["everywhere", "widespread", "common"],
    "myriad": ["many", "countless", "tons of", "loads of"],
    "plethora": ["plenty", "lots", "abundance", "wealth"],
    "paramount": ["top", "most important", "key", "number one"],
    "indispensable": ["essential", "necessary", "vital", "must-have"],
    "burgeoning": ["growing", "expanding", "booming", "rising"],
    "adept": ["skilled", "good at", "capable"],
    "propensity": ["tendency", "inclination", "likelihood"],
    "amalgamation": ["mix", "blend", "combination", "fusion"],
    "elucidating": ["explaining", "clarifying", "making clear"],
    "elucidate": ["explain", "clarify", "make clear"],
    "discerning": ["sharp", "perceptive", "keen", "observant"],
    "pertinent": ["relevant", "related", "applicable", "on-point"],
    "juxtaposition": ["contrast", "comparison", "side-by-side"],
    "imperative": ["necessary", "essential", "needed", "critical"],
    "undeniably": ["clearly", "obviously", "without doubt"],
    "inherently": ["naturally", "by nature", "at its core"],
    "fundamentally": ["at its core", "basically", "at heart"],
    "exceedingly": ["very", "really", "extremely", "super"],
    # --- Extended AI vocabulary (additional high-signal words) ---
    "utilized": ["used", "applied", "put to use"],
    "utilizes": ["uses", "applies", "puts to use"],
    "obtained": ["got", "got hold of", "picked up"],
    "endeavors": ["efforts", "attempts", "tries"],
    "subsequently": ["then", "after that", "next"],
    "regarding": ["about", "on", "concerning"],
    "concerning": ["about", "on", "with"],
    "demonstrate": ["show", "prove", "make clear"],
    "demonstrates": ["shows", "proves", "makes clear"],
    "demonstrated": ["showed", "proved", "made clear"],
    "indicating": ["showing", "suggesting", "meaning"],
    "indicate": ["show", "suggest", "mean"],
    "indicates": ["shows", "suggests", "means"],
    "significant": ["big", "major", "notable", "real"],
    "significantly": ["a lot", "greatly", "noticeably", "quite a bit"],
    "substantial": ["big", "large", "considerable", "hefty"],
    "substantially": ["a lot", "considerably", "quite a bit"],
    "considerable": ["quite a bit of", "a good amount of", "solid"],
    "considerably": ["quite a bit", "a lot", "noticeably"],
    "numerous": ["many", "lots of", "plenty of", "a bunch of"],
    "various": ["different", "several", "all sorts of"],
    "certain": ["some", "specific", "particular"],
    "particular": ["specific", "certain", "exact"],
    "specific": ["exact", "particular", "precise"],
    "currently": ["right now", "at the moment", "these days", "now"],
    "previously": ["before", "earlier", "in the past"],
    "initially": ["at first", "to start", "originally"],
    "ultimately": ["in the end", "when all's said and done", "at the end of the day"],
    "essentially": ["basically", "at heart", "in practice"],
    "approximately": ["around", "about", "roughly", "give or take"],
    "primarily": ["mainly", "mostly", "for the most part"],
    "relatively": ["fairly", "pretty", "somewhat", "quite"],
    "extremely": ["very", "really", "super", "seriously"],
    "highly": ["very", "really", "quite", "pretty"],
    "rapidly": ["quickly", "fast", "in no time", "at speed"],
    "effectively": ["well", "successfully", "in practice"],
    "efficiently": ["well", "smoothly", "without wasting time"],
    "consistently": ["reliably", "steadily", "all the time"],
    "continuously": ["constantly", "all the time", "without stopping"],
    "simultaneously": ["at the same time", "at once", "together"],
    "accordingly": ["so", "because of that", "as a result"],
    "thereby": ["so", "and so", "which means"],
    "whereas": ["while", "but", "on the other hand"],
    "whilst": ["while", "as", "even as"],
    "thus": ["so", "therefore", "as a result"],
    "hence": ["so", "that's why", "which is why"],
    "consequently": ["so", "as a result", "that's why"],
    "nonetheless": ["still", "even so", "that said"],
    "nevertheless": ["still", "even so", "despite this"],
    "moreover": ["plus", "on top of that", "what's more"],
    "furthermore": ["also", "beyond that", "and"],
    "additionally": ["also", "on top of that", "plus"],
    "exemplifies": ["shows", "is a good example of", "illustrates"],
    "exemplify": ["show", "illustrate", "be a good example of"],
    "facilitate": ["help with", "make easier", "support"],
    "ascertain": ["find out", "figure out", "determine"],
    "ascertained": ["found out", "figured out", "determined"],
    "constitute": ["make up", "form", "be"],
    "constitutes": ["makes up", "forms", "is"],
    "prioritize": ["focus on", "put first", "tackle first"],
    "delineate": ["outline", "describe", "spell out"],
    "articulate": ["explain", "put into words", "express"],
    "articulates": ["explains", "puts into words", "expresses"],
    "encompasses": ["covers", "includes", "takes in"],
    "commence": ["start", "begin", "kick off"],
    "commences": ["starts", "begins", "kicks off"],
    "initiate": ["start", "begin", "kick off", "launch"],
    "terminates": ["ends", "stops", "finishes"],
    "terminate": ["end", "stop", "finish"],
    "endeavor": ["try", "attempt", "work", "effort"],
    "procurement": ["buying", "getting", "sourcing", "purchase"],
    "implementation": ["rollout", "setup", "putting into practice"],
    "functionality": ["features", "how it works", "what it does"],
    "methodology": ["approach", "method", "way of doing things"],
    "methodology": ["approach", "method", "way"],
    "parameters": ["limits", "settings", "boundaries", "rules"],
    "leverage": ["use", "make use of", "tap into"],
    "leverages": ["uses", "makes use of", "taps into"],
    "optimize": ["improve", "fine-tune", "get the most out of"],
    "optimizes": ["improves", "fine-tunes"],
    "enhance": ["boost", "improve", "step up"],
    "enhanced": ["improved", "boosted", "upgraded"],
    "innovative": ["new", "fresh", "different", "creative"],
    "cutting-edge": ["latest", "brand-new", "modern", "current"],
    "state-of-the-art": ["top-of-the-line", "latest", "best available"],
    "best-in-class": ["top", "leading", "best around"],
    "world-class": ["top-tier", "excellent", "top-notch"],
    "robust": ["solid", "strong", "reliable", "tough"],
    "seamless": ["smooth", "effortless", "easy"],
    "scalable": ["flexible", "adaptable", "can grow with you"],
    "intuitive": ["easy to use", "user-friendly", "simple"],
    "versatile": ["flexible", "adaptable", "all-purpose"],
    "dynamic": ["lively", "active", "fast-moving"],
}

# --- Rhetorical questions (humans naturally ask these, AI never does) ---
RHETORICAL_QUESTIONS = [
    "But why does this matter?",
    "So what does this actually mean?",
    "And where does that leave us?",
    "What's the takeaway here?",
    "Sound familiar?",
    "Makes sense, right?",
    "But is that really the case?",
    "Why should you care?",
]

# --- Parenthetical asides (humans do this constantly) ---
PARENTHETICAL_ASIDES = [
    " (and this is important)",
    " (believe it or not)",
    " (which makes sense if you think about it)",
    " (at least in theory)",
    " (for better or worse)",
    " (not surprisingly)",
    " (to put it simply)",
    " (as you might expect)",
]

# --- Short punchy sentences (creates burstiness AI lacks) ---
SHORT_INTERJECTIONS = [
    "That's huge.",
    "It works.",
    "Simple as that.",
    "Think about it.",
    "No surprise there.",
    "Makes sense.",
    "And it shows.",
    "Not ideal.",
    "Worth it.",
    "Exactly.",
    "Pretty clear.",
    "Bottom line.",
    "That's the key.",
    "Can't argue with that.",
]

# --- AI-Overused PHRASES common in reviews/financial content ---
# These multi-word patterns are dead giveaways.
AI_OVERUSED_PHRASES = {
    "it is important to note that": ["worth knowing:", "keep in mind,", "here's the thing:"],
    "it should be noted that": ["keep in mind,", "heads up:", "one thing:"],
    "it is worth noting that": ["one thing to note:", "quick note:", "worth mentioning:"],
    "it is essential to understand": ["you need to know", "here's what matters:"],
    "play a significant role in": ["matter a lot for", "are a big deal for"],
    "in today's digital landscape": ["these days", "right now", "nowadays"],
    "in the ever-evolving world of": ["in the fast-moving world of", "as things keep changing in"],
    "this is particularly important": ["this really matters", "pay attention to this"],
    "it goes without saying": ["obviously", "clearly", "needless to say"],
    "at the end of the day": ["ultimately", "when it comes down to it", "all things considered"],
    "in a nutshell": ["basically", "long story short", "to sum it up"],
    "offers a wide range of": ["has lots of", "comes with plenty of", "gives you a bunch of"],
    "provides a comprehensive": ["gives you a full", "delivers a complete", "offers a thorough"],
    "boasts an impressive": ["comes with a solid", "has a strong", "packs a nice"],
    "stands out as": ["really shines as", "is clearly", "works great as"],
    "on the other hand": ["but then again", "flip side though", "that said"],
    "when it comes to": ["for", "with", "talking about"],
    "in addition to": ["on top of", "plus", "beyond"],
    "as a result": ["so", "because of that", "that's why"],
    "in particular": ["especially", "mainly", "particularly"],
    "for instance": ["like", "say", "take this example:"],
    "to summarize": ["so basically", "long story short", "in short"],
    "it can be concluded that": ["bottom line:", "the takeaway is", "what this means:"],
    "significantly impacts": ["really affects", "has a big effect on", "changes"],
    "a wide array of": ["lots of", "plenty of", "all sorts of"],
    "cater to the needs of": ["work well for", "fit", "help out"],
    "the bottom line is": ["here's the deal:", "what it comes down to:"],
}

# --- Human sentence starters ("And", "But", "So" — AI almost never starts with these) ---
HUMAN_SENTENCE_STARTERS = [
    "And ",
    "But ",
    "So ",
    "Or ",
    "Plus, ",
    "Now, ",
    "Sure, ",
    "Yeah, ",
    "Granted, ",
    "True, ",
]

# --- Idiomatic replacements (formal phrase → natural idiom) ---
# AI almost never uses idioms; inserting them dramatically raises perplexity.
IDIOM_REPLACEMENTS = {
    "is very important": "really matters",
    "are very important": "really matter",
    "has a significant impact": "makes a real difference",
    "have a significant impact": "make a real difference",
    "is a good option": "is worth a shot",
    "are a good option": "are worth a shot",
    "should be considered": "deserves a look",
    "can be challenging": "isn't always easy",
    "it is not easy": "it's no walk in the park",
    "requires effort": "takes some work",
    "requires a lot of effort": "takes a lot of work",
    "has many benefits": "has a lot going for it",
    "have many benefits": "have a lot going for them",
    "can be difficult to understand": "can be tricky to wrap your head around",
    "is easy to understand": "is pretty straightforward",
    "are easy to understand": "are pretty straightforward",
    "is widely used": "gets used everywhere",
    "are widely used": "get used everywhere",
    "has been proven": "has been shown to work",
    "have been proven": "have been shown to work",
    "plays an important role": "plays a big part",
    "play an important role": "play a big part",
    "in recent years": "lately",
    "over the past few years": "in the last couple of years",
    "it is clear that": "clearly,",
    "it is obvious that": "obviously,",
    "it is evident that": "it's plain to see that",
    "research has shown": "studies back this up",
    "studies have shown": "research confirms",
    "experts agree": "most people in the field agree",
    "according to experts": "if you ask the pros",
    "the majority of": "most",
    "a large number of": "a lot of",
    "in the field of": "in",
    "the field of": "",
    "across different industries": "in all sorts of industries",
    "across various industries": "across many different industries",
}

# --- Personal voice inserts (first-person commentary humans naturally add) ---
# AI text never includes personal opinions or direct-address commentary.
PERSONAL_VOICE_INSERTS = [
    "Honestly, I think",
    "In my experience,",
    "If you ask me,",
    "From what I've seen,",
    "Speaking from experience,",
    "I'll be honest —",
    "Here's my take:",
    "Between you and me,",
    "I've found that",
    "What I've noticed is",
]

# --- Natural self-corrections (humans rephrase mid-thought — AI never does) ---
# These create highly unpredictable token sequences that spike perplexity scores.
NATURAL_CORRECTIONS = [
    " — or, well, more accurately,",
    " — actually, let me rephrase that —",
    ", which is to say,",
    " — or rather,",
    " — or, more precisely,",
    ", or maybe a better way to put it,",
]

# --- Concessive openers (humans acknowledge the other side naturally) ---
CONCESSIVE_OPENERS = [
    "Sure, it's not perfect, but",
    "Look, nothing is flawless —",
    "To be fair,",
    "Granted, there are downsides, but",
    "Yes, there are trade-offs, but",
    "It's not for everyone, but",
    "Of course, your mileage may vary, but",
]


def replace_ai_phrases(text):
    """Replace AI-typical formal phrases with casual human alternatives."""
    result = text
    # Sort by length descending so longer phrases match first
    sorted_phrases = sorted(AI_PHRASE_REPLACEMENTS.keys(), key=len, reverse=True)
    for ai_phrase in sorted_phrases:
        human_phrase = AI_PHRASE_REPLACEMENTS[ai_phrase]
        # Case-insensitive replacement, preserving one match at a time
        pattern = re.compile(re.escape(ai_phrase), re.IGNORECASE)
        result = pattern.sub(human_phrase, result)
    return result


def add_contractions(text):
    """Convert formal expanded forms to contractions (humans contract!)."""
    result = text
    # Sort by length descending to match longer phrases first
    sorted_forms = sorted(HUMAN_CONTRACTIONS.items(), key=lambda x: len(x[0]), reverse=True)
    for formal, contracted in sorted_forms:
        # Use word boundary matching to avoid partial replacements
        pattern = re.compile(r'\b' + re.escape(formal) + r'\b')
        result = pattern.sub(contracted, result)
    return result


def vary_sentence_structure(sentences, p_split=0.15, p_merge=0.10):
    """Vary sentence lengths for natural 'burstiness'.

    AI text tends to have uniform sentence lengths. Human text has a mix
    of short punchy sentences and longer flowing ones. This function
    randomly splits long sentences and merges short adjacent ones.
    """
    result = []
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        words = sent.split()
        word_count = len(words)

        # Skip sentences that contain placeholders (protect structure)
        if "[[REF_" in sent or "[[CODE_BLOCK_" in sent:
            result.append(sent)
            i += 1
            continue

        # SPLIT: long sentences (>20 words) can be split at conjunctions
        if word_count > 20 and random.random() < p_split:
            split_points = []
            for j, w in enumerate(words):
                if w.lower().rstrip(',') in ('and', 'but', 'which', 'while', 'because', 'since', 'although', 'however'):
                    if 8 < j < word_count - 5:
                        split_points.append(j)
            if split_points:
                sp = random.choice(split_points)
                first_half = " ".join(words[:sp]).rstrip(',').rstrip()
                second_half = " ".join(words[sp:])
                # Capitalize second half
                if second_half and second_half[0].islower():
                    second_half = second_half[0].upper() + second_half[1:]
                if not first_half.endswith('.'):
                    first_half += '.'
                result.append(first_half)
                result.append(second_half)
                i += 1
                continue

        # MERGE: short adjacent sentences (<10 words each) can be merged
        if (word_count < 10 and i + 1 < len(sentences)
                and len(sentences[i + 1].split()) < 10
                and random.random() < p_merge
                and "[[REF_" not in sentences[i + 1]):
            merged = sent.rstrip('.!?,;') + " — " + sentences[i + 1]
            result.append(merged)
            i += 2
            continue

        result.append(sent)
        i += 1

    return result


def add_human_transitions(sentences, p_transition=0.08):
    """Sparingly add natural human sentence starters.

    Unlike the old academic transitions ('Moreover,', 'Furthermore,')
    these sound like a real person writing. Probability is kept LOW
    to avoid over-doing it.
    """
    result = []
    for i, sent in enumerate(sentences):
        if "[[REF_" in sent:
            result.append(sent)
            continue
        # Only apply to sentences after the first one
        if i > 0 and random.random() < p_transition:
            transition = random.choice(HUMAN_TRANSITIONS)
            # Don't double-add if sentence already starts with a transition
            if not any(sent.startswith(t.split()[0]) for t in HUMAN_TRANSITIONS):
                sent = f"{transition} {sent[0].lower()}{sent[1:]}" if sent else sent
        result.append(sent)
    return result


def add_hedge_words(text, p_hedge=0.05):
    """Insert natural hedge words that humans use but AI rarely does.

    Very low probability to avoid making every sentence hedgy.
    """
    if "[[REF_" in text:
        return text

    words = text.split()
    if len(words) < 6:
        return text

    if random.random() < p_hedge:
        hedge = random.choice(HEDGE_INSERTIONS)
        # Insert after a verb or at a natural break point
        insert_pos = random.randint(2, min(5, len(words) - 1))
        words.insert(insert_pos, hedge)
        return " ".join(words)

    return text


def soften_sentence_endings(text, p_soften=0.05):
    """Occasionally replace a period with a more natural ending.

    Humans sometimes add asides or emphatic endings.
    """
    if "[[REF_" in text:
        return text
    if text.endswith('.') and random.random() < p_soften and len(text.split()) > 8:
        softener = random.choice(SENTENCE_SOFTENERS)
        return text[:-1] + softener
    return text


########################################
# NEW: Advanced Anti-Detection Engine
# These functions specifically target the
# patterns AI detectors look for.
########################################

def replace_ai_overused_words(text):
    """Replace words that AI models consistently overuse.

    This is the SINGLE most effective technique for reducing AI detection.
    Detectors heavily weight these specific vocabulary choices.
    """
    if "[[REF_" in text or "[[CODE_BLOCK_" in text:
        return text

    result = text
    for ai_word, replacements in AI_OVERUSED_WORDS.items():
        pattern = re.compile(r'\b' + re.escape(ai_word) + r'\b', re.IGNORECASE)
        matches = list(pattern.finditer(result))
        for match in reversed(matches):  # Reverse to preserve positions
            replacement = random.choice(replacements)
            original = match.group(0)
            if original[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            result = result[:match.start()] + replacement + result[match.end():]
    return result


def convert_passive_to_active(sentence):
    """Convert passive voice to active voice using spaCy dependency parsing.

    AI text heavily uses passive constructions ('was analyzed by', 'is used by').
    Converting some to active voice is a strong anti-detection signal.
    """
    if not nlp or "[[REF_" in sentence or "[[CODE_BLOCK_" in sentence:
        return sentence

    doc = nlp(sentence)

    # Detect passive voice: look for nsubjpass (passive nominal subject)
    has_passive = any(token.dep_ == "nsubjpass" for token in doc)

    if not has_passive:
        return sentence

    # Only convert with 60% probability to maintain a natural mix
    if random.random() > 0.6:
        return sentence

    # Find the passive subject, aux verb, main verb, and agent
    passive_subj = None
    main_verb = None
    agent = None
    aux_verb = None

    for token in doc:
        if token.dep_ == "nsubjpass":
            passive_subj = token
            main_verb = token.head
        if token.dep_ == "agent":  # the "by" phrase
            for child in token.children:
                if child.dep_ == "pobj":
                    agent = child
        if token.dep_ == "auxpass":
            aux_verb = token

    # Only convert when we have all three parts (subject, verb, agent)
    if passive_subj and main_verb and agent:
        agent_span = doc[agent.left_edge.i:agent.right_edge.i + 1].text
        subj_span = doc[passive_subj.left_edge.i:passive_subj.right_edge.i + 1].text

        # Use simple past tense of the verb
        verb_lemma = main_verb.lemma_
        if verb_lemma.endswith('e'):
            active_verb = verb_lemma + 'd'
        else:
            active_verb = verb_lemma + 'ed'

        # Reconstruct: "The agent active-verb the subject"
        active = f"{agent_span} {active_verb} {subj_span}"
        active = active[0].upper() + active[1:]
        if not active.endswith('.'):
            active += '.'

        return active

    return sentence


def diversify_sentence_starters(sentences):
    """Fix repetitive sentence openers.

    AI text often starts 3+ consecutive sentences with 'The', 'This', 'It'.
    Detectors specifically check for this uniformity pattern.
    """
    if len(sentences) < 3:
        return sentences

    result = list(sentences)

    OPENER_ALTERNATIVES = {
        "The": ["This particular", "That", "One", "A key", "An important"],
        "This": ["That", "The", "Such a", "One such"],
        "It": ["That", "The result", "What we see", "The outcome"],
        "These": ["Such", "Those", "All of these", "The"],
        "There": ["We find", "You'll notice", "What stands out"],
        "They": ["The team", "Those involved", "People", "Researchers"],
        "In": ["Within", "Across", "Throughout", "When looking at"],
    }

    for i in range(2, len(result)):
        if "[[REF_" in result[i] or "[[CODE_BLOCK_" in result[i]:
            continue

        words_i = result[i].split()
        words_prev = result[i-1].split()
        words_prev2 = result[i-2].split()

        if not words_i or not words_prev or not words_prev2:
            continue

        first_word = words_i[0].rstrip('.,;:')
        if (words_prev[0].rstrip('.,;:') == first_word and
                words_prev2[0].rstrip('.,;:') == first_word):

            if first_word in OPENER_ALTERNATIVES:
                new_opener = random.choice(OPENER_ALTERNATIVES[first_word])
                words_i[0] = new_opener
                result[i] = " ".join(words_i)

    return result


def inject_rhetorical_devices(sentences, p_question=0.04, p_aside=0.05, p_short=0.06):
    """Add rhetorical questions, parenthetical asides, and short interjections.

    These are hallmarks of natural human writing that AI almost never produces.
    Probabilities are kept LOW to avoid over-doing it.
    """
    result = []

    for i, sent in enumerate(sentences):
        if "[[REF_" in sent or "[[CODE_BLOCK_" in sent:
            result.append(sent)
            continue

        # Occasionally insert a rhetorical question after a statement
        if (i > 0 and i < len(sentences) - 1
                and random.random() < p_question
                and not sent.endswith('?')):
            result.append(sent)
            result.append(random.choice(RHETORICAL_QUESTIONS))
            continue

        # Occasionally add a parenthetical aside mid-sentence
        if random.random() < p_aside and len(sent.split()) > 12:
            words = sent.split()
            insert_pos = len(words) // 2
            aside = random.choice(PARENTHETICAL_ASIDES)
            words.insert(insert_pos, aside)
            result.append(" ".join(words))
            continue

        # Occasionally insert a short punchy interjection after a long sentence
        if (random.random() < p_short
                and len(sent.split()) > 15
                and i < len(sentences) - 1):
            result.append(sent)
            result.append(random.choice(SHORT_INTERJECTIONS))
            continue

        result.append(sent)

    return result


def add_natural_imperfections(text, p_dash=0.08):
    """Add natural punctuation that humans use but AI doesn't.

    Humans frequently use em-dashes and informal punctuation.
    These are subtle but effective anti-detection signals.
    """
    if "[[REF_" in text or "[[CODE_BLOCK_" in text:
        return text

    words = text.split()
    if len(words) < 8:
        return text

    # Occasionally replace a comma with an em-dash
    if random.random() < p_dash and ', ' in text:
        comma_positions = [i for i, c in enumerate(text) if c == ',' and i > 5]
        if comma_positions:
            pos = random.choice(comma_positions)
            text = text[:pos] + ' —' + text[pos+1:]

    return text


def replace_ai_overused_phrases(text):
    """Replace multi-word AI-typical phrases with casual human alternatives.

    These phrase-level patterns are harder for AI to avoid and
    easier for detectors to catch than single words.
    """
    if "[[REF_" in text or "[[CODE_BLOCK_" in text:
        return text

    result = text
    # Sort by length descending so longer phrases match first
    sorted_phrases = sorted(AI_OVERUSED_PHRASES.keys(), key=len, reverse=True)
    for ai_phrase in sorted_phrases:
        human_alts = AI_OVERUSED_PHRASES[ai_phrase]
        pattern = re.compile(re.escape(ai_phrase), re.IGNORECASE)
        if pattern.search(result):
            replacement = random.choice(human_alts)
            result = pattern.sub(replacement, result, count=1)
    return result


def boost_perplexity(sentences, p_starter=0.10, p_filler=0.08):
    """Boost text perplexity by adding patterns AI almost never generates.

    AI detectors measure perplexity (how predictable each word is).
    Starting sentences with 'And', 'But', 'So' and inserting natural
    fillers dramatically increases perplexity scores.
    """
    result = []
    for i, sent in enumerate(sentences):
        if "[[REF_" in sent or "[[CODE_BLOCK_" in sent:
            result.append(sent)
            continue

        words = sent.split()
        if not words:
            result.append(sent)
            continue

        # Start some sentences with And/But/So (AI almost never does this)
        if (i > 0 and random.random() < p_starter
                and words[0] not in ['And', 'But', 'So', 'Or', 'Plus,', 'Now,', 'Sure,', 'Yeah,', 'Granted,', 'True,']
                and not any(sent.startswith(t.split()[0]) for t in HUMAN_TRANSITIONS)):
            starter = random.choice(HUMAN_SENTENCE_STARTERS)
            # Lowercase the original first word when prepending
            if words[0][0].isupper() and words[0] not in ['I', 'I\'m', 'I\'ve', 'I\'ll', 'I\'d']:
                words[0] = words[0][0].lower() + words[0][1:]
            sent = starter + " ".join(words)

        # Occasionally insert a natural filler word
        words = sent.split()
        if random.random() < p_filler and len(words) > 6:
            fillers = ["actually", "really", "honestly", "clearly", "obviously", "basically", "definitely"]
            filler = random.choice(fillers)
            insert_pos = random.randint(1, min(4, len(words) - 1))
            words.insert(insert_pos, filler)
            sent = " ".join(words)

        result.append(sent)
    return result


def replace_synonyms(sentence, p_syn=0.2):
    """Replace words with natural synonyms using spaCy + WordNet.

    Improved version: filters out obscure/archaic synonyms and prefers
    common, natural-sounding alternatives.
    """
    if not nlp:
        return sentence

    doc = nlp(sentence)
    new_tokens = []
    for token in doc:
        if "[[REF_" in token.text:
            new_tokens.append(token.text)
            continue
        if (token.pos_ in ["ADJ", "VERB", "ADV"]
                and len(token.text) > 3
                and wordnet.synsets(token.text)
                and random.random() < p_syn):
            synonyms = get_filtered_synonyms(token.text, token.pos_)
            if synonyms:
                new_tokens.append(random.choice(synonyms))
            else:
                new_tokens.append(token.text)
        else:
            new_tokens.append(token.text)
    return " ".join(new_tokens)


def get_filtered_synonyms(word, pos):
    """Get synonyms filtered for naturalness.

    Only returns common, natural-sounding alternatives —
    filters out obscure, archaic, or overly technical words.
    """
    wn_pos = None
    if pos.startswith("ADJ"):
        wn_pos = wordnet.ADJ
    elif pos.startswith("NOUN"):
        wn_pos = wordnet.NOUN
    elif pos.startswith("ADV"):
        wn_pos = wordnet.ADV
    elif pos.startswith("VERB"):
        wn_pos = wordnet.VERB

    synonyms = set()
    if wn_pos:
        for syn in wordnet.synsets(word, pos=wn_pos)[:3]:  # Only top 3 synsets
            for lemma in syn.lemmas()[:3]:  # Only top 3 lemmas per synset
                lemma_name = lemma.name().replace("_", " ")
                # Filter: skip if same word, too short, has spaces, or is rare
                if (lemma_name.lower() != word.lower()
                        and len(lemma_name) > 2
                        and " " not in lemma_name
                        and lemma.count() > 0):  # Has usage frequency
                    synonyms.add(lemma_name)
    return list(synonyms)


def shuffle_clause_order(sentence, p_shuffle=0.10):
    """Occasionally rearrange clauses in a sentence.

    AI tends to present information in a very predictable order.
    Humans sometimes lead with the outcome or the qualifier.
    """
    if "[[REF_" in sentence:
        return sentence
    if random.random() > p_shuffle:
        return sentence

    # Look for comma-separated clauses
    parts = sentence.split(', ')
    if len(parts) == 2 and len(parts[0].split()) > 3 and len(parts[1].split()) > 3:
        # Swap the two clauses
        second = parts[1]
        if second and second[0].islower():
            second = second[0].upper() + second[1:]
        first = parts[0]
        if first and first[0].isupper():
            first = first[0].lower() + first[1:]
        return f"{second}, {first}"

    return sentence


def replace_idioms(text):
    """Replace formal phrases with idiomatic expressions.

    AI text is almost entirely idiom-free. CopyLeaks perplexity scoring
    spikes when it encounters natural idioms because they are genuinely
    unpredictable token sequences.
    """
    if "[[REF_" in text or "[[CODE_BLOCK_" in text:
        return text

    result = text
    sorted_idioms = sorted(IDIOM_REPLACEMENTS.keys(), key=len, reverse=True)
    for phrase in sorted_idioms:
        replacement = IDIOM_REPLACEMENTS[phrase]
        if not replacement:
            continue
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub(replacement, result, count=1)
    return result


def inject_personal_voice(sentences, p_voice=0.04):
    """Occasionally prepend a first-person commentary phrase.

    AI-generated text never expresses personal opinions or speaks directly
    to the reader. Adding a small number of these inserts is one of the
    strongest signals of human authorship that CopyLeaks checks.
    """
    result = []
    for i, sent in enumerate(sentences):
        if "[[REF_" in sent or "[[CODE_BLOCK_" in sent:
            result.append(sent)
            continue

        # Only apply to mid-document sentences, not the opener or closer
        if (i > 1 and i < len(sentences) - 1
                and random.random() < p_voice
                and len(sent.split()) > 8):
            voice_insert = random.choice(PERSONAL_VOICE_INSERTS)
            # Lowercase the sentence opener after inserting the voice phrase
            lowered = sent[0].lower() + sent[1:] if sent else sent
            sent = f"{voice_insert} {lowered}"
        result.append(sent)
    return result


def add_natural_corrections(sentences, p_correction=0.03):
    """Insert mid-sentence self-correction phrases.

    Humans regularly pause and rephrase: 'or rather', 'more accurately', etc.
    These patterns create highly unpredictable token sequences that push
    perplexity scores well above the AI-detection threshold.
    """
    result = []
    for sent in sentences:
        if "[[REF_" in sent or "[[CODE_BLOCK_" in sent:
            result.append(sent)
            continue

        words = sent.split()
        # Only apply to longer sentences
        if (len(words) > 12 and random.random() < p_correction):
            correction = random.choice(NATURAL_CORRECTIONS)
            # Insert at roughly mid-point of sentence
            mid = len(words) // 2
            # Find a natural insertion point near the middle (after a verb/noun)
            insert_at = mid
            words_lower = [w.lower() for w in words]
            # Prefer inserting after a comma or before a conjunction
            for offset in range(-3, 4):
                pos = mid + offset
                if 0 < pos < len(words) - 1:
                    if words[pos].endswith(',') or words[pos].lower() in ('which', 'that', 'and', 'but', 'as'):
                        insert_at = pos + 1
                        break
            words.insert(insert_at, correction.strip() + " ")
            result.append(" ".join(words))
        else:
            result.append(sent)
    return result


def add_concessive_opener(sentences, p_concede=0.03):
    """Occasionally start a paragraph with a concessive phrase.

    Acknowledging trade-offs or downsides is a strong human-authorship
    signal. AI almost always presents only the positive or neutral side.
    """
    result = []
    for i, sent in enumerate(sentences):
        if "[[REF_" in sent or "[[CODE_BLOCK_" in sent:
            result.append(sent)
            continue

        if (i > 0 and random.random() < p_concede
                and len(sent.split()) > 10):
            opener = random.choice(CONCESSIVE_OPENERS)
            lowered = sent[0].lower() + sent[1:] if sent else sent
            result.append(f"{opener} {lowered}")
        else:
            result.append(sent)
    return result


########################################
# Step 3: Combined Humanization Pipeline
########################################
def minimal_humanize_line(line, p_syn=0.2, p_trans=0.2):
    """Apply the full humanization pipeline to a single sentence.

    Pipeline order is critical — detection-heavy patterns are removed
    first, then structural changes, then vocabulary variation,
    then human-like finishing touches.

    ALL probabilities are now slider-responsive via p_syn and p_trans.
    """
    # Phase 1: Kill AI-specific vocabulary (HIGHEST IMPACT on detection)
    line = replace_ai_overused_words(line)
    line = replace_ai_overused_phrases(line)
    line = replace_ai_phrases(line)
    line = add_contractions(line)

    # Phase 1b: Idiom replacement — spikes perplexity with unpredictable tokens
    line = replace_idioms(line)

    # Phase 2: Structural changes (slider-responsive)
    line = convert_passive_to_active(line)
    line = shuffle_clause_order(line, p_shuffle=0.10 + p_trans * 0.18)

    # Phase 3: Vocabulary variation (slider-responsive)
    line = replace_synonyms(line, p_syn=p_syn)

    # Phase 4: Human finishing touches (ALL slider-responsive now)
    line = add_hedge_words(line, p_hedge=0.06 + p_trans * 0.32)
    line = soften_sentence_endings(line, p_soften=0.04 + p_trans * 0.28)
    line = add_natural_imperfections(line, p_dash=0.06 + p_trans * 0.38)

    return line


def minimal_rewriting(text, p_syn=0.2, p_trans=0.2):
    """Full humanization: sentence-level rewriting with structural variation.

    Enhanced pipeline with 10 anti-detection layers targeting CopyLeaks
    perplexity and burstiness scoring.
    At default (0.3): solid transformation (~30-40% detection).
    At max (1.0): very aggressive, targeting ~15% detection.
    """
    sentences = sent_tokenize(text)

    # Layer 1: Per-sentence humanization (vocabulary + idioms + structure)
    humanized = [
        minimal_humanize_line(s, p_syn=p_syn, p_trans=p_trans)
        for s in sentences
    ]

    # Layer 2: Vary sentence structure — aggressive burstiness
    humanized = vary_sentence_structure(
        humanized,
        p_split=0.15 + p_trans * 0.28,   # 0.24 at 0.3 → 0.43 at 1.0
        p_merge=0.08 + p_trans * 0.20    # 0.14 at 0.3 → 0.28 at 1.0
    )

    # Layer 3: Fix repetitive sentence starters (AI detector signal)
    humanized = diversify_sentence_starters(humanized)

    # Layer 4: Boost perplexity (And/But/So starters + fillers)
    humanized = boost_perplexity(
        humanized,
        p_starter=0.05 + p_trans * 0.38,  # 0.164 at 0.3 → 0.43 at 1.0
        p_filler=0.04 + p_trans * 0.28    # 0.124 at 0.3 → 0.32 at 1.0
    )

    # Layer 5: Add natural transitions (very sparingly)
    humanized = add_human_transitions(humanized, p_transition=0.05 + p_trans * 0.38)

    # Layer 6: Inject rhetorical devices (questions, asides, interjections)
    humanized = inject_rhetorical_devices(
        humanized,
        p_question=0.03 + p_trans * 0.22,
        p_aside=0.03 + p_trans * 0.22,
        p_short=0.04 + p_trans * 0.28
    )

    # Layer 7: Inject personal voice commentary (strong human-authorship signal)
    humanized = inject_personal_voice(
        humanized,
        p_voice=0.02 + p_trans * 0.06   # 0.038 at 0.3 → 0.08 at 1.0
    )

    # Layer 8: Natural mid-sentence self-corrections (perplexity spike)
    humanized = add_natural_corrections(
        humanized,
        p_correction=0.02 + p_trans * 0.04
    )

    # Layer 9: Concessive openers (acknowledging trade-offs — very human)
    humanized = add_concessive_opener(
        humanized,
        p_concede=0.02 + p_trans * 0.04
    )

    return " ".join(humanized)


def preserve_linebreaks_rewrite(text, p_syn=0.2, p_trans=0.2):
    """Rewrite text while preserving original line breaks.

    Splits the input on newline characters and rewrites each non-empty line
    independently, keeping blank lines and original line structure.
    """
    lines = text.splitlines()
    out_lines = []
    for ln in lines:
        if not ln.strip():
            out_lines.append("")
        else:
            out_lines.append(minimal_rewriting(
                ln, p_syn=p_syn, p_trans=p_trans))
    # Rejoin using single newline to preserve original paragraph/line breaks
    return "\n".join(out_lines)


########################################
# Final: Show Humanize Page
########################################
def show_humanize_page():
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Main", type="secondary"):
            st.session_state["current_page"] = "Main"
            st.rerun()
    with col2:
        if st.button("Switch to PDF Detection →", type="secondary"):
            st.session_state["current_page"] = "PDF Detection & Annotation"
            st.rerun()
    
    st.title("✍️ AI Text Humanizer & Enhancer")

    st.markdown("""
    ### Transform AI-Generated Text into Natural, Human-Like Content
    
    Our advanced text humanization tool intelligently rewrites AI-generated content to sound more natural, 
    authentic, and human-written while preserving your original meaning and academic integrity. Perfect for 
    refining articles, essays, reports, and any content that needs a more personal touch.
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 🛡️ **Smart Citation Protection**
        - **APA citation preservation** - automatically detects and protects academic references
        - **No data loss** - your citations remain intact and properly formatted
        - **Academic integrity** - maintain proper referencing while enhancing text
        - **Multiple citation styles** - handles various academic formatting standards
        """)

    with col2:
        st.markdown("""
        #### 🔧 **AI Detection Evasion**
        - **AI phrase replacement** - replaces formal AI patterns with casual human alternatives
        - **Natural contractions** - adds contractions like real humans write
        - **Sentence burstiness** - varies sentence lengths to break AI uniformity
        - **HTML/Shortcode-aware** - protects WordPress shortcodes and HTML tags
        """)

    with col3:
        st.markdown("""
        #### 📊 **Customizable Processing**
        - **Adjustable intensity** - control how much transformation is applied
        - **Real-time preview** - see word and sentence count changes
        - **Grammar correction** - fix grammar without losing humanized style
        - **Batch processing** - handle large documents efficiently
        """)

    st.markdown("---")

    st.markdown("""
    ### 🎯 **Ideal For:**
    - **Students & Researchers** - enhancing academic papers while keeping citations
    - **Content Creators** - making AI-generated articles sound more authentic
    - **Business Professionals** - refining reports and presentations
    - **Writers & Editors** - improving flow and readability of draft content
    - **Marketing Teams** - humanizing product descriptions and blog posts
    """)

    st.success("🚀 **Fast & Secure**: Humanization happens locally. Grammar correction supports LanguageTool (free, no key), Groq Llama 3.3 (free key), or Anthropic Claude (paid).")

    st.markdown("---")

    st.subheader("🎛️ Customize Your Humanization Settings")

    col1, col2 = st.columns(2)
    
    with col1:
        p_syn = st.slider(
            "**Synonym Replacement Intensity**", 
            0.0, 1.0, 0.3, 0.05,
            help="Higher values replace more words with natural synonyms for greater variation"
        )
    
    with col2:
        p_trans = st.slider(
            "**Human Touch Intensity**", 
            0.0, 1.0, 0.3, 0.05,
            help="Higher values add more casual transitions and sentence variation"
        )

    st.subheader("📝 Enter Your Text to Humanize")
    
    input_text = st.text_area(
        "Paste your AI-generated text below:", 
        height=200,
        placeholder="Paste your text here... We'll automatically protect your citations, code blocks, and enhance the writing style.",
        label_visibility="collapsed"
    )

    if st.button("🚀 Humanize Text", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("📝 Please enter some text to humanize first.")
            return

        # Show original stats
        orig_wc = count_words(input_text)
        orig_sc = count_sentences(input_text)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Word Count", orig_wc)
        with col2:
            st.metric("Original Sentence Count", orig_sc)

        with st.spinner("🔍 Analyzing text and protecting code & logic..."):
            # 1. Extract markdown code blocks first (they become [[CODE_BLOCK_n]])
            code_protected_text, code_map = extract_code_blocks(input_text)
            
            # 2. Use the elegant split logic: protect HTML <...>, Shortcodes [...], and code block placeholders
            # re.split with a capture group keeps the delimiters in the resulting list
            parts = re.split(r'(\[.*?\]|<.*?>)', code_protected_text, flags=re.DOTALL)
            
            final_parts = []
            
        with st.spinner("✍️ Enhancing plain text while preserving structure..."):
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    # Odd indices are the protected tags/shortcodes matched by the regex - LEAVE UNTOUCHED
                    final_parts.append(part)
                else:
                    # Even indices are plain text
                    if not part.strip():
                        final_parts.append(part)
                    else:
                        # Extract academic citations just from this plain text chunk
                        no_refs_text, placeholders = extract_citations(part)
                        
                        # Apply humanization line-by-line
                        h_lines = no_refs_text.splitlines()
                        out_lines = []
                        for ln in h_lines:
                            if not ln.strip():
                                out_lines.append("")
                            elif is_code_block(ln.strip()):
                                # Protect plain-text lines that look like code
                                out_lines.append(ln)
                            else:
                                out_lines.append(minimal_rewriting(ln, p_syn=p_syn, p_trans=p_trans))
                                
                        rewritten = "\n".join(out_lines)
                        
                        # Restore citations
                        rewritten = restore_citations(rewritten, placeholders)
                        
                        # Apply punctuation cleanup
                        rewritten = re.sub(r"[ \t]+([.,;:!?])", r"\1", rewritten)
                        rewritten = re.sub(r"(\()[ \t]+", r"\1", rewritten)
                        rewritten = re.sub(r"[ \t]+(\))", r"\1", rewritten)
                        rewritten = re.sub(r"[ \t]{2,}", " ", rewritten)
                        rewritten = re.sub(r"``\s*(.+?)\s*''", r'"\1"', rewritten)
                        
                        final_parts.append(rewritten)
                        
            final_text = "".join(final_parts)
            
        with st.spinner("✅ Restoring code blocks and finalizing..."):
            # Restore the markdown code blocks
            final_text = restore_code_blocks(final_text, code_map)

        # Store the humanized text in session state for grammar correction
        st.session_state["humanized_output"] = final_text

        # Calculate new stats
        new_wc = count_words(final_text)
        new_sc = count_sentences(final_text)

        st.subheader("🎉 Your Humanized Text")

        st.success(f"✅ Successfully enhanced your text! Added **{new_wc - orig_wc} words** and **{new_sc - orig_sc} sentences** for better flow.")

        # Single editable output box that preserves original line breaks and paragraphs
        st.text_area(
            "Humanized Result",
            final_text,
            height=300,
            label_visibility="collapsed"
        )

        # Copy to clipboard functionality
        st.download_button(
            "📋 Download Humanized Text",
            data=final_text,
            file_name="humanized_text.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.markdown("""
        ### 📊 Enhancement Summary
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Words Added", new_wc - orig_wc, delta="Enhancement")
        with col2:
            st.metric("Sentences Added", new_sc - orig_sc, delta="Flow")
        with col3:
            st.metric("Final Word Count", new_wc)
        with col4:
            st.metric("Final Sentence Count", new_sc)

    else:
        st.info("""
        👆 **Ready to enhance your text?** 
        - Paste your AI-generated content above
        - Adjust the sliders to control enhancement intensity  
        - Click the 'Humanize Text' button to transform your writing
        - Your citations and code blocks will be automatically protected!
        """)

    # ========================================
    # GRAMMAR CORRECTION SECTION
    # ========================================
    st.markdown("---")
    st.subheader("📝 Grammar Correction (Preserves Humanized Style)")

    st.markdown("""
    Fix grammar, spelling, and punctuation **without undoing the humanization**.
    Choose your preferred engine below — two are completely free.
    """)

    # Engine selector
    engine = st.radio(
        "Select grammar engine:",
        options=[
            "LanguageTool (FREE — no key needed)",
            "Groq / Llama 3.3 (FREE — free API key required)",
            "Anthropic Claude (PAID — best quality)",
        ],
        index=0,
        horizontal=True,
    )

    # Conditional API key input
    api_key = ""
    if engine == "Groq / Llama 3.3 (FREE — free API key required)":
        st.info("Get a free Groq API key at **console.groq.com** — takes 30 seconds, no credit card needed.")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Your key is not stored anywhere.",
        )
    elif engine == "Anthropic Claude (PAID — best quality)":
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Your key is not stored anywhere.",
        )
    else:
        st.success("LanguageTool works with no API key — just paste your text and click Fix Grammar.")

    # Grammar correction input — auto-fill from humanized output if available
    default_grammar_text = st.session_state.get("humanized_output", "")

    grammar_input = st.text_area(
        "Paste humanized text for grammar correction:",
        value=default_grammar_text,
        height=250,
        placeholder="Paste your humanized text here...",
        key="grammar_input_area",
    )

    if st.button("Fix Grammar", type="primary", use_container_width=True):
        if not grammar_input.strip():
            st.warning("Please paste some text to correct grammar.")
        elif engine != "LanguageTool (FREE — no key needed)" and not api_key.strip():
            st.warning("Please enter your API key for the selected engine.")
        else:
            with st.spinner("Fixing grammar while preserving humanized style..."):
                try:
                    if engine == "LanguageTool (FREE — no key needed)":
                        corrected_text = fix_grammar_languagetool(grammar_input)
                    elif engine == "Groq / Llama 3.3 (FREE — free API key required)":
                        corrected_text = fix_grammar_groq(grammar_input, api_key.strip())
                    else:
                        corrected_text = fix_grammar_with_api(grammar_input, api_key.strip())
                except Exception as e:
                    corrected_text = ""
                    st.error(f"Error: {e}")

            if corrected_text and corrected_text.strip():
                st.session_state["grammar_corrected_output"] = corrected_text

                st.subheader("Grammar-Corrected Text")
                st.success("Grammar fixed! Your humanized writing style has been preserved.")

                st.text_area(
                    "Grammar Corrected Result",
                    corrected_text,
                    height=300,
                    label_visibility="collapsed",
                    key="grammar_output_area",
                )

                st.download_button(
                    "Download Grammar-Corrected Text",
                    data=corrected_text,
                    file_name="grammar_corrected_text.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

                if grammar_input.strip() != corrected_text.strip():
                    with st.expander("View Changes (Before vs After)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Before (Humanized)**")
                            st.text_area("Before", grammar_input, height=200,
                                         label_visibility="collapsed", key="before_compare",
                                         disabled=True)
                        with col2:
                            st.markdown("**After (Grammar Fixed)**")
                            st.text_area("After", corrected_text, height=200,
                                         label_visibility="collapsed", key="after_compare",
                                         disabled=True)
                else:
                    st.info("No grammar issues found! Your text looks great.")
            elif not corrected_text:
                pass  # error already shown above
            else:
                st.error("Grammar correction returned empty output. Please try again.")

# Run the app
if __name__ == "__main__":
    show_humanize_page()