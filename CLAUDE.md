# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Accuse** is an LLM-powered, chat-based murder mystery game built with Streamlit. Players interrogate AI suspects (powered by GPT-4o) to identify a killer. Stories are pre-generated and encrypted — the app picks one randomly at startup, so no LLM calls happen until the player interacts with a character.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run st_app_with_evidence.py
# Access at http://localhost:8501
```

**API key setup** — all three entries go in `.streamlit/secrets.toml` (git-ignored):
```toml
openai_api_key = "sk-..."
encryption_key = "..."                    # see story generation below
stories_google_drive_link = "https://drive.google.com/drive/folders/..."
```

There is no automated test suite; validation is done through manual UI testing.

## Story Generation (offline, run once)

Stories are pre-generated via a writer/supervisor LLM pipeline, encrypted, and stored in a shared Google Drive folder — **not committed to the repo**. The app downloads and decrypts them at cold start.

```bash
# 1. Generate an encryption key (one-time)
python scripts/generate_stories.py --generate-key
# Add the printed key to .streamlit/secrets.toml as encryption_key = "..."

# 2. Generate all 10 stories (uses gpt-5.4, takes several minutes)
python scripts/generate_stories.py
# Output: stories/story_01.enc ... stories/story_10.enc

# 3. Regenerate a single story (e.g. story 5)
python scripts/generate_stories.py --start-from 5 --count 1
```

After generating, **manually upload** the `.enc` files to the shared Google Drive folder and ensure the folder link is set as `stories_google_drive_link` in `secrets.toml`. The `stories/*.enc` files are git-ignored.

## Architecture

### Entry Point

`st_app_with_evidence.py` — the Streamlit app. On cold start `fetch_all_stories()` (`@st.cache_resource`) downloads all `.enc` files from the Google Drive folder, decrypts them, and holds them in memory. Each new session then picks one randomly via `load_story()` and initialises three `CharacterGen` agents. All subsequent LLM usage is character interview traffic only.

### Models (`models/`)

**`controls.py`** — LLM wrappers used at runtime:
| Class | Model | Temp | Purpose |
|---|---|---|---|
| `QuestionCap` | gpt-4o-mini | 0.0 | Detect multi-part questions (Pydantic `Tagging`) |
| `Character` | gpt-4o | 0.1 | Run character interviews with chat history |

**`controls.py`** also contains `EvidenceExtractor` and `POVExtractor` — used only by the generation script, not the app.

**`story_generator.py`** — offline-only writer/supervisor classes:
| Class | Model | Temp | Purpose |
|---|---|---|---|
| `StoryWriter` | gpt-5.4 | 0.8 | Write and revise mystery narrative |
| `StorySupervisor` | gpt-5.4 | 0.2 | Editorial critic; returns `SupervisorFeedback` |

**`character.py`** — two character classes:
- `Character` — basic character with in-memory chat history
- `CharacterGen` — injects pre-computed POV backstory into the system prompt

### Prompts (`prompts/`)

- `initiation_prompt_omniscient.txt` — story generation template (accepts `{killer}` variable)
- `welcome_message.txt` — game intro text (accepts `{number_of_interactions}`)
- `Jennifer_backstory.txt`, `Cindy_backstory.txt`, `James_backstory.txt` — system prompts for each suspect

### Story Data Schema

Each `.enc` file decrypts to a JSON object:
```json
{
  "story_number": 1,
  "killer": "Jennifer",
  "backstory": "...",
  "pieces_of_evidence": [{"name": "...", "spoiler_description": "...", "spoiler_free_description": "..."}],
  "individual_povs": {"Jennifer": "...", "Cindy": "...", "James": "..."}
}
```

### Writer/Supervisor Cycle (in `scripts/generate_stories.py`)

```
random killer → StoryWriter.write_initial()
             → StorySupervisor.review()        # round 1 feedback
             → StoryWriter.revise()
             → StorySupervisor.review()        # round 2 feedback
             → StoryWriter.revise()            # final story
             → EvidenceExtractor              # 3 physical clues + updated story
             → POVExtractor × 3              # per-character first-person accounts
             → Fernet.encrypt() → story_NN.enc
```

### Game Mechanics

- **5-question cap** per character per round (2 rounds = 10 messages per character)
- `QuestionCap` blocks multi-part questions before they reach the character agent
- Strict turn order enforced: must finish one character before switching
- Story number shown on the Intro page as `Case #NN`
- All game state lives in `st.session_state`; story loading is cached there to prevent re-randomising on reruns
