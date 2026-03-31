#!/usr/bin/env python3
"""
Generate and encrypt murder mystery stories using a writer/supervisor pattern.

Stories are saved as encrypted .enc files in stories/ and loaded at runtime by
the Streamlit app. This eliminates LLM calls on app startup — only character
interactions use the API.

Usage:
    # First-time setup: generate an encryption key
    python scripts/generate_stories.py --generate-key

    # Add the printed key to .streamlit/secrets.toml as:
    #   encryption_key = "..."

    # Generate all 10 stories
    python scripts/generate_stories.py

    # Regenerate a single story (e.g. story 3)
    python scripts/generate_stories.py --start-from 3 --count 1

Prerequisites:
    - openai_api_key in .streamlit/secrets.toml (or OPENAI_API_KEY env var)
    - encryption_key in .streamlit/secrets.toml (generate with --generate-key)
    - pip install -r requirements.txt
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptography.fernet import Fernet
from models.story_generator import StoryWriter, StorySupervisor
from models.controls import EvidenceExtractor, POVExtractor

STORIES_DIR = Path(__file__).parent.parent / "stories"
SECRETS_PATH = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "initiation_prompt_omniscient.txt"


def load_secrets() -> dict:
    if not SECRETS_PATH.exists():
        return {}
    try:
        import tomllib
        with open(SECRETS_PATH, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        # Fallback for Python < 3.11
        import re
        secrets = {}
        with open(SECRETS_PATH, "r") as f:
            for line in f:
                m = re.match(r'^\s*(\w+)\s*=\s*"(.+)"\s*$', line)
                if m:
                    secrets[m.group(1)] = m.group(2)
        return secrets


def run_writer_supervisor_cycle(
    writer: StoryWriter,
    supervisor: StorySupervisor,
    story_template: str,
    killer: str,
) -> str:
    """Two-round writer/supervisor feedback cycle. Returns polished story text."""
    print("    [Writer] Initial draft...")
    story = writer.write_initial(story_template, killer)

    print("    [Supervisor] Round 1 feedback...")
    feedback1 = supervisor.review(story)
    print(f"      {len(feedback1.issues)} issue(s): " + "; ".join(
        f.split("]")[0].lstrip("[") if "]" in f else f[:40]
        for f in feedback1.issues[:3]
    ))

    print("    [Writer] Revision 1...")
    story = writer.revise(story, feedback1)

    print("    [Supervisor] Round 2 feedback...")
    feedback2 = supervisor.review(story)
    print(f"      {len(feedback2.issues)} issue(s): " + "; ".join(
        f.split("]")[0].lstrip("[") if "]" in f else f[:40]
        for f in feedback2.issues[:3]
    ))

    print("    [Writer] Final revision...")
    story = writer.revise(story, feedback2)

    return story


def generate_one_story(
    story_number: int,
    writer: StoryWriter,
    supervisor: StorySupervisor,
    story_template: str,
    api_key: str,
) -> dict:
    killer = random.choice(["Jennifer", "Cindy", "James"])
    print(f"  Killer: {killer}")

    story = run_writer_supervisor_cycle(writer, supervisor, story_template, killer)

    print("    [EvidenceExtractor] Generating physical clues...")
    evidence_extractor = EvidenceExtractor(story, api_key, model_name="gpt-4o", temperature=0.4)
    extracted = evidence_extractor.chain.invoke({})
    story = extracted["updated_story"]
    pieces_of_evidence = extracted["people"]

    print("    [POVExtractor] Generating character perspectives...")
    pov_extractor = POVExtractor(story, api_key, model_name="gpt-4o", temperature=0.0)
    individual_povs = pov_extractor.chain.invoke({})

    return {
        "story_number": story_number,
        "killer": killer,
        "backstory": story,
        "pieces_of_evidence": pieces_of_evidence,
        "individual_povs": individual_povs,
    }


def encrypt_story(story_data: dict, fernet: Fernet) -> bytes:
    return fernet.encrypt(json.dumps(story_data, ensure_ascii=False).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate encrypted murder mystery stories for Accuse."
    )
    parser.add_argument(
        "--generate-key",
        action="store_true",
        help="Generate a new Fernet encryption key and print it.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of stories to generate (default: 10).",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        metavar="N",
        help="Start numbering from N (default: 1). Useful to regenerate a specific story.",
    )
    args = parser.parse_args()

    if args.generate_key:
        key = Fernet.generate_key()
        print("\nGenerated encryption key — add this to .streamlit/secrets.toml:\n")
        print(f'  encryption_key = "{key.decode()}"')
        print()
        return

    secrets = load_secrets()
    api_key = secrets.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: no OpenAI API key found.")
        print("Set openai_api_key in .streamlit/secrets.toml or the OPENAI_API_KEY env var.")
        sys.exit(1)

    encryption_key_str = secrets.get("encryption_key")
    if not encryption_key_str:
        print("Error: no encryption_key found in .streamlit/secrets.toml.")
        print("Run with --generate-key to create one, then add it to secrets.toml.")
        sys.exit(1)

    fernet = Fernet(encryption_key_str.encode())
    STORIES_DIR.mkdir(exist_ok=True)

    with open(PROMPT_PATH, "r") as f:
        story_template = f.read()

    writer = StoryWriter(api_key)
    supervisor = StorySupervisor(api_key)

    total = args.count
    for i in range(args.start_from, args.start_from + total):
        print(f"\n--- Story {i} of {args.start_from + total - 1} ---")
        story_data = generate_one_story(i, writer, supervisor, story_template, api_key)

        output_path = STORIES_DIR / f"story_{i:02d}.enc"
        with open(output_path, "wb") as f:
            f.write(encrypt_story(story_data, fernet))
        print(f"  Saved → {output_path.relative_to(Path(__file__).parent.parent)}")

    print(f"\nDone. {total} story file(s) in {STORIES_DIR.relative_to(Path(__file__).parent.parent)}/")


if __name__ == "__main__":
    main()
