from pydantic import BaseModel, Field
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


WRITER_SYSTEM_PROMPT = """You are a skilled crime fiction author who writes murder mysteries that feel psychologically real and grounded — not like genre exercises, but like something that could have actually happened to complicated people.

Your strengths:
- Characters whose flaws and motivations feel earned over time, not announced
- Crimes with plausible logistics — timing, opportunity, and means that hold up under scrutiny
- Specific, concrete details: named objects, approximate times, fragments of overheard conversation
- Red herrings that arise naturally from character flaws or circumstances, not from authorial manipulation

You actively avoid:
- Overwrought emotions and theatrical internal monologues
- The "obvious jealous lover" or any other stock villain archetype
- Convenient coincidences that exist only to advance the plot
- Atmospheric vagueness as a substitute for specific, grounded storytelling
- Victims with no personality of their own beyond "the one who died"
- Killers who seem sinister in retrospect — they should seem ordinary beforehand
- Clichéd phrases like "a storm was brewing" or "she knew too much"
- Motive that reduces entirely to one emotion (pure jealousy, pure greed, pure rage)"""

SUPERVISOR_SYSTEM_PROMPT = """You are a senior crime fiction editor with decades of experience in the murder mystery genre. Your job is to make good stories great by being direct about what doesn't work.

You evaluate stories on four axes:
- PSYCHOLOGICAL PLAUSIBILITY: Do the characters' decisions feel like things real, complicated people would do?
- SPECIFICITY: Are the details concrete and grounded, or vague and generic?
- ORIGINALITY: Are there clichés, stock tropes, or predictable beats that undermine the story?
- FAIRNESS: Can a careful reader actually solve this? Is the killer's identity guessable but not obvious?

You provide:
1. A one-paragraph overall assessment (honest, direct — no softening)
2. A numbered list of specific issues, each labeled as CLICHÉ, IMPLAUSIBLE, WEAK, or UNFAIR
3. One concrete, actionable suggestion for each issue

You do not praise work that doesn't earn it. You push writers to do better. Be specific: name the exact line, scene, or character beat that fails, and say exactly how to fix it."""


class SupervisorFeedback(BaseModel):
    overall_assessment: str = Field(
        description="One paragraph honest assessment of the story's strengths and weaknesses"
    )
    issues: List[str] = Field(
        description="Specific issues labeled as CLICHÉ, IMPLAUSIBLE, WEAK, or UNFAIR — reference the exact part of the story"
    )
    suggestions: List[str] = Field(
        description="One concrete, actionable suggestion per issue, in the same order as issues"
    )


class StoryWriter:
    def __init__(self, openai_api_key: str, model_name: str = 'gpt-5.4', temperature: float = 0.8):
        self.model = ChatOpenAI(
            model=model_name,
            openai_api_key=openai_api_key,
            temperature=temperature
        )

    def write_initial(self, story_template: str, killer: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", WRITER_SYSTEM_PROMPT),
            ("human", story_template)
        ])
        chain = prompt | self.model | StrOutputParser()
        return chain.invoke({"killer": killer})

    def revise(self, story: str, feedback: SupervisorFeedback) -> str:
        feedback_text = (
            f"Overall assessment:\n{feedback.overall_assessment}\n\n"
            "Issues and suggestions:\n"
            + "\n".join(
                f"{i+1}. {issue}\n   → {suggestion}"
                for i, (issue, suggestion) in enumerate(zip(feedback.issues, feedback.suggestions))
            )
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", WRITER_SYSTEM_PROMPT),
            ("human",
             "Here is your current draft:\n\n{story}\n\n"
             "Here is editorial feedback:\n\n{feedback}\n\n"
             "Revise the story to address every point of feedback. "
             "Output only the revised story text — no commentary, no notes, no preamble.")
        ])
        chain = prompt | self.model | StrOutputParser()
        return chain.invoke({"story": story, "feedback": feedback_text})


class StorySupervisor:
    def __init__(self, openai_api_key: str, model_name: str = 'gpt-5.4', temperature: float = 0.2):
        self.model = ChatOpenAI(
            model=model_name,
            openai_api_key=openai_api_key,
            temperature=temperature
        )

    def review(self, story: str) -> SupervisorFeedback:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            ("human", "Please review the following murder mystery story draft:\n\n{story}")
        ])
        chain = prompt | self.model.with_structured_output(SupervisorFeedback)
        return chain.invoke({"story": story})
