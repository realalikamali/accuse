# %%
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser, JsonOutputFunctionsParser
from langchain_core.output_parsers import StrOutputParser

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List

class KnowItAll:
    def __init__(self, prompt_path, openai_api_key, model_name='gpt-4o', temperature= 0.2):

        self.model_name = model_name
        self.temperature = temperature
        self.prompt_path = prompt_path
        self.openai_api_key = openai_api_key
        self.model = self.initialize_model()
        self.prompt = self.create_prompt()
        self.chain = self.create_chain()

    def initialize_model(self):
        model = ChatOpenAI(model=self.model_name, openai_api_key=self.openai_api_key)
        model.temperature = self.temperature
        return model
    
    def create_prompt(self):
        with open(self.prompt_path, 'r') as file:
            story_seed = file.read()

        prompt = ChatPromptTemplate.from_template(
            story_seed
            )
        return prompt
    def create_chain(self):
        chain = self.prompt | self.model | StrOutputParser()
        return chain

class POVExtractor:
    def __init__(self, backstory, openai_api_key, model_name='gpt-4o', temperature= 0.0):

        self.model_name = model_name
        self.temperature = temperature
        self.backstory = backstory
        self.openai_api_key = openai_api_key
        self.model = self.initialize_model()
        self.prompt = self.create_prompt()
        self.chain = self.create_chain()

    def initialize_model(self):
        model = ChatOpenAI(model=self.model_name, openai_api_key=self.openai_api_key)
        model.temperature = self.temperature
        return model

    def create_prompt(self):
        
        instructions_pre = "The following is the events leading up to someone\'s murder on a camping trip. \n\n"
        instructions_post = ("\n\nRecount the events from each character\'s point of view in a detailed manner."
                            " Include their thoughts, feelings, and actions. Use first person perspective."
                            " Ensure that the recounting done by the three characters do not contradict each other")


        # Set up a parser + inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=EventsRecounting)

        prompt = PromptTemplate(
            template= instructions_pre + self.backstory + instructions_post + '\n{format_instructions}',
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt
    
    def create_chain(self):
        # Set up a parser + inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=EventsRecounting)
        chain = self.prompt | self.model | parser
        return chain

class EventsRecounting(BaseModel):
    Jennifer: str = Field(description="Description of the events from Jennifer's point of view in first person")
    Cindy: str = Field(description="Description of the events from Cindy's point of view in first person")
    James: str = Field(description="Description of the events from James' point of view in first person")

class EvidenceExtractor:
    def __init__(self, backstory, openai_api_key, model_name='gpt-4o', temperature= 0.2):

        self.model_name = model_name
        self.temperature = temperature
        self.backstory = backstory
        self.openai_api_key = openai_api_key
        self.model = self.initialize_model()
        self.prompt = self.create_prompt()
        self.chain = self.create_chain()

    def initialize_model(self):
        model = ChatOpenAI(model=self.model_name, openai_api_key=self.openai_api_key)
        model.temperature = self.temperature
        return model

    def create_prompt(self):

        pre_prompt = (
            "You are trying to devise the backstory to a compelling murder mystery game. "
            "Using the following story, come up with three physical pieces of evidence "
            "and create an updated story that incorporates those components directly into the story "
            "rather than stating them after the end of the events. "
            "These pieces of evidence should be tangible and not anectodes of the suspects. "
            "None of these should directly implicate a character and ruin the game. "
            "They should rather give some context to the player so they can bring them up when interviewing suspects. "
            "The killer should remain the same.\n\n"
            )

        prompt = ChatPromptTemplate.from_template(
        pre_prompt + self.backstory
        )

        return prompt
    
    def create_chain(self):
        # Set up a parser + inject instructions into the prompt template.
        extraction_functions = [convert_to_openai_function(StoryWithEvidence)]
        extraction_model = self.model.bind(functions=extraction_functions, function_call={"name": "StoryWithEvidence"})
        chain = self.prompt | extraction_model | JsonOutputFunctionsParser()
        return chain

class Evidence(BaseModel):
    """Information about a person."""
    name: str = Field(description="Name of the piece of evidence")
    spoiler_description: str = Field(description="Detailed description of the evidence and who it relates to accesible to the game developers and not the player")
    spoiler_free_description: str = Field(description="Spoiler free but detailed description of the evidence that the player can see and use as context in their quest to solve the mystery")

class StoryWithEvidence(BaseModel):
    """Information to extract."""
    people: List[Evidence] = Field(description="List of all pieces of evidence")
    updated_story: str = Field(description="Updated story with evidence added")

class QuestionCap:
    def __init__(self, openai_api_key, model_name='gpt-4o-mini', temperature= 0.0):

        self.model_name = model_name
        self.temperature = temperature
        self.tagging_functions = [convert_to_openai_function(Tagging)]
        self.openai_api_key = openai_api_key
        self.model = self.initialize_model()
        self.prompt = self.create_prompt()
        self.tagging_chain = self.create_tagging_chain()

    
    def initialize_model(self):
        model = ChatOpenAI(model=self.model_name, openai_api_key=self.openai_api_key)
        model.temperature = self.temperature
        return model

    def create_prompt(self):
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'Think carefully and tag the text as instructed'),
            ('user', '{input}')
        ])
        return prompt

    def create_model_with_functions(self):
        model_with_functions = self.model.bind(
            functions=self.tagging_functions,
            function_call = {"name": "Tagging"}
        )
        return model_with_functions

    def create_tagging_chain(self):
        tagging_chain = self.prompt | self.create_model_with_functions() | JsonKeyOutputFunctionsParser(key_name="more_than_one_question")
        return tagging_chain

    def invoke(self, input: str):
        
        return self.tagging_chain.invoke({"input": input})

class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    more_than_one_question: bool = Field(description="Set to true if there are more than one question in the text. otherwise false.")


