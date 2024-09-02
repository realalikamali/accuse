# %%
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain_core.output_parsers import StrOutputParser

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class KnowItAll:
    def __init__(self, prompt_path, model_name='gpt-4o', temperature= 0.2):

        self.model_name = model_name
        self.temperature = temperature
        self.prompt_path = prompt_path
        self.model = self.initialize_model()
        self.prompt = self.create_prompt()
        self.chain = self.create_chain()

    def initialize_model(self):
        model = ChatOpenAI(model=self.model_name)
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
    def __init__(self, backstory, model_name='gpt-4o', temperature= 0.0):

        self.model_name = model_name
        self.temperature = temperature
        self.backstory = backstory
        self.model = self.initialize_model()
        self.prompt = self.create_prompt()
        self.chain = self.create_chain()

    def initialize_model(self):
        model = ChatOpenAI(model=self.model_name)
        model.temperature = self.temperature
        return model

    def create_prompt(self):
        
        instructions_pre = "The following is the events leading up to someone\'s murder on a camping trip. \n\n"
        instructions_post = ("\n\nYour task is to extract the events that each of the remaining three characters."
                            " Be detailed in recounting the events from each character\'s point of view."
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


class QuestionCap:
    def __init__(self, model_name='gpt-4o-mini', temperature= 0.0):

        self.model_name = model_name
        self.temperature = temperature
        self.tagging_functions = [convert_to_openai_function(Tagging)]
        self.model = self.initialize_model()
        self.prompt = self.create_prompt()
        self.tagging_chain = self.create_tagging_chain()

    
    def initialize_model(self):
        model = ChatOpenAI(model=self.model_name)
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


