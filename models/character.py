from langchain_openai import ChatOpenAI
import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

class Character:
    def __init__(self, character_name, character_story_path, model_name='gpt-4o-mini', temperature= 0.1):
        self.character_name = character_name
        self.character_story_path = character_story_path
        self.model_name = model_name
        self.temperature = temperature

        self.model = self.initialize_model()
        self.prompt = self.create_prompt()
        self.chain = self.create_chain()
        self.chat_history = ChatMessageHistory()
    
    def initialize_model(self):
        model = ChatOpenAI(model=self.model_name)
        model.temperature = self.temperature
        return model
    
    def create_prompt(self):
        with open(self.character_story_path, 'r') as file:
            character_story = file.read()
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    character_story,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        return prompt

    def create_chain(self):
        chain = self.prompt | self.model
        return chain
    
class CharacterGen:
    def __init__(self, backstory, character_name, character_story_path, model_name='gpt-4o-mini', temperature= 0.1):
        self.backstory = backstory
        self.character_name = character_name
        self.character_story_path = character_story_path
        self.model_name = model_name
        self.temperature = temperature

        self.model = self.initialize_model()
        self.prompt = self.create_prompt()
        self.chain = self.create_chain()
        self.chat_history = ChatMessageHistory()
    
    def initialize_model(self):
        model = ChatOpenAI(model=self.model_name)
        model.temperature = self.temperature
        return model
    
    def create_prompt(self):
        with open(self.character_story_path, 'r') as file:
            character_story = file.read()
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    character_story + '\n\n' + self.backstory,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        return prompt

    def create_chain(self):
        chain = self.prompt | self.model
        return chain
