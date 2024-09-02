import streamlit as st
import os
import openai
from dotenv import load_dotenv, find_dotenv
from models.character import CharacterGen
from models.controls import QuestionCap, KnowItAll, POVExtractor
from PIL import Image
import random

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

st.sidebar.title('Accuse')
# an interactive chat-based adventure
st.sidebar.subheader('An interactive chat-based adventure')

permissible_length_of_chat = 2*5 # 2 rounds of 5 questions each
stream_response = True
character_avatar_emojis = {'Jennifer': '👩',
                           'Cindy': '👩‍🦱',
                           'James': '👨'}
@st.cache_resource
def startup_proceses():
    questioncap = QuestionCap()
    knowitall = KnowItAll(os.path.join('prompts','initiation_prompt_omniscient.txt'))
    # randomly choose a killer using np.random.choice
    killer = random.choice(['Jennifer', 'Cindy', 'James'])
    backstory = knowitall.chain.invoke({'killer': killer})

    povextractor = POVExtractor(backstory, model_name='gpt-4o', temperature=0.0)
    individual_povs = povextractor.chain.invoke({})
    # Create a dictionary of agents
    agents = {
        character_name: CharacterGen(
            character_name,
            individual_povs[character_name],
            os.path.join('prompts', (character_name + '_backstory.txt')),
            model_name='gpt-4o',
            temperature=0.1
        )
        for character_name in ['Jennifer', 'Cindy', 'James']
    }

    return questioncap, killer, backstory, individual_povs, agents

questioncap, killer, backstory, individual_povs, agents = startup_proceses()

# set up session state variables
if 'number_of_messages' not in st.session_state:
    st.session_state.number_of_messages = {}
    for character in ['Jennifer', 'Cindy', 'James']:
        st.session_state.number_of_messages[character] = 0
if 'messages' not in st.session_state:
    st.session_state.messages = {}
    for character in ['Jennifer', 'Cindy', 'James']:
        st.session_state.messages[character] = []
if 'agents' not in st.session_state:
    st.session_state.agents = agents
if 'more_than_one_question' not in st.session_state:
    st.session_state.more_than_one_question = False

if 'killer' not in st.session_state:
    st.session_state.killer = killer

if 'backstory' not in st.session_state:
    st.session_state.backstory = backstory

if 'individual_povs' not in st.session_state:
    st.session_state.individual_povs = individual_povs

radio = st.sidebar.radio(
    "Select mode:",
    ["Intro", "First-round Interview","Second-round Interview", "Solve","Spoiler"],
    key="radio")
container_top = st.empty()
container_bottom = st.empty()

if radio == "Intro":
    # read the welcome message and store it in a variable
    with open(os.path.join('prompts', "welcome_message.txt"), "r") as file:
        welcome_message = file.read()
    welcome_message = welcome_message.format(permissible_length_of_chat = 10)

    container_top.write(welcome_message)

    # Read the cover photo
    cover_photo = Image.open("m_mystery_cover_photo.webp")

    # Show the cover photo on the page
    container_bottom.image(cover_photo)

elif radio == 'First-round Interview':
    container_top.empty()
    container_bottom.empty()
    
    # create a dropdown menu to select the character in the sidebar
    st.sidebar.title('Select Character to interview')
    character = st.sidebar.selectbox('Character', ['Jennifer', 'Cindy', 'James'])

    # Use the selected character to fetch the appropriate agent
    selected_agent = st.session_state.agents[character]

    # st.header(character)

    # Display the history of the chat messages
    for i, message in enumerate(st.session_state.messages[character]):
        if message['role'] == 'user':
            st.chat_message(message['role'], avatar='🕵️‍♂️').markdown(f'{int(i/2+1)}. '+message['content'])
        else:
            st.chat_message(message['role'], avatar=character_avatar_emojis[character]).markdown(message['content'])

    # Input prompt from the user
    prompt = st.chat_input("Enter your message here:", key="input_" + character)
    st.session_state.more_than_one_question = questioncap.invoke(prompt)

    if prompt:
        # Check if any character has between 0 and permissible_length_of_chat messages
        chat_needs_to_end = False
        for char, num_messages in st.session_state.number_of_messages.items():
            if 0 < num_messages < permissible_length_of_chat and char != character:
                st.warning(f"First-round interview with {char} needs to end before starting a new chat.")
                chat_needs_to_end = True
                break
        
        # Check if the current chat has reached or exceeded the permissible length
        if st.session_state.number_of_messages[character] >= permissible_length_of_chat:
            st.warning(f"You reached the end of your conversation with {character}."
                       "Interview another character or if you are done with all the interviews, proceed to the second round.")
            chat_needs_to_end = True

        # Only proceed if no conditions to end the chat were met
        if not chat_needs_to_end:
            if st.session_state.more_than_one_question:
                st.warning(f"Ask only one question at a time.")
            else:
                st.chat_message('user', avatar = '🕵️‍♂️').markdown(f'{int(st.session_state.number_of_messages[character]/2+1)}. '+prompt)

                selected_agent.chat_history.add_user_message(prompt)

                st.session_state.messages[character].append({'role': 'user', 'content': prompt})

                if stream_response:
                    with st.chat_message('system', avatar = character_avatar_emojis[character]):
                        stream = selected_agent.chain.stream({"messages": selected_agent.chat_history.messages})
                        response = st.write_stream(stream)
                else:
                    response = selected_agent.chain.invoke({"messages": selected_agent.chat_history.messages})
                    st.chat_message('system', avatar = character_avatar_emojis[character]).markdown(response.content)

                st.session_state.messages[character].append({'role': 'system', 'content': response})

                selected_agent.chat_history.add_ai_message(response)
                # Update the number of messages for the character
                st.session_state.number_of_messages[character] = len(selected_agent.chat_history.messages)

                # Warn if the chat has reached the permissible length
                if st.session_state.number_of_messages[character] == permissible_length_of_chat:
                    st.warning(f"You reached the end of your conversation with {character}."
                               " Interview another character or if you are done with all the interviews, proceed to the second round.")
                    chat_needs_to_end = True

elif radio == 'Second-round Interview':
    container_top.empty()
    container_bottom.empty()
    
    # create a dropdown menu to select the character in the sidebar
    st.sidebar.title('Select Character to interview')
    character = st.sidebar.selectbox('Character', ['Jennifer', 'Cindy', 'James'])


    # Use the selected character to fetch the appropriate agent
    selected_agent = st.session_state.agents[character]

    # st.header(character)

    # Display the history of the chat messages
    for i, message in enumerate(st.session_state.messages[character]):
        if message['role'] == 'user':
            st.chat_message(message['role'], avatar='🕵️‍♂️').markdown(f'{int(i/2+1)}. '+message['content'])
        else:
            st.chat_message(message['role'], avatar=character_avatar_emojis[character]).markdown(message['content'])

    # Input prompt from the user
    prompt = st.chat_input("Enter your message here:", key="input_" + character)
    st.session_state.more_than_one_question = questioncap.invoke(prompt)

    if prompt:
        # Check if any character has between 0 and permissible_length_of_chat messages
        chat_needs_to_end = False

        for char, num_messages in st.session_state.number_of_messages.items():
            if 0 <= num_messages < permissible_length_of_chat:
                st.warning(f"First round interviews need to end before starting second round interviews.")
                chat_needs_to_end = True
                break

        for char, num_messages in st.session_state.number_of_messages.items():
            if permissible_length_of_chat < num_messages < 2*permissible_length_of_chat and char != character:
                st.warning(f"Chat with {char} needs to end before continuing this chat.")
                chat_needs_to_end = True
                break
        
        # Check if the current chat has reached or exceeded the permissible length
        if st.session_state.number_of_messages[character] >= 2*permissible_length_of_chat:
            st.warning(f"You reached the end of your conversation with {character}."
                       " Interview another character or if you are done with all the interviews, proceed to solve the case.")
            chat_needs_to_end = True

        # Only proceed if no conditions to end the chat were met
        if not chat_needs_to_end:
            if st.session_state.more_than_one_question:
                st.warning(f"Ask only one question at a time.")
            else:
                st.chat_message('user', avatar = '🕵️‍♂️').markdown(f'{int(st.session_state.number_of_messages[character]/2+1)}. '+prompt)

                selected_agent.chat_history.add_user_message(prompt)

                st.session_state.messages[character].append({'role': 'user', 'content': prompt})

                if stream_response:
                    with st.chat_message('system', avatar = character_avatar_emojis[character]):
                        stream = selected_agent.chain.stream({"messages": selected_agent.chat_history.messages})
                        response = st.write_stream(stream)
                else:
                    response = selected_agent.chain.invoke({"messages": selected_agent.chat_history.messages})
                    st.chat_message('system', avatar = character_avatar_emojis[character]).markdown(response.content)

                st.session_state.messages[character].append({'role': 'system', 'content': response})

                selected_agent.chat_history.add_ai_message(response)
                # Update the number of messages for the character
                st.session_state.number_of_messages[character] = len(selected_agent.chat_history.messages)

                # Warn if the chat has reached the permissible length
                if st.session_state.number_of_messages[character] == 2*permissible_length_of_chat:
                    st.warning(f"You reached the end of your conversation with {character}."
                               " Interview another character or if you are done with all the interviews, proceed to solve the case.")
                    chat_needs_to_end = True

elif radio == 'Solve':
    container_top.empty()
    container_bottom.empty()
    st.title('Solve the case')
    killer_guess = st.selectbox('Who do you think is the killer?', ['', 'Jennifer', 'Cindy', 'James'])
    if killer_guess in ['Jennifer', 'Cindy', 'James']:
        if killer_guess == st.session_state.killer:
            st.write('Congratulations! You have solved the case!')
        else:
            st.write('Sorry, you have guessed wrong! The killer is: ', st.session_state.killer)

else:
    st.title("Synopsis")
    st.write(st.session_state.backstory)
    st.write('The killer is: ', st.session_state.killer)
    for key in st.session_state.individual_povs.keys():
        st.write(key)
        st.write(st.session_state.individual_povs[key])