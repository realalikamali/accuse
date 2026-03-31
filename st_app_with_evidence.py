import json
import os
import random
import tempfile
from pathlib import Path

import gdown
import streamlit as st
from cryptography.fernet import Fernet
from PIL import Image

from models.character import CharacterGen
from models.controls import QuestionCap

from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache


@st.cache_resource
def get_llm_cache():
    return InMemoryCache()

set_llm_cache(get_llm_cache())

st.sidebar.title('Accuse')
st.sidebar.subheader('An Interactive Chat-based Adventure')

container_top = st.empty()
container_bottom = st.empty()

if 'api_key' not in st.session_state:
    st.session_state['api_key'] = st.secrets['openai_api_key']


@st.cache_resource
def fetch_all_stories() -> dict:
    """Download and decrypt all stories from Google Drive. Cached once per app instance."""
    folder_url = st.secrets['stories_google_drive_link']
    encryption_key = st.secrets['encryption_key'].encode()
    fernet = Fernet(encryption_key)

    stories = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        gdown.download_folder(url=folder_url, output=tmpdir, quiet=True, use_cookies=False)
        for enc_file in sorted(Path(tmpdir).glob('*.enc')):
            try:
                story_number = int(enc_file.stem.split('_')[1])
                with open(enc_file, 'rb') as f:
                    story_data = json.loads(fernet.decrypt(f.read()).decode('utf-8'))
                stories[story_number] = story_data
            except Exception as e:
                st.warning(f"Could not load {enc_file.name}: {e}")

    if not stories:
        st.error("No stories could be loaded from Google Drive.")
        st.stop()

    return stories


def load_story(stories: dict) -> tuple:
    """Pick a random story from the pre-loaded dict and initialise agents."""
    story_number = random.choice(list(stories.keys()))
    story_data = stories[story_number]

    killer = story_data['killer']
    backstory = story_data['backstory']
    pieces_of_evidence = story_data['pieces_of_evidence']
    individual_povs = story_data['individual_povs']

    questioncap = QuestionCap(st.session_state['api_key'])

    agents = {
        character_name: CharacterGen(
            individual_povs[character_name],
            character_name,
            os.path.join('prompts', character_name + '_backstory.txt'),
            st.session_state['api_key'],
            model_name='gpt-4o',
            temperature=0.1,
        )
        for character_name in ['Jennifer', 'Cindy', 'James']
    }

    return questioncap, killer, pieces_of_evidence, backstory, individual_povs, agents, story_number


permissible_length_of_chat = 2 * 5  # 2 rounds of 5 questions each
stream_response = True
character_avatar_emojis = {
    'Jennifer': '👩',
    'Cindy': '👩‍🦱',
    'James': '👨',
}

stories = fetch_all_stories()

if 'startup_data' not in st.session_state:
    st.session_state['startup_data'] = load_story(stories)

(
    questioncap,
    killer,
    pieces_of_evidence,
    backstory,
    individual_povs,
    agents,
    story_number,
) = st.session_state['startup_data']

if 'number_of_messages' not in st.session_state:
    st.session_state.number_of_messages = {c: 0 for c in ['Jennifer', 'Cindy', 'James']}

if 'messages' not in st.session_state:
    st.session_state.messages = {c: [] for c in ['Jennifer', 'Cindy', 'James']}

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

if 'story_number' not in st.session_state:
    st.session_state.story_number = story_number

radio = st.sidebar.radio(
    "Select mode:",
    ["Intro", "Evidence", "First-round Interview", "Second-round Interview", "Solve"],
    key="radio",
)

if radio == "Intro":
    with open(os.path.join('prompts', "welcome_message.txt"), "r") as file:
        welcome_message = file.read()
    welcome_message = welcome_message.format(
        number_of_interactions=int(permissible_length_of_chat / 2)
    )
    container_top.write(welcome_message)
    container_top.caption(f"Case #{st.session_state.story_number:02d}")
    cover_photo = Image.open("m_mystery_cover_photo.webp")
    container_bottom.image(cover_photo)

elif radio == "Evidence":
    container_top.header('Pieces of evidence')
    container_bottom.empty()
    pieces_of_evidence_description = ""
    for person in pieces_of_evidence:
        pieces_of_evidence_description += f"{person['name']}: {person['spoiler_free_description']}\n\n"
    container_bottom.write(pieces_of_evidence_description)

elif radio == 'First-round Interview':
    container_top.empty()
    container_bottom.empty()

    st.sidebar.title('Select Character to interview')
    character = st.sidebar.selectbox('Character', ['Jennifer', 'Cindy', 'James'])
    selected_agent = st.session_state.agents[character]

    for i, message in enumerate(st.session_state.messages[character]):
        if message['role'] == 'user':
            st.chat_message(message['role'], avatar='🕵️‍♂️').markdown(
                f'{int(i/2+1)}. ' + message['content']
            )
        else:
            st.chat_message(message['role'], avatar=character_avatar_emojis[character]).markdown(
                message['content']
            )

    prompt = st.chat_input("Enter your message here:", key="input_" + character)

    if prompt:
        st.session_state.more_than_one_question = questioncap.invoke(prompt)
        chat_needs_to_end = False

        for char, num_messages in st.session_state.number_of_messages.items():
            if 0 < num_messages < permissible_length_of_chat and char != character:
                st.warning(f"First-round interview with {char} needs to end before starting a new chat.")
                chat_needs_to_end = True
                break

        if st.session_state.number_of_messages[character] >= permissible_length_of_chat:
            st.warning(
                f"You reached the end of your conversation with {character}. "
                "Interview another character or if you are done with all the interviews, proceed to the second round."
            )
            chat_needs_to_end = True

        if not chat_needs_to_end:
            if st.session_state.more_than_one_question:
                st.warning("Ask only one question at a time.")
            else:
                st.chat_message('user', avatar='🕵️‍♂️').markdown(
                    f'{int(st.session_state.number_of_messages[character]/2+1)}. ' + prompt
                )
                selected_agent.chat_history.add_user_message(prompt)
                st.session_state.messages[character].append({'role': 'user', 'content': prompt})

                if stream_response:
                    with st.chat_message('system', avatar=character_avatar_emojis[character]):
                        stream = selected_agent.chain.stream({"messages": selected_agent.chat_history.messages})
                        response = st.write_stream(stream)
                else:
                    response = selected_agent.chain.invoke({"messages": selected_agent.chat_history.messages})
                    st.chat_message('system', avatar=character_avatar_emojis[character]).markdown(response.content)

                st.session_state.messages[character].append({'role': 'system', 'content': response})
                selected_agent.chat_history.add_ai_message(response)
                st.session_state.number_of_messages[character] = len(selected_agent.chat_history.messages)

                if st.session_state.number_of_messages[character] == permissible_length_of_chat:
                    st.warning(
                        f"You reached the end of your conversation with {character}. "
                        "Interview another character or if you are done with all the interviews, proceed to the second round."
                    )

elif radio == 'Second-round Interview':
    container_top.empty()
    container_bottom.empty()

    st.sidebar.title('Select Character to interview')
    character = st.sidebar.selectbox('Character', ['Jennifer', 'Cindy', 'James'])
    selected_agent = st.session_state.agents[character]

    for i, message in enumerate(st.session_state.messages[character]):
        if message['role'] == 'user':
            st.chat_message(message['role'], avatar='🕵️‍♂️').markdown(
                f'{int(i/2+1)}. ' + message['content']
            )
        else:
            st.chat_message(message['role'], avatar=character_avatar_emojis[character]).markdown(
                message['content']
            )

    prompt = st.chat_input("Enter your message here:", key="input_" + character)

    if prompt:
        st.session_state.more_than_one_question = questioncap.invoke(prompt)
        chat_needs_to_end = False

        for char, num_messages in st.session_state.number_of_messages.items():
            if 0 <= num_messages < permissible_length_of_chat:
                st.warning("First round interviews need to end before starting second round interviews.")
                chat_needs_to_end = True
                break

        for char, num_messages in st.session_state.number_of_messages.items():
            if permissible_length_of_chat < num_messages < 2 * permissible_length_of_chat and char != character:
                st.warning(f"Chat with {char} needs to end before continuing this chat.")
                chat_needs_to_end = True
                break

        if st.session_state.number_of_messages[character] >= 2 * permissible_length_of_chat:
            st.warning(
                f"You reached the end of your conversation with {character}. "
                "Interview another character or if you are done with all the interviews, proceed to solve the case."
            )
            chat_needs_to_end = True

        if not chat_needs_to_end:
            if st.session_state.more_than_one_question:
                st.warning("Ask only one question at a time.")
            else:
                st.chat_message('user', avatar='🕵️‍♂️').markdown(
                    f'{int(st.session_state.number_of_messages[character]/2+1)}. ' + prompt
                )
                selected_agent.chat_history.add_user_message(prompt)
                st.session_state.messages[character].append({'role': 'user', 'content': prompt})

                if stream_response:
                    with st.chat_message('system', avatar=character_avatar_emojis[character]):
                        stream = selected_agent.chain.stream({"messages": selected_agent.chat_history.messages})
                        response = st.write_stream(stream)
                else:
                    response = selected_agent.chain.invoke({"messages": selected_agent.chat_history.messages})
                    st.chat_message('system', avatar=character_avatar_emojis[character]).markdown(response.content)

                st.session_state.messages[character].append({'role': 'system', 'content': response})
                selected_agent.chat_history.add_ai_message(response)
                st.session_state.number_of_messages[character] = len(selected_agent.chat_history.messages)

                if st.session_state.number_of_messages[character] == 2 * permissible_length_of_chat:
                    st.warning(
                        f"You reached the end of your conversation with {character}. "
                        "Interview another character or if you are done with all the interviews, proceed to solve the case."
                    )

elif radio == 'Solve':
    container_top.empty()
    container_bottom.empty()

    st.title('Solve the case')
    killer_guess = st.selectbox('Who do you think is the killer?', ['', 'Jennifer', 'Cindy', 'James'])
    if killer_guess in ['Jennifer', 'Cindy', 'James']:
        if killer_guess == st.session_state.killer:
            st.write('Congratulations! You have solved the case!')
            st.title("Synopsis")
            st.write(st.session_state.backstory)
            st.write('The killer is ', st.session_state.killer, '.')
        else:
            st.write(f'Sorry, you have guessed wrong! The killer is {st.session_state.killer}.')
            st.title("Synopsis")
            st.write(st.session_state.backstory)
            st.write(f'The killer is {st.session_state.killer}.')
