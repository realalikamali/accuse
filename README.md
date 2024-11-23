## Accuse
Welcome to Accuse, an interactive chat-based adventure, where you take the role of a detective and interview suspects to solve a mystery.
Click [here](https://accuse.streamlit.app/) for a live demo of the game deployed on Streamlit Cloud.
![alt text](https://github.com/realalikamali/accuse/blob/main/m_mystery_cover_photo.webp)

This project explores the capability of language models to facilitate interactive, engaging, and entertaining ways of storytelling.

The game is built using a combination of Large Language Model (LLM) agents. The backstory is generated every time using a custom prompt and random assignment of the culprit. Additionally, each character that the player interviews is an independent agent having access to its history of events.

The interactions in the game are managed using a Streamlit web app.
![accuse6](https://github.com/user-attachments/assets/6221850a-ce30-4743-bc08-64792b2a4b4a)

# Usage
Install dependencies in your virtual environment using the `environment.yml` file and place your openai API key inside a .env file in the main directory.

Run either `st_app_with_evidence.py` or `st_app.py` in the activated environment using

`streamlit run <file_name.py>`
