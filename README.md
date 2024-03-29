# Chat with Your Data

## Introduction

This project facilitates chatting with your PDF data and can run locally, eliminating dependency on OpenAI API calls. This project is using [Langchain](https://python.langchain.com/docs/get_started/introduction) + [Ollama](https://ollama.ai/) + [Mistral](https://mistral.ai/news/announcing-mistral-7b/) + [streamlit-chat](https://github.com/AI-Yash/st-chat).




https://github.com/arsalansaleem96/chat-with-your-data/assets/13997274/74e316fb-4ef0-4511-bed7-6d9ae28d8d7a



# Getting Started

## Usage

- Download and Install Ollama from [here](https://ollama.ai/download).
- Download Mistral from [here](https://ollama.ai/library/mistral), an open-source model available at [Mistral](https://mistral.ai/news/announcing-mistral-7b/).
- Optionally, create a Python virtual environment.
- Run the following command: `pip install -r requirements.txt`.
- Finally, execute `streamlit run app.py`.

## Features

- Multiple File Uploads: Users can upload multiple files simultaneously.
- Conversation Memory: The system retains chat history, allowing users to ask follow-up questions based on previous interactions.

# Acknowledgments

- This project is inspired by Harrison Chase's course on deeplearning.ai [here](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/).
- The implementation is inspired by Duy Huynh's article on Medium.com [here](https://medium.com/@vndee.huynh/build-your-own-rag-and-run-it-locally-langchain-ollama-streamlit-181d42805895).
