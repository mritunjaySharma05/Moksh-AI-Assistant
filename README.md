# Moksh AI - Personal Voice Assistant

This is a desktop personal voice assistant built with Python. It features a modern dark-mode GUI, natural voice interaction, and is powered by a locally-running LLM (Llama 3 via Ollama) for intelligent, private conversations.

## Features

* **Modern UI:** A clean, dark-mode graphical interface built with `tkinter`.
* **Voice Interaction:** Uses high-quality Google Text-to-Speech (`gTTS`) for responses and `SpeechRecognition` for listening.
* **Local & Private LLM:** Connects to a locally-running Llama 3 model using Ollama. Your conversations stay on your machine.
* **Conversational Memory:** Remembers the context of your chat for natural follow-up questions.
* **Real-Time Information:**
    * Get live weather forecasts (via OpenWeatherMap).
    * Get the current time and date.
* **Productivity:**
    * Open websites (Google, YouTube).
    * Perform Google searches.
    * Get summaries from Wikipedia.
    * Open local applications (e.g., Notepad).
* **Entertainment:**
    * Can tell you jokes.

## How It Works

1.  You click the **"Start Listening"** button.
2.  Your voice is captured by `SpeechRecognition` and converted to text.
3.  The text is processed by `ui_assistant.py`.
4.  If it's a specific command (like "open YouTube"), it executes the action.
5.  If it's a general question, it's sent (with conversation history) to the Ollama server.
6.  Ollama and Llama 3 process the question and send back a text response.
7.  The response text is spoken aloud using `gTTS`.
8.  The conversation is updated in the UI.

## Installation

Follow these steps to get Moksh AI running on your computer.

### 1. Backend: Install Ollama and Llama 3

You must have the LLM backend running first.

1.  Download and install **Ollama** from [ollama.com](https://ollama.com/).
2.  Open your terminal or command prompt and run the following command to download the Llama 3 model:
    ```
    ollama run llama3
    ```
3.  Once it's running, **keep this terminal open** in the background.

### 2. Project: Download and Set Up

1.  **Get the code:**
    * Clone this repository: `git clone [repository_url]`
    * Or, download the files (`ui_assistant.py`, `requirements.txt`, `.gitignore`) and save them in a new folder.
2.  **Create a Virtual Environment (Recommended):**
    ```
    python -m venv venv
    source vD/bin/activate  # On Linux/macOS
    venv\Scripts\activate     # On Windows
    ```
3.  **Install Python Libraries:**
    Install all the required packages using the `requirements.txt` file:
    ```
    pip install -r requirements.txt
    ```

### 3. API Key: Weather Forecasts

1.  Go to [OpenWeatherMap](https://openweathermap.org/) and create a free account.
2.  Go to your dashboard and find your "API keys".
3.  Open the `ui_assistant.py` file in a text editor.
4.  Find this line:
    ```
    OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY"
    ```
5.  Replace `"YOUR_OPENWEATHER_API_KEY"` with the key you just copied.

## How to Run

1.  Make sure your Ollama server is running in a terminal (`ollama run llama3`).
2.  Open a **new** terminal, navigate to your project folder, and activate your virtual environment (if you made one).
3.  Run the main application:
    ```
    python ui_assistant.py
    ```
The assistant window will pop up. Click the "Start Listening" button to give a command.
