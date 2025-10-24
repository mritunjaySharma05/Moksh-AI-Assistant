import tkinter as tk
from tkinter import scrolledtext, DISABLED, END
import threading
import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import wikipedia
import pyjokes
import requests
import json
import os
from gtts import gTTS
from playsound import playsound
import time

# --- CONFIGURATION ---
ASSISTANT_NAME = "Moksh AI"
OLLAMA_MODEL = "llama3"
# --- IMPORTANT ---
# Get your free API key from https://openweathermap.org/
OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY" 
DEFAULT_CITY = "Kangra" #you can  add your city here 
conversation_history = []

# --- UI THEME ---
BG_COLOR = "#1a1a1a"
TEXT_COLOR = "#EAECEE"
CHAT_BG_COLOR = "#2c2c2c"
BUTTON_BG_COLOR = "#1E88E5" # A nice blue
BUTTON_FG_COLOR = "#FFFFFF"
FONT_FAMILY = "Segoe UI" # A cleaner font, falls back to Arial/system default
FONT_BOLD = (FONT_FAMILY, 12, "bold")
FONT_NORMAL = (FONT_FAMILY, 11)


def speak(text):
    """Converts text to speech using Google's TTS and plays it."""
    print(f"{ASSISTANT_NAME}: {text}")
    try:
        tts = gTTS(text=text, lang='en')
        filename = "temp_audio.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f"gTTS Error: {e}")
        # Fallback to offline engine if gTTS or playsound fails
        try:
            engine = pyttsx3.init('sapi5')
            engine.say(text)
            engine.runAndWait()
        except Exception as pyttsx_e:
            print(f"Fallback pyttsx3 error: {pyttsx_e}")


class AssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{ASSISTANT_NAME} - Personal Assistant")
        self.root.geometry("600x700")
        self.root.configure(bg=BG_COLOR)

        # --- Header ---
        header_label = tk.Label(root, text=ASSISTANT_NAME, font=(FONT_FAMILY, 16, "bold"), bg=BG_COLOR, fg=BUTTON_BG_COLOR, pady=10)
        header_label.pack(fill=tk.X)

        # --- Chat Window ---
        self.chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=DISABLED, 
                                                     font=FONT_NORMAL, bg=CHAT_BG_COLOR, fg=TEXT_COLOR,
                                                     bd=0, relief="flat", padx=10, pady=10)
        self.chat_window.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        # Configure tags for different speakers
        self.chat_window.tag_config('user', foreground="#82E0AA") # Light green for user
        self.chat_window.tag_config('assistant', foreground=TEXT_COLOR)
        self.chat_window.tag_config('bold', font=FONT_BOLD)

        # --- Status Label ---
        self.status_label = tk.Label(root, text="Click the button to start.", font=(FONT_FAMILY, 10), bg=BG_COLOR, fg=TEXT_COLOR, pady=5)
        self.status_label.pack(fill=tk.X)
        
        # --- Listen Button ---
        self.listen_button = tk.Button(root, text="Start Listening", command=self.start_listening_thread, 
                                       font=(FONT_FAMILY, 14, "bold"), bg=BUTTON_BG_COLOR, fg=BUTTON_FG_COLOR,
                                       activebackground="#1565C0", activeforeground="white", bd=0, relief="flat", padx=20, pady=10)
        self.listen_button.pack(pady=20)
        
        self.wish_me()

    def update_chat(self, sender, message):
        """Updates the chat window with colored text."""
        def _update():
            self.chat_window.config(state='normal')
            
            sender_tag = 'user' if sender == "You" else 'assistant'
            self.chat_window.insert(tk.END, f"{sender}: ", ('bold', sender_tag))
            self.chat_window.insert(tk.END, f"{message}\n\n", (sender_tag,))

            self.chat_window.config(state='disabled')
            self.chat_window.yview(tk.END)
        self.root.after(0, _update)

    def update_status(self, message):
        """Updates the status label."""
        def _update():
            self.status_label.config(text=message)
        self.root.after(0, _update)
        
    def start_listening_thread(self):
        """Starts the main assistant logic in a separate thread."""
        self.listen_button.config(state=DISABLED, bg="#555555", text="Listening...")
        thread = threading.Thread(target=self.run_assistant, daemon=True)
        thread.start()
        
    def wish_me(self):
        hour = int(datetime.datetime.now().hour)
        greeting = ""
        if 0 <= hour < 12: greeting = "Good Morning!"
        elif 12 <= hour < 18: greeting = "Good Afternoon!"
        else: greeting = "Good Evening!"
        
        welcome_message = f"{greeting} I am {ASSISTANT_NAME}. How can I help you today?"
        self.update_chat(ASSISTANT_NAME, welcome_message)
        threading.Thread(target=lambda: speak(welcome_message), daemon=True).start()

    def run_assistant(self):
        """Main loop for listening and processing commands."""
        query = self.take_command()

        if query and query != "None":
            self.update_chat("You", query)
            self.process_query(query)

        # Re-enable the button after processing is complete
        self.root.after(0, lambda: self.listen_button.config(state='normal', bg=BUTTON_BG_COLOR, text="Start Listening"))

    def take_command(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            self.update_status("Listening...")
            r.pause_threshold = 1
            r.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                self.update_status("Listening timed out. Click the button to try again.")
                return None

        try:
            self.update_status("Recognizing...")
            query = r.recognize_google(audio, language='en-in')
        except Exception as e:
            self.update_status(f"Could not recognize your voice. Error: {e}")
            return "None"
        
        self.update_status("Idle. Click the button to speak.")
        return query.lower()

    def process_query(self, query):
        """Handles the logic for different commands."""
        response = ""
        action_taken = False
        
        if 'wikipedia' in query:
            action_taken = True
            self.update_status('Searching Wikipedia...')
            query = query.replace("wikipedia", "").replace("search for", "").strip()
            try:
                results = wikipedia.summary(query, sentences=2)
                response = f"According to Wikipedia: {results}"
            except Exception:
                response = f"Sorry, I could not find any results for {query} on Wikipedia."

        elif 'open youtube' in query:
            action_taken = True
            response = "Opening YouTube"
            webbrowser.open("https://youtube.com")

        elif 'open google' in query:
            action_taken = True
            response = "Opening Google"
            webbrowser.open("https://google.com")
            
        elif 'open notepad' in query:
            action_taken = True
            response = "Opening Notepad"
            # Use os.system for cross-platform compatibility (works on Windows)
            # For macOS: os.system('open -a TextEdit')
            # For Linux: os.system('gedit')
            if os.name == 'nt': # Windows
                os.system('notepad.exe')
            else:
                response = "Opening applications is only configured for Windows."


        elif 'clear conversation' in query or 'forget everything' in query:
            action_taken = True
            global conversation_history
            conversation_history = []
            response = "I've cleared our conversation history. We can start fresh."

        elif 'weather' in query:
            action_taken = True
            words = query.split()
            city = DEFAULT_CITY
            if "in" in words:
                try:
                    city_index = words.index("in") + 1
                    if city_index < len(words):
                        city = words[city_index]
                except ValueError:
                    pass
            response = self.get_weather(city)

        elif 'the time' in query:
            action_taken = True
            str_time = datetime.datetime.now().strftime("%I:%M %p")
            response = f"The time is {str_time}"

        elif 'the date' in query:
            action_taken = True
            str_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            response = f"Today is {str_date}"

        elif 'tell me a joke' in query:
            action_taken = True
            response = pyjokes.get_joke()

        elif 'goodbye' in query or 'stop' in query or 'exit' in query:
            action_taken = True
            response = "Goodbye! Have a great day."
            self.root.after(2000, self.root.destroy)

        elif 'search for' in query:
            action_taken = True
            search_query = query.replace("search for", "").strip()
            response = f"Searching Google for {search_query}"
            webbrowser.open(f"https://www.google.com/search?q={search_query}")

        if not action_taken:
            response = self.ask_llm(query, conversation_history)
            if response and response != "Error":
                conversation_history.append(f"User: {query}")
                conversation_history.append(f"{ASSISTANT_NAME}: {response}")
                if len(conversation_history) > 10: # Keep memory to last 5 interactions (10 items)
                    conversation_history = conversation_history[-10:]

        if response:
            self.update_chat(ASSISTANT_NAME, response)
            speak(response)

    def get_weather(self, city):
        if OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY":
            return "Weather service is not configured. Please add your OpenWeatherMap API key."
        base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        try:
            response_data = requests.get(base_url).json()
            main = response_data["main"]
            report = f"The temperature in {city} is {main['temp']}° Celsius with {response_data['weather'][0]['description']}."
            return report
        except Exception:
            return f"Sorry, I couldn't find the weather for {city}."

    def ask_llm(self, question, history):
        self.update_status("Thinking...")
        full_prompt = "\n".join(history) + f"\nUser: {question}\n{ASSISTANT_NAME}:"
        try:
            payload = {"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False}
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "No response from model.").strip()
        except requests.exceptions.ConnectionError:
            self.update_status("Ollama connection error. Is it running?")
            return "Error: Could not connect to Ollama. Please ensure it is running."
        except Exception as e:
            self.update_status(f"An error occurred: {e}")
            return f"Error: An issue occurred with the LLM. {e}"


if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantApp(root)
    root.mainloop()
