import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import mainthread
from Pypline import *

# Global state and threading lock
state = "stopped"
state_lock = threading.Lock()

# Initialize Pypline components
generate = Synthesis()
recognizer = SpeechRecognition()
commands = CommandRecognition()


# Brain function to handle listening
def brain():
    global state
    while True:
        with state_lock:
            if state != "listening":
                break  # Exit loop if state changes

        dialog = recognizer.ASR(wake_word="rodger", gmm_dir="speaker_test_gmm", unknown_speakers=False)
        print(f"Dialogue: {dialog}")

        if not dialog:
            print("No valid input detected, retrying...")
            continue
        elif commands.Text(dialog):
            continue
        else:
            response = generate.Text(dialog)
            print(response)
            generate.Speech(response)


# Function to stop listening
def stop_listening():
    global state
    with state_lock:
        state = "stopped"


# GUI class definition
class MainGUI(BoxLayout):
    def __init__(self, **kwargs):
        super(MainGUI, self).__init__(**kwargs)
        self.orientation = 'vertical'

        # Create buttons
        self.start_button = Button(text="Start Listening", on_press=self.start_listening)
        self.stop_button = Button(text="Stop Listening", on_press=self.stop_listening)

        # Initially add the Start Listening button
        self.add_widget(self.start_button)

        # Continuously check the global state to update buttons
        self.update_button_state()

    @mainthread
    def update_button_state(self):
        # Update the buttons based on the global state
        with state_lock:
            if state == "listening":
                if self.start_button in self.children:
                    self.remove_widget(self.start_button)
                if self.stop_button not in self.children:
                    self.add_widget(self.stop_button)
            else:
                if self.stop_button in self.children:
                    self.remove_widget(self.stop_button)
                if self.start_button not in self.children:
                    self.add_widget(self.start_button)

        # Schedule another update after 500ms
        self.schedule_update()

    def schedule_update(self):
        from kivy.clock import Clock
        Clock.schedule_once(lambda dt: self.update_button_state(), 0.5)

    def start_listening(self, instance):
        global state
        with state_lock:
            state = "listening"
        # Start the brain function in a separate thread
        threading.Thread(target=brain, daemon=True).start()

    def stop_listening(self, instance):
        # Call the function to stop listening
        stop_listening()


# Main app class
class MyApp(App):
    def build(self):
        return MainGUI()


if __name__ == '__main__':
    MyApp().run()
