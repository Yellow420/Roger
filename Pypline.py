from pypline_utils import *

class Synthesis():
    def __init__(self):
        pass

    def Text(self, text):
        result = Response(text)
        return result

    def Speech(self, text, gmm=None):
        if gmm:
            tts_with_gmm(text, gmm)
        else:
            tts(text)

class SpeechRecognition():
    def __init__(self):
        pass

    async def APSR(self, wake_word=None, timer=None, word_count=None, gmm_dir=None, unknown_speakers=False, output="text"):
        async for result in async_record_and_transcribe(wake_word, timer, word_count, gmm_dir, unknown_speakers,
                                                        output):
            yield result

    def ASR(self, wake_word=None, timer=1, word_count=None, gmm_dir=None, unknown_speakers=False, output="text"):
        results = record_and_transcribe(wake_word, timer, word_count, gmm_dir, unknown_speakers, output)
        print(f"Results:{results}")
        return results

class InputRecognition:
    def __init__(self):
        self.synthesizer = Synthesis()
    def Speech(self, dialogue=None,gmm=None, need_audio=False):
        if dialogue:
            # Use TTS to generate audio
            if gmm:
                tts_with_gmm(dialogue, gmm)
            else:
                tts(dialogue)


        if need_audio:
            # ASR with custom ASR engine
            result_text, result_audio = record_and_transcribe(output="both")
            return result_text, result_audio
        else:
            # ASR without audio
            result_text = record_and_transcribe(output="text")
            return result_text

    def Text(self, dialogue=None, choice=None):
        root = Tk()
        root.withdraw()  # Hide the main window
        result = None
        if dialogue:
            if choice:
                result = simpledialog.askstring("Input", dialogue, initialvalue=choice[0], parent=root)
            else:
                result = simpledialog.askstring("Input", dialogue, parent=root)
        else:
            if choice:
                result = simpledialog.askstring("Input", "Click on your choice:", initialvalue=choice[0], parent=root)
            else:
                result = simpledialog.askstring("Input", "Enter text:", parent=root)
        return result

    def Path(self, dialogue=None, extension=None):
        root = Tk()
        root.withdraw()  # Hide the main window

        if dialogue:
            dialog_label = tkinter.Label(root, text=dialogue)
            dialog_label.pack()

        file_path = filedialog.askopenfilename()

        if extension and not file_path.endswith(extension):
            print(f"Selected file must have extension '{extension}'")
            file_path = None

        return file_path

class CommandRecognition:
    def __init__(self):
        pass

    def Text(self, text):
        if process_commands(text):
            return True
        else:
            return False

    def Speech(self, wake_word=None, gmm_dir=None, unknown_speakers=True):
        for result in record_and_transcribe(wake_word, gmm_dir, unknown_speakers, output="text"):
            dialog_string = result
            print(dialog_string)
            if process_commands(dialog_string):
                continue

class Profiler:
    def __init__(self):
        self.get = InputRecognition()
    def Create(self, key, path=None):
        result = {}

        # Iterate through each key-value pair in the key dictionary
        for k, v in key.items():
            if v == "get_speech":
                # Call InputRecognition.Speech and replace the key's value
                result[k] = self.get.Speech(f"Please tell me your {k}")
            elif v == "get_text":
                # Call InputRecognition.Text and replace the key's value
                result[k] = self.get.Text(f"Please enter your {k}")
            elif v == "get_path":
                # Call InputRecognition.Path and replace the key's value
                result[k] = self.get.Path(f"Please select your {k}")
            else:
                # Keep the original value if not "get_speech", "get_text", or "get_path"
                result[k] = v

        # Save the profile
        if path:
            profile_path = os.path.join(path, "profile.json")
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            profile_path = f"profiles/{timestamp}/profile.json"

        os.makedirs(os.path.dirname(profile_path), exist_ok=True)
        with open(profile_path, "w") as f:
            json.dump(result, f, indent=4)

        return result

    def Update(self, profile, key):
        # Update existing keys or add new ones
        for k, v in key.items():
            if k in profile:
                if v == "get_speech":
                    profile[k] = self.get.Speech(f"Please tell me your {k}")
                elif v == "get_text":
                    profile[k] = self.get.Text(f"Please enter your {k}")
                elif v == "get_path":
                    profile[k] = self.get.Path(f"Please select your {k}")
                else:
                    profile[k] = v
            else:
                # Add new key-value pair
                profile[k] = v

        # Save the updated profile
        with open("profile.json", "w") as f:
            json.dump(profile, f, indent=4)

        return profile

    def Delete(self, profile):
        # Get the directory path of the profile file
        profile_dir = os.path.dirname(profile)

        # Delete the directory and its contents
        if os.path.exists(profile_dir):
            os.rmdir(profile_dir)
            print(f"Profile '{profile}' and its contents have been deleted.")
        else:
            print(f"Profile directory '{profile_dir}' does not exist.")