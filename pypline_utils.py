import requests
import json
import torch
import tempfile
import inflect
import threading
import pyaudio
from librosa.feature import spectral_centroid
from scipy.signal import butter, sosfilt
import math
import noisereduce as nr
import resampy
import joblib
from transformers import pipeline, SpeechT5ForTextToSpeech
import soundfile as sf
import simpleaudio as sa
import os
import shutil
from datetime import timedelta
from pyannote.audio import Inference, Pipeline
import pyttsx3
import torchaudio
import librosa
from speechbrain.inference.speaker import EncoderClassifier
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
import speech_recognition as sr
from pydub import AudioSegment
import time
from datetime import datetime
import pyautogui
import keyboard
import mouse
import ctypes
import psutil
import win32gui
import pygetwindow as gw
from pywinauto import application
import tkinter
import re
from tkinter import Tk, simpledialog, filedialog
# Paths for command storage
CUSTOM_COMMANDS_FOLDER = "Mods/CUSTOMCOMMANDS"
COMMANDS_JSON_FOLDER = os.path.join(CUSTOM_COMMANDS_FOLDER, "JSONS")
root = None
# Global variable to control recording
is_recording = False
command_actions = []
ROOT_URL = "https://api.ai21.com/studio/v1/"
trial = False
trial_timer = 0
# Get the current date and time
now = datetime.now()

# Format the date and time
formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

'''Text Utilities'''

class TextMetrics:
    def __init__(self, word_count, token_count):
        self.word_count = word_count
        self.token_count = token_count

def tokenize(text):
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

def measure_text(text):
    if isinstance(text, str):
        texts = [text]
    elif isinstance(text, list):
        texts = text
    else:
        raise ValueError("Input should be a string or a list of strings")

    total_word_count = 0
    total_token_count = 0

    for txt in texts:
        words = txt.split()
        tokens = tokenize(txt)
        total_word_count += len(words)
        total_token_count += len(tokens)

    return TextMetrics(word_count=total_word_count, token_count=total_token_count)


def check_text(text, hotword, case_sense=False, non_letters=False):
    # Convert hotword to a list if it's a single string
    if isinstance(hotword, str):
        hotword = [hotword]

    # Function to remove special characters from a string
    def remove_special_chars(s):
        return re.sub(r'[^A-Za-z0-9\s]', '', s)

    # Handle case sensitivity
    if not case_sense:
        print("Not case sensitive")
        text = text.lower()
        hotword = [word.lower() for word in hotword]
        print(f"Hotword: ({hotword}) Text: ({text}) ")

    # Handle non-letters: remove special characters and replace numbers with words
    if not non_letters:
        print("No Non Letters")
        text = remove_special_chars(text)
        text = number_to_text(text)

        hotword = [remove_special_chars(word) for word in hotword]
        hotword = [number_to_text(word) for word in hotword]
        print(f"Hotword: ({hotword}) Text: ({text}) ")

    # Check for hotwords in the text
    found_hotwords = [word for word in hotword if word in text]

    return found_hotwords if found_hotwords else False

def detokenize(tokens):
    # Improved detokenize function
    text = ' '.join(tokens)
    text = re.sub(r"\s([.,!?;:])", r"\1", text)  # Remove space before punctuation
    text = text.replace(" ' ", "'").replace(" n't", "n't")
    return text

def crop_text(text, options):
    original_text = text  # Save the original text for rev_crop

    # Handle token cropping
    if 'token_count' in options:
        token_count = options['token_count']
        crop_end = options.get('crop_end', False)
        tokens = tokenize(text)
        if crop_end:
            tokens = tokens[:-token_count]
        else:
            tokens = tokens[token_count:]
        text = detokenize(tokens)

    # Handle word cropping
    if 'word_count' in options:
        word_count = options['word_count']
        crop_end = options.get('crop_end', False)
        words = text.split()
        if crop_end:
            words = words[:-word_count]
        else:
            words = words[word_count:]
        text = ' '.join(words)

    # Handle start_word cropping
    if 'start_word' in options:
        start_word = options['start_word']
        start_index = text.find(start_word)
        if start_index != -1:
            text = text[start_index:]

    # Handle stop_word cropping
    if 'stop_word' in options:
        stop_word = options['stop_word']
        stop_index = text.rfind(stop_word)
        if stop_index != -1:
            text = text[:stop_index + len(stop_word)]

    # Handle crop_text removal
    if 'crop_text' in options:
        crop_phrases = options['crop_text']
        if isinstance(crop_phrases, str):
            crop_phrases = [crop_phrases]
        for phrase in crop_phrases:
            text = text.replace(phrase, '')

    # Handle rev_crop to return cropped text instead of remaining text
    if options.get('rev_crop', False):
        return original_text.replace(text, '')

    return text



def split_text(text, word_limit):
    words = text.split()
    chunks = []

    for i in range(0, len(words), word_limit):
        chunk = ' '.join(words[i:i + word_limit])
        chunks.append(chunk)

    return chunks


def number_to_text(text):
    p = inflect.engine()

    # Regular expression to find all numbers in the text
    number_pattern = re.compile(r'\d+')

    def replace_number_with_text(match):
        number_str = match.group()
        number_int = int(number_str)
        return p.number_to_words(number_int)

    # Substitute all numbers in the text using the above function
    converted_text = number_pattern.sub(replace_number_with_text, text)

    return converted_text

def prompt_formatter(system_prompt, user_prompt, assistant_prompt=None, context=None, input_limit=None):
    if input_limit:
        max_word_count = input_limit
    else:
        max_word_count = 8000

    system_user_text = f"system\n\n{system_prompt}\n\nuser\n\n{user_prompt}\n\n"
    metrics = measure_text(system_user_text)
    total_word_count = metrics.word_count

    if total_word_count > max_word_count:
        excess_words = total_word_count - max_word_count
        user_prompt = crop_text(user_prompt, {'word_count': excess_words, 'crop_end': False})
        return system_prompt, user_prompt, None, None

    remaining_word_count = max_word_count - total_word_count

    context_text = "\n\n".join([f"{msg['role']}\n\n{msg['content']}" for msg in context]) if context else ""
    assistant_text = assistant_prompt if assistant_prompt else ""

    context_metrics = measure_text(context_text)
    assistant_metrics = measure_text(assistant_text)

    total_context_assistant_words = context_metrics.word_count + assistant_metrics.word_count

    if total_context_assistant_words <= remaining_word_count:
        return system_prompt, user_prompt, context, assistant_prompt
    else:
        if assistant_metrics.word_count <= remaining_word_count:
            remaining_word_count -= assistant_metrics.word_count
            context_text = crop_text(context_text, {'word_count': context_metrics.word_count - remaining_word_count, 'crop_end': True})
            cropped_context = [{"role": msg.split("\n\n")[0], "content": msg.split("\n\n")[1]} for msg in context_text.split("\n\nassistant\n\n") if msg]
            return system_prompt, user_prompt, cropped_context, assistant_prompt
        else:
            assistant_text = crop_text(assistant_text, {'word_count': assistant_metrics.word_count - remaining_word_count, 'crop_end': True})
            return system_prompt, user_prompt, context, assistant_text

def load_api_key(file_path):
    global trial
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            api_key_match = re.search(r'API="(.+?)"', content)
            free_api_match = re.search(r'FREE_API="(.+?)"', content)
            if api_key_match:
                return api_key_match.group(1)
            elif free_api_match:
                trial = True
                return free_api_match.group(1)
            else:
                print("No valid API key found in the file.")
                return None
    except FileNotFoundError:
        print("API key file not found.")
        return None

def check_trial_timer():
    global trial_timer
    if trial:
        while trial_timer < 1:
            time.sleep(0.1)
            trial_timer += 0.1
        trial_timer = 0

def generate_text(prompt):
    check_trial_timer()

    api_key = load_api_key('API.txt')
    try:
        response = requests.post(
            f'{ROOT_URL}j2-mid/complete',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            data=json.dumps({
                'prompt': prompt,
                'maxTokens': 150,
                'temperature': 0.7
            })
        )
        response.raise_for_status()
        result = response.json()
        text = result.get('completions', [{}])[0].get('data', {}).get('text', '').strip()
        return text
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except json.JSONDecodeError:
        print("Failed to parse response JSON.")
        return None

def generate_text_adv(data):
    check_trial_timer()

    api_key = load_api_key('API.txt')
    system_prompt = data.get('system_prompt')
    user_prompt = data.get('user_prompt')
    assistant_prompt = data.get('assistant_prompt')
    history = data.get('history', [])

    other_keys = {key: value for key, value in data.items() if key not in {'system_prompt', 'user_prompt', 'assistant_prompt', 'history'}}

    history_list = []
    if history:
        roles = ["user", "assistant"]
        history_list.extend([{"role": roles[i % 2], "content": history[i]} for i in range(len(history))])

    if system_prompt:
        history_list.insert(0, {"role": "system", "content": system_prompt})
    if other_keys:
        for key, value in other_keys.items():
            history_list.append({"role": "system", "content": value})
    if assistant_prompt:
        history_list.append({"role": "assistant", "content": assistant_prompt})
    if user_prompt:
        history_list.append({"role": "user", "content": user_prompt})

    try:
        url = f'{ROOT_URL}chat/completions'
        response = requests.post(
            url,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'jamba-instruct',
                'messages': history_list,
                'max_tokens': 200,
                'temperature': 1.3
            }
        )
        response.raise_for_status()
        result = response.json()
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return text
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except json.JSONDecodeError:
        print("Failed to parse response JSON.")
        return None

def text_segmentation(source, source_type='URL'):
    check_trial_timer()

    api_key = load_api_key('API.txt')
    try:
        response = requests.post(
            f'{ROOT_URL}segmentation',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'source': source,
                'sourceType': source_type
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except json.JSONDecodeError:
        print("Failed to parse response JSON.")
        return None

def summarize_text(source, source_type='TEXT', focus=None):
    check_trial_timer()

    api_key = load_api_key('API.txt')
    try:
        payload = {
            'source': source,
            'sourceType': source_type
        }
        if focus:
            payload['focus'] = focus

        response = requests.post(
            f'{ROOT_URL}summarize',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except json.JSONDecodeError:
        print("Failed to parse response JSON.")
        return None

def summarize_by_segment(source, source_type='TEXT', focus=None):
    check_trial_timer()

    api_key = load_api_key('API.txt')
    try:
        payload = {
            'source': source,
            'sourceType': source_type
        }
        if focus:
            payload['focus'] = focus

        response = requests.post(
            f'{ROOT_URL}summarize-by-segment',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except json.JSONDecodeError:
        print("Failed to parse response JSON.")
        return None

def contextual_answer(context, question):
    check_trial_timer()

    api_key = load_api_key('API.txt')
    try:
        response = requests.post(
            f'{ROOT_URL}answer',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'context': context,
                'question': question
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except json.JSONDecodeError:
        print("Failed to parse response JSON.")
        return None

# Response function to handle both types of input
def Response(input_data):
    if isinstance(input_data, str):
        return generate_text(input_data)
    elif isinstance(input_data, dict):
        return generate_text_adv(input_data)
    else:
        raise ValueError("Input must be either a string or a dictionary.")

'''Audio Utilities'''

def combine_audio_data(audio_data_list):
    combined = AudioSegment.empty()

    for audio_data in audio_data_list:
        if isinstance(audio_data, AudioSegment):
            # If the audio_data is already an AudioSegment, just add it
            combined += audio_data
        elif hasattr(audio_data, 'get_wav_data') and hasattr(audio_data, 'sample_rate'):
            # If the audio_data has get_wav_data and sample_rate, convert it to an AudioSegment
            segment = AudioSegment(
                data=audio_data.get_wav_data(),
                sample_width=2,  # Assuming 16-bit audio
                frame_rate=audio_data.sample_rate,
                channels=1  # Assuming mono audio
            )
            combined += segment
        else:
            raise TypeError("Unsupported audio data type.")

    return combined

def play_audio(audio_segment):
    # Extract raw audio data
    raw_data = audio_segment.raw_data

    # Get frame rate, sample width, and channels from AudioSegment
    frame_rate = audio_segment.frame_rate
    sample_width = audio_segment.sample_width
    channels = audio_segment.channels

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream
    stream = p.open(format=p.get_format_from_width(sample_width),
                    channels=channels,
                    rate=frame_rate,
                    output=True)

    # Play the audio
    stream.write(raw_data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    p.terminate()
def dub_array(audio_dict):
    # Step 1: Save numpy array as a temporary audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file_name = temp_file.name
        # Save numpy array as WAV file using soundfile
        sf.write(temp_file_name, audio_dict["audio"], samplerate=audio_dict["sampling_rate"])

    # Step 2: Reload the audio segment using pydub
    audio_segment = AudioSegment.from_file(temp_file_name, format='wav')

    # Step 3: Clean up the temporary file
    os.remove(temp_file_name)

    return audio_segment
def combine_audio_arrays(array_list):
    # Step 1: Save numpy arrays as temporary audio files
    temp_files = []
    for i, audio_dict in enumerate(array_list):
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_files.append(temp_file.name)

        # Save numpy array as WAV file using soundfile
        sf.write(temp_file.name, audio_dict["audio"], samplerate=audio_dict["sampling_rate"])

    # Step 2: Reload and combine audio segments using pydub
    combined_audio = None
    for temp_file in temp_files:
        segment = AudioSegment.from_file(temp_file, format='wav')
        if combined_audio is None:
            combined_audio = segment
        else:
            combined_audio += segment

    # Step 3: Clean up temporary files
    for temp_file in temp_files:
        os.remove(temp_file)

    return combined_audio



# Function to convert audio files to mono 16 kHz if necessary
def convert_audio_to_mono_16k(file_path, output_path):
    audio = AudioSegment.from_file(file_path)
    needs_conversion = False

    if audio.channels > 1:
        audio = audio.set_channels(1)
        needs_conversion = True
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
        needs_conversion = True

    if needs_conversion:
        audio.export(output_path, format="wav")
    else:
        output_path = file_path  # No conversion needed

    return output_path

def adjust_rms(audio, target_rms):
    current_rms = audio.rms
    gain = 10 * math.log10(target_rms / current_rms)
    return audio.apply_gain(gain)

def adjust_spectral_properties(y, sr, target_centroid, target_bandwidth):
    current_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    current_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Calculate gain adjustments
    gain_low = target_centroid / current_centroid
    gain_high = target_bandwidth / current_bandwidth

    # Define the filter frequencies and corresponding gains
    eq_bands = [
        (50 / (sr / 2), 300 / (sr / 2), gain_low),    # Low frequencies
        (300 / (sr / 2), 3000 / (sr / 2), 1),         # Mid frequencies
        (3000 / (sr / 2), 15000 / (sr / 2), gain_high)  # High frequencies
    ]

    # Apply a bandpass filter for each band and combine them
    y_adjusted = np.zeros_like(y)
    for low_freq_normalized, high_freq_normalized, gain in eq_bands:
        sos = butter(N=2, Wn=[low_freq_normalized, high_freq_normalized], btype='band', output='sos')
        filtered = sosfilt(sos, y) * gain
        y_adjusted += filtered

    return y_adjusted

def adjust_tempo(y, sr, target_tempo):
    # Step 1: Determine the current tempo
    current_tempo, _ = librosa.beat.beat_track(sr=sr)

    # Step 2: Calculate the tempo ratio
    tempo_ratio = target_tempo / current_tempo

    # Step 3: Time-stretch the audio
    y_stretched = librosa.effects.time_stretch(y, rate=tempo_ratio)

    return y_stretched

def apply_vad(audio, sample_rate, aggressiveness=3, window_duration=0.03, vad_max_silence=0.7):
    window_length = int(window_duration * sample_rate)
    energy = np.square(audio)
    energy_smoothed = np.convolve(energy, np.ones(window_length) / window_length, mode='same')
    threshold = np.percentile(energy_smoothed, 100 - aggressiveness)
    above_threshold = np.array(energy_smoothed > threshold)  # Convert to a NumPy array
    vad_segments = []
    start_sample = None
    for i, is_speech in enumerate(above_threshold):
        if is_speech and start_sample is None:
            start_sample = i
        elif not is_speech and start_sample is not None:
            if i - start_sample >= vad_max_silence * sample_rate:
                vad_segments.append({'start': start_sample, 'stop': i, 'is_speech': True})
                start_sample = None
    if start_sample is not None:
        vad_segments.append({'start': start_sample, 'stop': len(audio), 'is_speech': True})
    return vad_segments

def prepare_for_vad(audio, audio_sample_rate, target_sr=16000):
    if np.max(np.abs(audio)) <= 1:
        print("Audio is already normalized")
    else:
        audio = audio.astype(np.float32) / 32768
    if audio_sample_rate != target_sr:
        audio = resampy.resample(audio.astype(float), audio_sample_rate, target_sr)
    audio = (audio * 32768).astype(np.int16)
    return audio, target_sr

def reduce_noise(audio, sr, noise_sample_length=10000, prop_decrease=1.0):
    noise_sample_length = min(noise_sample_length, len(audio))
    if len(audio) < noise_sample_length:
        padding = np.zeros(noise_sample_length - len(audio))
        audio = np.concatenate((audio, padding))
    try:
        audio_denoised = nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease)
    except Exception as e:
        print(f"Error during noise reduction: {e}")
        audio_denoised = audio
    return audio_denoised

def clean_audio(audio_data, sample_rate=44100):
    audio_denoised = reduce_noise(audio_data, sample_rate, noise_sample_length=10000, prop_decrease=1.0)
    audio_denoised_resampled, new_sample_rate = prepare_for_vad(audio_denoised, sample_rate)
    segments = apply_vad(audio_denoised_resampled, new_sample_rate)
    speech_audio = np.concatenate(
        [audio_denoised_resampled[seg['start']:seg['stop']] for seg in segments if seg['is_speech']]
    )
    if len(speech_audio) == 0:
        print("No segments containing speech were found.")
        return None
    return speech_audio, new_sample_rate


def prep_audio(file_path, output_path):
    print(output_path)
    # Constants for target adjustments
    target_rms = 0.075 * (2 ** 15)
    target_centroid = (2328.5 + 1861.1) / 2
    target_bandwidth = (1864.1 + 1432.0) / 2
    target_tempo = (152 + 172.3) / 2

    audio = AudioSegment.from_file(file_path)

    try:
        # Step 1: Clean audio
        audio_data, sample_rate = sf.read(file_path)
        cleaned_audio, new_sample_rate = clean_audio(audio_data, sample_rate)
        if cleaned_audio is not None:
            sf.write(output_path, cleaned_audio, new_sample_rate)
            audio = AudioSegment.from_file(output_path)

        # Step 2: Normalize RMS if needed
        current_rms = audio.rms
        if abs(current_rms - target_rms) / target_rms > 0.01:  # Allow small deviation
            audio = adjust_rms(audio, target_rms)
            audio.export(output_path, format="wav")
            audio = AudioSegment.from_file(output_path)

        # Step 3: Adjust Spectral Properties if needed
        y, sr = librosa.load(output_path, sr=None)
        current_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        current_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        if abs(current_centroid - target_centroid) / target_centroid > 0.01 or abs(
                current_bandwidth - target_bandwidth) / target_bandwidth > 0.01:  # Allow small deviation
            y = adjust_spectral_properties(y, sr, target_centroid, target_bandwidth)
            sf.write(output_path, y, sr)

        # Step 4: Adjust Tempo if needed
        y, sr = librosa.load(output_path, sr=None)
        current_tempo, _ = librosa.beat.beat_track(sr=sr)
        if abs(current_tempo - target_tempo) / target_tempo > 0.01:  # Allow small deviation
            y = adjust_tempo(y, sr, target_tempo)
            sf.write(output_path, y, sr)

        # Step 5: Convert audio to mono and 16kHz
        converted_path = convert_audio_to_mono_16k(output_path, output_path)
    except Exception as e:
        print(f"Error processing audio file: {e}")
        converted_path = output_path  # Return the original file path

    print(converted_path)
    return converted_path


# Function to process each audio file in the input folder to extract embeddings
def process_audio_files(input_folder):

    embeddings_list = []
    audio_files = []

    prepped_audio_folder = os.path.join("voice", "prepped_audio")
    os.makedirs(prepped_audio_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(prepped_audio_folder, filename)
            prepped_audio_path = prep_audio(file_path, output_path)
            embeddings = extract_sb_embeddings(prepped_audio_path)
            embeddings_list.append(embeddings)
            audio_files.append(filename)

    return np.array(embeddings_list), audio_files

# Function to train a Gaussian Mixture Model (GMM) on the extracted embeddings
def train_gmm(embeddings):
    n_samples = len(embeddings)
    if n_samples == 1:
        embeddings = np.concatenate([embeddings, embeddings])
    n_components = min(n_samples, 10)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(embeddings)
    return gmm

# Function to save the trained GMM model to a file
def save_gmm_model(gmm, output_path):
    joblib.dump(gmm, output_path)
    print(f"GMM model saved to {output_path}")

def clone_voice(input_folder, gmm_save_path):
    print("Processing audio files to extract embeddings...")
    embeddings, audio_files = process_audio_files(input_folder)
    print("Training GMM model...")
    gmm=train_gmm(embeddings)
    save_gmm_model(gmm, gmm_save_path)
    print(f"GMM model trained and saved to {gmm_save_path}")
    # Delete the processed audio files after embeddings extraction
    prepped_audio_folder = "voice/prepped_audio"
    for file in os.listdir(prepped_audio_folder):
        file_path = os.path.join(prepped_audio_folder, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    os.rmdir(prepped_audio_folder)
# Function to extract an X-vector from a trained Gaussian Mixture Model (GMM)
def extract_xvector_from_gmm(trained_gmm):
    x_vector = trained_gmm.means_[0]
    x_vector /= np.linalg.norm(x_vector)
    return x_vector

# Function to blend two GMMs and train a new GMM on the combined embeddings
def gmm_blender(gmm1, gmm2, ratio):
    # Extract X-vectors from both GMMs
    xvector1 = extract_xvector_from_gmm(gmm1)
    xvector2 = extract_xvector_from_gmm(gmm2)

    # Duplicate the first GMM's X-vectors based on the ratio
    if ratio > 1:
        xvector1 = np.tile(xvector1, (int(ratio), 1))

    # Combine the embeddings
    combined_embeddings = np.vstack([xvector1, xvector2])

    # Train a new GMM on the combined embeddings
    blended_gmm = train_gmm(combined_embeddings)

    return blended_gmm

# Load or download the transformers model
def load_or_download_transformers_model(model_class, model_name, local_path):
    if os.path.exists(local_path):
        print(f"Loading {model_name} from {local_path}...")
        model = model_class.from_pretrained(local_path)
    else:
        print(f"Downloading and saving {model_name} to {local_path}...")
        model = model_class.from_pretrained(model_name)
        model.save_pretrained(local_path)
    return model

# Function to generate speech from text using a given speaker embedding
def generate_speech(text, speaker_embedding):
    print("Loading TTS models...")
    model_dir = "models"
    tts_model_name = "microsoft/speecht5_tts"
    tts_model_path = os.path.join(model_dir, "microsoft_speecht5_tts")
    tts_model = load_or_download_transformers_model(SpeechT5ForTextToSpeech, tts_model_name, tts_model_path)
    print("TTS models loaded...")
    input_text = number_to_text(text)
    # Ensure speaker embedding is of type Float
    speaker_embedding = speaker_embedding.float()

    # Measure the text to get the word count
    text_metrics = measure_text(input_text)
    word_count = text_metrics.word_count

    # Check if the text needs to be split
    if word_count <= 40:
        # Prepare the text-to-speech synthesizer
        synthesiser = pipeline("text-to-speech", tts_model_name)
        # Generate speech for the entire text
        speech_data = synthesiser(input_text, forward_params={"speaker_embeddings": speaker_embedding})
        speech = dub_array(speech_data)
        return speech
    else:
        # Split the text into chunks of word_limit
        text_chunks = split_text(input_text, word_limit=40)

        # Prepare the text-to-speech synthesizer
        synthesiser = pipeline("text-to-speech", tts_model_name)

        # Generate speech for each chunk and collect the audio data
        audio_data_list = []
        for chunk in text_chunks:
            speech = synthesiser(chunk, forward_params={"speaker_embeddings": speaker_embedding})
            audio_data_list.append(speech)

        # Combine the audio data into one continuous audio segment
        combined_speech = combine_audio_arrays(audio_data_list)

        return combined_speech

# Function to process audio files and generate speech using extracted x-vectors
def speech_with_gmm(text, gmm):
    # Extract xvector from GMM
    xvector = extract_xvector_from_gmm(gmm)
    speaker_embedding = torch.tensor(xvector).unsqueeze(0).float()  # Ensure it's of type Float

    # Generate speech
    speech = generate_speech(text, speaker_embedding)

    return speech



def play_audio_file(path):
    wave_obj = sa.WaveObject.from_wave_file(path)
    play_obj = wave_obj.play()
    play_obj.wait_done()



def tts_with_gmm(text, gmm):
    gmm_base = joblib.load(gmm)
    speech = speech_with_gmm(text, gmm_base)

    if isinstance(speech, AudioSegment):

        def play():
            play_audio(speech)

        threading.Thread(target=play).start()
    else:
        print("Error: The TTS output is not an AudioSegment object.")
        raise TypeError("Invalid audio format")

def tts(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed percent (can go over 100)
    engine.setProperty('volume', 0.9)  # Volume 0-1

    # Convert text to speech
    engine.say(text)

    # Wait for the speech to finish
    engine.runAndWait()

# Function to load the pre-trained SpeechBrain model for speaker recognition
def load_sb_model():
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb"
    )
    return classifier


# Function to extract embeddings using the SpeechBrain model
def extract_sb_embeddings(audio_path):
    xvmodel = load_sb_model()
    signal, fs = torchaudio.load(audio_path)
    sb_embeddings = xvmodel.encode_batch(signal)
    return sb_embeddings.squeeze().cpu().numpy()


# Function to load the pre-trained pyannote model
def load_pn_model():
    pnmodel = Inference("pyannote/wespeaker-voxceleb-resnet34-LM")
    return pnmodel


# Function to extract embeddings using the pyannote model
def extract_pn_embeddings(audio_path):
    extractor = load_pn_model()
    pn_embeddings = extractor(audio_path)
    return pn_embeddings.data


# Main function to extract all audio features
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Extract all features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).flatten()
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max).flatten()
    lpc_coeffs = librosa.lpc(y, order=13).flatten()
    plp = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, lifter=22).flatten()
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).flatten()
    roots = np.roots(lpc_coeffs)
    roots = roots[np.imag(roots) >= 0]
    angles = np.angle(roots)
    freqs = angles * (sr / (2 * np.pi))
    formant_freqs = np.sort(freqs).flatten()
    zcr = librosa.feature.zero_crossing_rate(y=y).flatten()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).flatten()
    spectral_flatness = librosa.feature.spectral_flatness(y=y).flatten()
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).flatten()
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    f0 = np.array([pitch[np.argmax(mag)] for pitch, mag in zip(pitches.T, magnitudes.T) if np.max(mag) > 0]).flatten()
    frame_size = 2048
    short_time_energy = np.array(
        [np.sum(np.abs(y[i:i + frame_size] ** 2)) for i in range(0, len(y), frame_size)]).flatten()
    sb_embeddings = extract_sb_embeddings(file_path).flatten()
    pn_embeddings = extract_pn_embeddings(file_path).flatten()

    # Combine all features into one array
    all_features = np.concatenate([
        mfccs, log_mel_spectrogram, lpc_coeffs, plp, chroma, formant_freqs,
        zcr, spectral_centroid, spectral_bandwidth, spectral_flatness,
        spectral_contrast, f0, short_time_energy, sb_embeddings, pn_embeddings
    ])

    return all_features


# Function to train and save a single GMM for all combined features
def train_and_save_gmm(features, output_path, name, n_components=50, max_iter=500):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gmm = GaussianMixture(n_components=n_components, max_iter=max_iter)
    gmm.fit(features.reshape(-1, 1))

    model_path = os.path.join(output_path, f"{name}.pkl")
    with open(model_path, 'wb') as model_file:
        pickle.dump(gmm, model_file)

def extract_and_train_gmms(audio_file, output_path, name, n_components=50, max_iter=500):
    """
    Extract features from an audio file and train GMMs on them.

    Args:
    - audio_file (str): Path to the audio file.
    - output_path (str): Path to save the trained GMMs.
    - n_components (int): Number of components for GMMs.
    - max_iter (int): Maximum number of iterations for GMM training.
    """
    # Extract features from the audio file
    features = extract_audio_features(audio_file)

    # Train GMMs and save them
    train_and_save_gmm(features, output_path, name, n_components=n_components, max_iter=max_iter)

# Function to match an audio file against GMMs in a directory
def match_audio_with_gmms(audio_file, gmm_directory, threshold=None):
    extracted_features = extract_audio_features(audio_file)

    best_score = float('-inf')
    best_match = "Unknown Speaker"

    for gmm_file in os.listdir(gmm_directory):
        if gmm_file.endswith('.pkl'):
            model_path = os.path.join(gmm_directory, gmm_file)

            with open(model_path, 'rb') as model_file:
                gmm = pickle.load(model_file)

            score = gmm.score(extracted_features.reshape(-1, 1))
            print(gmm_file)
            print(score)
            if score > best_score:
                best_score = score
                best_match = gmm_file.replace('.pkl', '')

    if threshold is not None and best_score < threshold:
        return "Unknown Speaker"

    return best_match



def load_audio(file_path):
    return AudioSegment.from_file(file_path)

def diarize_audio(file_path):
    # Initialize the pre-trained pipeline for speaker diarization including overlapped speech
    pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
# Add your Huggingface token here
  use_auth_token='hf_')

    # Apply the pipeline to an audio file
    diarization = pipeline(file_path)

    return diarization


def segment_and_save_audio(diarization, audio, output_dir):
    # Empty the output directory before starting
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    segments = []
    current_speaker = None
    current_start = None
    current_end = None
    current_tracks = []
    current_audio_segments = AudioSegment.empty()  # Initialize as an empty AudioSegment

    for segment, track, label in diarization.itertracks(yield_label=True):
        start = segment.start
        end = segment.end
        speaker = label

        if speaker == current_speaker:
            # Extend the current segment and append new audio to the current segment
            current_end = end
            current_tracks.append(track)
            current_audio_segments += audio[start * 1000:end * 1000]  # Combine the audio segments
        else:
            # Save the previous segment if it exists and is at least 1 second long
            if current_speaker is not None and (current_end - current_start) >= 1:
                file_name = f"speaker_{current_speaker}_start_{current_start:.2f}_end_{current_end:.2f}.wav"
                output_path = os.path.join(output_dir, file_name)
                current_audio_segments.export(output_path, format="wav")

                segments.append({
                    "speaker": current_speaker,
                    "start": str(timedelta(seconds=current_start)),
                    "end": str(timedelta(seconds=current_end)),
                    "tracks": current_tracks,
                    "file": output_path
                })

            # Start a new segment
            current_speaker = speaker
            current_start = start
            current_end = end
            current_tracks = [track]
            current_audio_segments = audio[start * 1000:end * 1000]  # Initialize with the current segment

    # Save the last segment if it is at least 1 second long
    if current_speaker is not None and (current_end - current_start) >= 1:
        file_name = f"speaker_{current_speaker}_start_{current_start:.2f}_end_{current_end:.2f}.wav"
        output_path = os.path.join(output_dir, file_name)
        current_audio_segments.export(output_path, format="wav")

        segments.append({
            "speaker": current_speaker,
            "start": str(timedelta(seconds=current_start)),
            "end": str(timedelta(seconds=current_end)),
            "tracks": current_tracks,
            "file": output_path
        })

    return segments

def separate_speakers(file_path, output_dir="segmented_audio"):
    # Load the audio file
    audio = load_audio(file_path)
    # Diarize the audio file
    diarization = diarize_audio(audio)

    # Segment the audio and save the clips
    segments = segment_and_save_audio(diarization, audio, output_dir)

    return segments

def find_speakers(audio_file, gmm_dir, threshold=-5.2, unknown_speakers=False):
    # Separate the audio into segments based on speakers
    speaker_segments = separate_speakers(audio_file)
    speaker_name_map = {}
    failed_segments = []

    # Iterate over each segment and match it to a speaker using the GMMs
    for segment_info in speaker_segments:
        segment_path = segment_info['file']
        speaker_label = segment_info['speaker']

        try:
            # Identify the speaker using the GMMs
            speaker_name = match_audio_with_gmms(segment_path, gmm_dir, threshold)
            # Update the speaker name map
            speaker_name_map[speaker_label] = speaker_name
            # Update the segment info with the identified speaker name
            segment_info['speaker'] = speaker_name
        except Exception as e:
            print(f"Error processing segment {segment_path}: {e}")
            # Add to the failed list if identification fails
            failed_segments.append(segment_info)

    # Handle failed segments
    for failed_segment in failed_segments:
        speaker_label = failed_segment['speaker']

        # Use the identified name for the same speaker label, if available
        if speaker_label in speaker_name_map:
            failed_segment['speaker'] = speaker_name_map[speaker_label]
        else:
            if unknown_speakers is True:
                failed_segment['speaker'] = "Unknown Speaker"
            else:
                speaker_segments.remove(failed_segment)  # Remove segment if unknown_speakers=False

    # If no recognized speakers are found, return an empty list
    if not speaker_segments:
        print("No recognized speaker")
        return []

    return speaker_segments

def transcribe_audio(audio_data):
    """
    Method for transcribing audio using Google Speech-to-Text.

    Args:
        audio_data (sr.AudioData): Captured audio data.

    Returns:
        str: Transcribed text.
    """
    recognizer = sr.Recognizer()

    try:
        print("Transcribing...")
        text = recognizer.recognize_google(audio_data)
        print("Returning text...")
        return text.strip()
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return ""

def diarized_transcription(audio_file, gmm_dir, unknown_speakers=False):
    """
    Transcribe the audio file with speaker diarization.

    Args:
        audio_file (str): Path to the audio file.
        gmm_dir (str): Directory containing GMMs for speaker identification.
        unknown_speakers (bool): Whether to keep segments with unknown speakers.

    Returns:
        str: Diarized text transcription or a message if no recognized speaker.
    """
    # Separate the audio into segments and identify speakers
    speaker_segments = find_speakers(audio_file, gmm_dir, unknown_speakers=unknown_speakers)

    if not speaker_segments:
        return "No recognized speaker"

    # Sort the segments based on their start times
    speaker_segments.sort(key=lambda x: x['start'])

    transcribed_text = []

    # Transcribe each segment
    for segment_info in speaker_segments:
        speaker_name = segment_info['speaker']

        # Skip transcription if the speaker is "Unknown Speaker" and unknown_speakers is False
        if not unknown_speakers and speaker_name == "Unknown Speaker":
            continue

        segment_path = segment_info['file']

        # Load the audio segment for transcription
        audio_segment = AudioSegment.from_file(segment_path, format="wav")

        # Check if the audio is empty before attempting to transcribe it
        if len(audio_segment) == 0:
            print(f"Skipping empty audio segment: {segment_path}")
            continue

        audio_data = sr.AudioData(audio_segment.raw_data, audio_segment.frame_rate, audio_segment.sample_width)

        # Transcribe the segment
        text_transcription = transcribe_audio(audio_data)
        if text_transcription:
            transcribed_text.append(f"{speaker_name} said \"{text_transcription}\" At:{formatted_time} ")

    # Concatenate the transcriptions into a single text string
    diarized_text = ' '.join(transcribed_text)

    return diarized_text if transcribed_text else "No recognized speaker"

async def transcribe_audio_async(audio_data):
    """
    Asynchronous method for transcribing audio using Google Speech-to-Text.

    Args:
        audio_data (sr.AudioData): Captured audio data.

    Yields:
        str: Transcribed text.
    """
    recognizer = sr.Recognizer()

    try:
        print("Transcribing...")
        text = recognizer.recognize_google(audio_data)
        print("Yielding text...")
        yield text.strip()
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def record_and_transcribe(wake_word=None, timer=None, word_count=None, gmm_dir=None, unknown_speakers=False, output="text"):
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    dialog_string = ""
    speech_audio = []
    start_time = time.time()

    while True:
        # Use the microphone as the source
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source)
            print("Recording... Speak now.")

            # Record the audio
            audio_data = recognizer.listen(source)
            print("Recording complete.")

        if gmm_dir is not None:
            # Save the audio to a temporary file

            temp_filename = "temp_audio.wav"
            with open(temp_filename, "wb") as temp_file:
                temp_file.write(audio_data.get_wav_data())

            # Diarized transcription
            transcription = diarized_transcription(temp_filename, gmm_dir, unknown_speakers)

            # Skip if transcription returned "No recognized speaker"
            if transcription == "No recognized speaker":
                continue
        else:
            # Pass the audio data to transcribe_audio
            transcription = transcribe_audio(audio_data)

        current_time = time.time()
        time_passed = current_time - start_time

        if timer is not None and time_passed > timer:
            dialog_string = transcription
            speech_audio = [audio_data] if output != "text" else []
            start_time = current_time  # Reset the timer
        else:
            if word_count is None:
                dialog_string += " " + transcription
            else:
                dialog_string = transcription

            dialog_string = dialog_string.strip()

            if word_count is not None:
                metrics = measure_text(dialog_string)
                if metrics.word_count > word_count:
                    dialog_string = transcription

            if output != "text":
                speech_audio.append(audio_data)

        if wake_word:
            print(dialog_string)
            if check_text(dialog_string, wake_word):
                if output == "audio":
                    return combine_audio_data(speech_audio)
                elif output == "text":
                    return dialog_string
                else:
                    return combine_audio_data(speech_audio), dialog_string
                # Reset after returning due to wake_word
                dialog_string = ""
                speech_audio = []
        else:
            if output == "audio":
                return combine_audio_data(speech_audio)
            elif output == "text":
                return dialog_string
            else:
                return combine_audio_data(speech_audio), dialog_string
async def async_record_and_transcribe(wake_word=None, timer=None, word_count=None, gmm_dir=None, unknown_speakers=False, output="text"):
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    dialog_string = ""
    speech_audio = []
    start_time = time.time()

    while True:
        # Use the microphone as the source
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source)
            print("Recording... Speak now.")

            # Record the audio
            audio_data = recognizer.listen(source)
            print("Recording complete.")

        if gmm_dir is not None:
            # Save the audio to a temporary file
            temp_filename = "temp_audio.wav"
            with open(temp_filename, "wb") as temp_file:
                temp_file.write(audio_data.get_wav_data())

            # Diarized transcription
            transcription = diarized_transcription(temp_filename, gmm_dir, unknown_speakers)

            # Skip if transcription returned "No recognized speaker"
            if transcription == "No recognized speaker":
                continue
        else:
            # Pass the audio data to transcribe_audio_async
            transcription = transcribe_audio_async(audio_data)

        current_time = time.time()
        time_passed = current_time - start_time

        if timer is not None and time_passed > timer:
            dialog_string = transcription
            speech_audio = [audio_data] if output != "text" else []
            start_time = current_time  # Reset the timer
        else:
            if word_count is None:
                dialog_string += " " + transcription
            else:
                dialog_string = transcription

            dialog_string = dialog_string.strip()

            if word_count is not None:
                metrics = measure_text(dialog_string)
                if metrics.word_count > word_count:
                    dialog_string = transcription

            if output != "text":
                speech_audio.append(audio_data)

        if wake_word:
            if check_text(dialog_string, wake_word):
                if output == "audio":
                    yield combine_audio_data(speech_audio)
                elif output == "text":
                    yield dialog_string
                else:
                    yield combine_audio_data(speech_audio), dialog_string
                # Reset after yielding due to wake_word
                dialog_string = ""
                speech_audio = []
        else:
            if output == "audio":
                yield combine_audio_data(speech_audio)
            elif output == "text":
                yield dialog_string
            else:
                yield combine_audio_data(speech_audio), dialog_string



sentences = {
    1: "Seven slippery snakes slither silently through the tall grass, seeking sunshine",
    2: "The brilliant blue bird perched on the branch, singing a sweet serenade at sunrise",
    3: "Glistening glaciers glowed under the golden sun, melting into a cascading waterfall",
    4: "The old oak tree stood tall, its leaves rustling in the gentle evening Breeze",
    5: "Whispering winds wound their way through the winding, wooded Path",
    6: "The mighty mountain lion leaped gracefully over the rocky ledge, landing softly below",
    7: "A fleet of fast, fiery foxes raced across the frozen field, leaving tracks in the snow",
    8: "The clever cat cautiously crept closer to the curious crow, watching its every move",
    9: "Bright beams of sunlight streamed through the stained glass, casting colorful shadows",
    10: "The quick quail quietly quivered, hiding from the hungry hawk overhead",
    11: "The relentless rain rattled the rooftop, resonating in the silent night",
    12: "Soft, silken sands shifted beneath the feet of the strolling couple on the shore",
    13: "The thunderous roar of the waterfall echoed through the cavern, shaking the ground",
    14: "The luminous lantern lit the way through the dark, damp, and dreary dungeon",
    15: "The busy bumblebee buzzed busily between blooming, bright, and beautiful blossoms",
    16: "Crimson clouds cascaded across the horizon as the sun dipped below the distant hills",
    17: "The tall tower teetered precariously in the turbulent wind, creaking ominously",
    18: "A flock of feathery flamingos fluttered their wings, rising gracefully into the air",
    19: "The gentle giant giraffe grazed on the green, leafy branches of the towering trees",
    20: "The eager eagle soared high above the mountains, scanning the land below for prey",
    21: "The ancient archway stood as a testament to time, weathered but unyielding",
    22: "Golden grains of sand sparkled in the sun, stretching endlessly along the seashore",
    23: "The wise old owl hooted softly from its perch, hidden deep within the forest",
    24: "A symphony of crickets played in the background as the moon rose in the clear night sky",
    25: "The fierce falcon dived rapidly, its sharp eyes locked on its unsuspecting target",
    26: "The gurgling brook wound its way through the verdant valley, reflecting the azure sky",
    27: "The shimmering stars dotted the midnight sky, twinkling like diamonds",
    28: "The jagged peaks of the mountain range cut into the sky, sharp and foreboding",
    29: "The playful puppy pounced on the pile of autumn leaves, scattering them everywhere",
    30: "The crisp crunch of fallen leaves underfoot echoed in the quiet, cool air of the forest",
    31: "The vast ocean stretched out before them, its waves crashing against the rocky cliffs",
    32: "The bold bear bounded through the thick underbrush, its heavy paws thudding on the ground",
    33: "The flickering flames danced in the fireplace, casting a warm glow throughout the room",
    34: "The delicate daffodil swayed gently in the breeze, its yellow petals catching the light",
    35: "The distant drumbeat reverberated through the night, a call to the tribal dance",
    36: "The mysterious mist enveloped the landscape, obscuring the path ahead",
    37: "The swirling snowflakes fell silently, covering the world in a blanket of white",
    38: "The mischievous monkey swung from vine to vine, chattering loudly as it moved",
    39: "The jaguar’s amber eyes glinted in the darkness as it stalked its prey silently",
    40: "The ancient ruins crumbled under the weight of time, yet still stood in defiance",
    41: "The sleek submarine sliced through the depths of the ocean, silent and unseen",
    42: "The glowing embers of the campfire pulsed rhythmically, holding the night's chill at bay",
    43: "The gentle hum of the hive filled the air as the bees busied themselves with their work",
    44: "The old, rusty gate creaked open, revealing a hidden garden bursting with color",
    45: "The sly fox slipped silently into the henhouse, its eyes gleaming with cunning",
    46: "The rolling thunder grew louder, signaling the approach of a fierce storm",
    47: "The reflective surface of the lake mirrored the sky, creating a perfect illusion",
    48: "The vast savannah stretched out before them, dotted with acacia trees and grazing animals",
    49: "The rhythmic sound of the waves lulled them into a peaceful slumber on the beach",
    50: "The mighty oak tree's roots dug deep into the earth, anchoring it firmly in place",
    51: "The colorful chameleon shifted its hues, blending seamlessly into its surroundings",
    52: "The distant lighthouse beacon swept across the ocean, guiding ships safely to shore",
    53: "The rhythmic clatter of the train on the tracks was a comforting sound in the night",
    54: "The curious kangaroo hopped closer, its large eyes filled with innocent wonder",
    55: "The solitary wolf howled mournfully at the full moon, its call echoing through the forest",
    56: "The glittering city skyline reflected in the calm waters of the river below",
    57: "The powerful stallion galloped across the open plain, its mane flying in the wind",
    58: "The ancient manuscript was filled with faded, handwritten notes and mysterious symbols",
    59: "The sleek dolphin leaped gracefully from the water, performing an aerial dance",
    60: "The peaceful meadow was filled with the scent of wildflowers and the sound of buzzing bees",
    61: "The quick brown fox jumps over the lazy dog, while the curious zebra quietly observes from afar, wondering if the vibrant parrot will sing a song at dusk"
}
def display_text(text):
    global root
    # Create the main window
    root = tkinter.Tk()
    root.title("Text Display")

    # Create a Text widget
    text_box = tkinter.Text(root, height=10, width=50)
    text_box.pack()

    # Insert the text into the Text widget
    text_box.insert(tkinter.END, text)

    # Run the application
    root.mainloop()

def close_text_box():
    global root
    if root is not None:
        root.destroy()
        root = None

def remember_user(username, sentence_num=1):
    speaker_audio = []
    max_segments = 12
    current_sentence_num = sentence_num

    while len(speaker_audio) < max_segments:
        sentence = sentences.get(current_sentence_num, None)
        if sentence is None:
            break

        # Display the sentence to the user
        text = f"Please read the following sentence out loud: {sentence}"
        display_text(text)  # Replace with the actual display function if needed

        # Use ASR to capture audio
        wake_word = sentence.split()[-1]
        print(wake_word)
        audio = record_and_transcribe(wake_word=wake_word, output="audio")
        speaker_audio.append(audio)
        close_text_box()
        # Move to the next sentence
        current_sentence_num += 1

        # Ask if the user wants to continue every 10 sentences
        if current_sentence_num % 10 == 0:
            continuation_prompt = "Would you like to continue?"
            print(continuation_prompt)  # Replace with the actual display function if needed
            response_text = record_and_transcribe(wake_word=None, output="text")
            if check_text(response_text, "Yes"):
                continue  # Continue the loop
            elif check_text(response_text, "No"):
                break  # Exit early
            else:
                print("Unrecognized response. Exiting.")
                break

    # Combine all collected audio segments
    combined_audio = combine_audio_data(speaker_audio)

    # Save the combined audio
    temp_filename = "combined_audio.wav"
    combined_audio.export(temp_filename, format="wav")

    # Train and save GMMs
    gmm_dir = f"data/user_data/{username}"
    os.makedirs(gmm_dir, exist_ok=True)
    extract_and_train_gmms(temp_filename, gmm_dir, name=username)

    return current_sentence_num

'''Command Utilities'''

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def record_event(event_type, event_details, command_actions, start_time):
    elapsed_time = time.time() - start_time
    command_actions.append({
        "time": elapsed_time,
        "event": event_type,
        **event_details
    })

def is_desktop_active():
    # Get the handle of the desktop window
    desktop_hwnd = win32gui.GetDesktopWindow()
    # Get the handle of the currently active window
    active_hwnd = win32gui.GetForegroundWindow()
    return active_hwnd == desktop_hwnd

def is_program_active(program_name):
    # Get the title of the currently active window
    active_window = gw.getActiveWindow()
    if active_window:
        active_title = active_window.title
        print(f"Active Window Title: {active_title}")
        # List all running programs
        for process in psutil.process_iter(['pid', 'name']):
            if process.info['name'].lower() == program_name.lower():
                print(f"{program_name} is running.")
                if program_name.lower() in active_title.lower():
                    print(f"{program_name} is the active window.")
                    return True
                else:
                    print(f"{program_name} is not the active window.")
                    return False
        print(f"{program_name} is not running.")
    else:
        print("No active window found.")
    return False

def record_command(command_name):
    """Start recording the input sequence for a custom command."""
    global is_recording, command_actions
    command_folder = os.path.join(CUSTOM_COMMANDS_FOLDER, command_name)
    create_folder(command_folder)
    create_folder(COMMANDS_JSON_FOLDER)

    command_actions = []
    start_time = time.time()
    is_recording = True

    # Check if the desktop is active
    if is_desktop_active():
        print("Desktop is active. The first recorded event will be directed to the desktop.")
    else:
        active_window = gw.getActiveWindow()
        if active_window:
            active_title = active_window.title
            print(f"Active Window Title: {active_title}. The first recorded event will be directed to this program.")
            record_event("window_switch", {"title": active_title}, command_actions, start_time)

    def on_key_event(event):
        print(f"Key Event: {event.name}, Type: {event.event_type}")  # Debug line
        key = event.name
        if event.event_type == "down":
            if key in ["volume_up", "volume_down", "volume_mute", "play_pause", "next_track", "prev_track"]:
                record_event("special_key_down", {"key": key}, command_actions, start_time)
            else:
                record_event("key_down", {"key": key}, command_actions, start_time)
        elif event.event_type == "up":
            if key in ["volume_up", "volume_down", "volume_mute", "play_pause", "next_track", "prev_track"]:
                record_event("special_key_up", {"key": key}, command_actions, start_time)
            else:
                record_event("key_up", {"key": key}, command_actions, start_time)

    def on_mouse_click(event):
        if isinstance(event, mouse.ButtonEvent):
            event_type = "mouse_down" if event.event_type == "down" else "mouse_up"
            record_event(event_type, {"button": event.button}, command_actions, start_time)

    def on_mouse_move(event):
        if isinstance(event, mouse.MoveEvent):
            record_event("mouse_move", {"x": event.x, "y": event.y}, command_actions, start_time)

    def on_mouse_scroll(event):
        if isinstance(event, mouse.WheelEvent):
            print(f"Scroll Event: Delta: {event.delta}")  # Debug line
            record_event("mouse_scroll", {"delta": int(event.delta)}, command_actions, start_time)

    # Hooking into keyboard and mouse events
    keyboard.hook(on_key_event)
    mouse.hook(on_mouse_click)
    mouse.hook(on_mouse_move)
    mouse.hook(on_mouse_scroll)

    print("Recording... Call stop_recording() to stop.")

def scroll_mouse(delta):
    # Positive delta scrolls up, negative delta scrolls down
    ctypes.windll.user32.mouse_event(0x0800, 0, 0, int(delta * 120), 0)

def stop_recording(command_name):
    """Stop recording and save the recorded input."""
    global is_recording
    if not is_recording:
        print("No recording in progress.")
        return

    is_recording = False
    keyboard.unhook_all()
    mouse.unhook_all()

    # Save actions to JSON file
    json_path = os.path.join(COMMANDS_JSON_FOLDER, f"{command_name}.json")
    with open(json_path, 'w') as json_file, open(f"{command_name}_actions.txt", 'w') as txt_file:
        json.dump(command_actions, json_file, indent=4)
        txt_file.write(json.dumps(command_actions, indent=4))  # Save a plain text version

    print(f"Command '{command_name}' saved successfully.")

def fast_move(x, y):
    """Move the mouse cursor instantly to the specified coordinates."""
    ctypes.windll.user32.SetCursorPos(x, y)

def press_special_key(key):
    print(key)
    special_keys = {
        "volume up": 0xAF,  # Volume Up
        "volume down": 0xAE,  # Volume Down
        "volume mute": 0xAD,  # Volume Mute
        "play/pause media": 0xB3,  # Play/Pause
        "next track": 0xB0,  # Next Track
        "prev track": 0xB1,  # Previous Track
    }
    if key in special_keys:
        print(key)
        vk_code = special_keys[key]
        # Press the key
        ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)
        # Release the key
        ctypes.windll.user32.keybd_event(vk_code, 0, 2, 0)
        # Add a small delay to ensure the key event is processed
        time.sleep(0.1)

def minimize_all_windows():
    ctypes.windll.user32.keybd_event(0x5B, 0, 0, 0)  # Press the Windows key
    ctypes.windll.user32.keybd_event(0x4D, 0, 0, 0)  # Press 'M'
    ctypes.windll.user32.keybd_event(0x4D, 0, 2, 0)  # Release 'M'
    ctypes.windll.user32.keybd_event(0x5B, 0, 2, 0)  # Release the Windows key
    time.sleep(1)  # Give some time for the desktop to come into focus




def update_command_map(command_phrase):
    # Path to MAP.json
    map_file_path = os.path.join(CUSTOM_COMMANDS_FOLDER, "MAP.json")

    # Check if MAP.json exists
    if os.path.exists(map_file_path):
        # Load the existing map
        with open(map_file_path, 'r') as map_file:
            command_map = json.load(map_file)
    else:
        # If it doesn't exist, create an empty map
        command_map = {}

    # Determine the number of existing commands
    num_commands = len(command_map)

    # Generate a new command name
    new_command_name = f"command{num_commands + 1}"

    # Update the map with the new command name and phrase
    command_map[new_command_name] = command_phrase

    # Save the updated map back to MAP.json
    with open(map_file_path, 'w') as map_file:
        json.dump(command_map, map_file, indent=4)

    # Return the new command name
    return new_command_name

def check_custom_commands(text, matched_commands=None):
    if matched_commands is None:
        matched_commands = []
    # Path to MAP.json
    map_file_path = os.path.join(CUSTOM_COMMANDS_FOLDER, "MAP.json")

    # Check if MAP.json exists
    if not os.path.exists(map_file_path):
        print("MAP.json not found.")
        return False

    # Load the command map
    with open(map_file_path, 'r') as map_file:
        command_map = json.load(map_file)


    # Check each phrase in the command map
    for command_name, command_phrase in command_map.items():
        if command_phrase in text:
            matched_commands.append(command_name)

    # Return the list of found commands or False if no matches
    if matched_commands:
        return matched_commands
    else:
        return False

def check_folder(base_dir, text):
    matched_paths = []
    text_lower = text.lower()

    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        entry_name = entry.lower()
        entry_name_without_extension = os.path.splitext(entry_name)[0].lower()
        # If the entry is a file and matches the text, add it to matched_paths
        if entry_name_without_extension in text_lower:
            print(f"Matched file: {entry_path}")
            matched_paths.append(entry_path)

        # If the entry is a folder and its name matches, search inside it
        elif os.path.isdir(entry_path):
            if entry_name in text_lower:
                print(f"Checking folder: {entry_path}")
                # Recursively check the contents of the matching folder
                deeper_matches = check_folder(entry_path, text)
                matched_paths.extend(deeper_matches)
            else:
                # Continue checking subfolders even if the folder name doesn't match
                matched_paths.extend(check_folder(entry_path, text))

    return matched_paths

def check_basic_commands( text, matched_commands=None):
    if matched_commands is None:
        matched_commands = []
    try:
        from Mods.IMPORTS import COMMANDS

        for command_dict in COMMANDS:
            commands = command_dict['commands']
            action = command_dict['action']
            print(f'Checking against commands: {commands}')
            for cmd in commands:
                if cmd in text:
                    print(f'Matched command: {cmd}, triggering action: {action.__name__}')
                    matched_commands.append((action, []))
        return bool(matched_commands)
    except ImportError:
        print(f"Error Importing Commands")
        return bool(matched_commands)

def check_shortcuts(text, shortcut_dir, matched_commands=None):
    os.makedirs(shortcut_dir, exist_ok=True)
    matched_entries = []
    if matched_commands is None:
        matched_commands = []

    for entry in os.listdir(shortcut_dir):
        entry_path = os.path.join(shortcut_dir, entry)
        entry_name_without_extension = os.path.splitext(entry)[0].lower()

        if os.path.isdir(entry_path):
            print(f"Found folder: {entry_path}, checking its contents...")
            matched_paths = check_folder(entry_path, text)
            if matched_paths:
                print(f"Matched paths inside folder: {matched_paths}")
                matched_commands.extend([(os.startfile, [path]) for path in matched_paths])
        elif entry_name_without_extension in text.lower():
            print(f"Matched file in root: {entry_path}")
            matched_entries.append(entry_path)

    if matched_entries:
        matched_commands.extend([(os.startfile, [entry]) for entry in matched_entries])
        return True

    return bool(matched_commands)

def execute_command(text, commands, command_executed_tags, argument_dictionary=None):
    try:
        from Mods.IMPORTS import MAP
        for action, _ in commands:
            print(f"Executing action: {action.__name__}")

            # Get the arguments from MAP
            map_args = MAP.get(action, ())

            # Resolve dynamic arguments if any
            if argument_dictionary:
                resolved_args = []
                for arg in map_args:
                    if isinstance(arg, str) and arg.startswith("{") and arg.endswith("}"):
                        key = arg.strip("{}")
                        resolved_args.append(argument_dictionary.get(key, ""))
                    else:
                        resolved_args.append(arg)
            else:
                resolved_args = map_args

            # Execute the action with the resolved arguments
            action(*resolved_args)
            command_executed_tags.append(f"Executed action: {action.__name__}")

        dialogue = f"The user said '{text}'. In your own words, tell them you will perform these actions: {', '.join(command_executed_tags)}"
        response = Response(dialogue)
        tts(response)
    except ImportError:
        print(f"Error Importing MAP")

def execute_shortcut(text, actions, command_executed_tags):
    for action, args in actions:
        try:
            print(f"Attempting to execute: {args[0]}")
            os.startfile(args[0])
            print(f"Executed or opened: {args[0]}")
            command_executed_tags.append(f"Opening/Executing: {args[0]}")
        except Exception as e:
            print(f"Failed to execute {args[0]}: {e}")

    dialogue = f"The user said '{text}'. In your own words, tell them you will perform these actions: {', '.join(command_executed_tags)}"
    response = Response(dialogue)
    tts(response)

def execute_custom_command(text, command_name):
    json_path = os.path.join(COMMANDS_JSON_FOLDER, f"{command_name}.json")
    if not os.path.exists(json_path):
        print(f"Command '{command_name}' not found.")
        return

    with open(json_path, 'r') as json_file:
        command_actions = json.load(json_file)

    # Check if a window switch was recorded
    window_switch_action = next((action for action in command_actions if action.get("event") == "window_switch"), None)

    if window_switch_action:
        # Get the window title from the recorded action
        window_title = window_switch_action["title"]

        # Check if the title is empty or indicates 'Program Manager'
        if not window_title or window_title == "Program Manager":
            print(f"Assuming desktop is the target since the title is empty or 'Program Manager'.")
            minimize_all_windows()
        else:
            print(f"Switching to window with title: {window_title}")
            try:
                # Use pywinauto to bring the window to the front
                app = application.Application().connect(title=window_title)
                window = app.window(title=window_title)
                window.set_focus()
                time.sleep(1)  # Give some time for the window to come into focus
            except Exception as e:
                print(f"Failed to switch to window '{window_title}': {e}")
                print("Proceeding with the recorded actions.")
    else:
        # If no window was active during recording, ensure the desktop is active
        if not is_desktop_active():
            print("Switching to the desktop.")
            minimize_all_windows()

    # Process command actions
    start_time = time.time()

    for action in command_actions:
        while time.time() - start_time < action["time"]:
            time.sleep(0.01)  # Sleep for 10ms to reduce CPU usage

        if action["event"] == "key_down":
            pyautogui.keyDown(action["key"])
        elif action["event"] == "key_up":
            pyautogui.keyUp(action["key"])
        elif action["event"] == "mouse_down":
            pyautogui.mouseDown(button=action["button"])
        elif action["event"] == "mouse_up":
            pyautogui.mouseUp(button=action["button"])
        elif action["event"] == "mouse_move":
            fast_move(action["x"], action["y"])
        elif action["event"] == "mouse_scroll":
            scroll_mouse(action["delta"])

    print(f"Command '{command_name}' executed successfully.")
    dialogue = f"In your own words, tell the User you will perform the custom command they taught you and thank them for showing you how to {text}"
    response = Response(dialogue)
    tts(response)

def process_commands(text, argument_dictionary=None):
    command_executed_tags = []
    matched_commands = []

    if check_basic_commands(text, matched_commands):
        execute_command(text, matched_commands, command_executed_tags, argument_dictionary)
        return True

    shortcut_dir = './Mods/COMMANDS'
    if check_shortcuts(text, shortcut_dir, matched_commands):
        execute_shortcut(text, matched_commands, command_executed_tags)
        return True


    found_commands = check_custom_commands(text)
    if found_commands:
        for command_name in found_commands:
            execute_custom_command(text, command_name)
        return True

    return False
