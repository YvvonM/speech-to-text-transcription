import whisper
import gradio as gr
import sounddevice as sd
import numpy as np
import threading
import wave
import os
from docx import Document
from queue import Queue
from pydub import AudioSegment  # For audio format conversion

# Load Whisper Model (You can use "tiny", "base", "small", "medium", or "large")
model = whisper.load_model("base")

# Audio Config
RATE = 16000  # Sample rate
CHANNELS = 1  # Mono audio
BUFFER_SIZE = 4096  # Chunk size

# Global Variables
is_recording = False
transcription_text = ""
saved_transcription_path = "saved_transcription.txt"
audio_queue = Queue()

# Load previous transcription if available
if os.path.exists(saved_transcription_path):
    with open(saved_transcription_path, "r") as file:
        transcription_text = file.read()

def audio_callback(indata, frames, time, status):
    """Collects audio chunks in real-time."""
    if is_recording:
        audio_queue.put(indata.copy())

def start_transcription():
    """Starts real-time transcription."""
    global is_recording
    is_recording = True

    def process_audio():
        global transcription_text
        audio_data = []
        
        while is_recording:
            while not audio_queue.empty():
                audio_data.append(audio_queue.get())

        if audio_data:
            audio_np = np.concatenate(audio_data, axis=0)
            audio_wav_path = "real_time_audio.wav"
            with wave.open(audio_wav_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                wf.writeframes(audio_np.astype(np.int16).tobytes())

            result = model.transcribe(audio_wav_path)
            transcription_text += result["text"] + "\n"

    thread = threading.Thread(target=process_audio)
    thread.start()

    return "🎤 Recording started..."

def stop_transcription():
    """Stops real-time transcription."""
    global is_recording
    is_recording = False
    return transcription_text

def transcribe_audio_file(audio_path):
    """Transcribes an uploaded audio file."""
    global transcription_text
    temp_wav_path = "temp_audio.wav"

    # Convert audio file to WAV format if necessary
    if not audio_path.endswith(".wav"):
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(RATE).set_channels(1)  # Convert to mono, 16kHz
        audio.export(temp_wav_path, format="wav")
        audio_path = temp_wav_path

    # Transcribe with Whisper
    result = model.transcribe(audio_path)
    transcription_text = result["text"]
    
    return transcription_text

def save_transcription():
    """Saves transcription to a text file."""
    global transcription_text
    if not transcription_text.strip():
        return "⚠️ Nothing to save."
    
    with open(saved_transcription_path, "w") as file:
        file.write(transcription_text)
    
    return "✅ Transcription saved successfully!"

def load_transcription():
    """Loads the last saved transcription."""
    global transcription_text
    if os.path.exists(saved_transcription_path):
        with open(saved_transcription_path, "r") as file:
            transcription_text = file.read()
    
    return transcription_text

def download_transcription():
    """Downloads transcription as a .docx file."""
    global transcription_text
    if not transcription_text.strip():
        return None
    
    doc = Document()
    doc.add_paragraph(transcription_text)
    doc_path = "transcription.docx"
    doc.save(doc_path)
    return doc_path

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎤 Whisper AI - Real-Time & File-Based Speech Transcription")

    with gr.Row():
        start_btn = gr.Button("🎙️ Start Recording")
        stop_btn = gr.Button("🛑 Stop Recording")

    output_text = gr.Textbox(label="Transcription", interactive=True, lines=10)
    save_btn = gr.Button("💾 Save Progress")
    load_btn = gr.Button("🔄 Load Saved Transcription")
    download_btn = gr.File(label="📄 Download Transcription (.docx)")

    with gr.Row():
        file_input = gr.File(label="📂 Upload Audio File", type="filepath")
        transcribe_btn = gr.Button("📜 Transcribe File")

    start_btn.click(start_transcription, outputs=None)
    stop_btn.click(stop_transcription, outputs=output_text)
    save_btn.click(save_transcription, outputs=None)
    load_btn.click(load_transcription, outputs=output_text)
    download_trigger = gr.Button("📥 Download Transcription")

    download_trigger.click(download_transcription, outputs=download_btn)

    transcribe_btn.click(transcribe_audio_file, inputs=file_input, outputs=output_text)

demo.launch()
