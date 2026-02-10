# Language-detection-
Use of ffmpeg library to run the code 
import whisper


def detect_language(audio_file):

  
    model = whisper.load_model("tiny")

    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)

    lang = max(probs, key=probs.get)
    confidence = probs[lang] * 100

    return lang, confidence


if __name__ == "__main__":

    file = r"C:\Users\pytho\OneDrive\Desktop\web\sample.mp3"   

    lang, conf = detect_language(file)

    print("Detected Language:", lang.upper())
    print(f"Confidence: {conf:.2f}%")
