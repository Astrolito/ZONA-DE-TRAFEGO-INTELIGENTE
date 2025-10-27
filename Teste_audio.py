# deteccao_sirene.py
import sounddevice as sd
import numpy as np
import librosa

def detectar_sirene(caminho_sirene="collected_audio/siren_1761581600.wav", duracao=3, limite=0.8):
    """
    Escuta o ambiente por alguns segundos e compara com o som de sirene.
    Retorna True se detectar som de ambulância.
    """

    try:
        # Carrega o som da sirene de referência
        sirene, sr_sirene = librosa.load(caminho_sirene, sr=None)

        print("🎧 Ouvindo o ambiente...")
        audio = sd.rec(int(duracao * sr_sirene), samplerate=sr_sirene, channels=1)
        sd.wait()
        print("🎙️ Gravação concluída.")

        # Converte para o mesmo formato
        audio = audio.flatten()

        # Calcula similaridade (correlação cruzada)
        correlacao = np.correlate(audio, sirene, mode='valid')
        semelhanca = np.max(correlacao) / len(sirene)

        print(f"🔊 Nível de semelhança: {semelhanca:.2f}")

        return semelhanca > limite

    except Exception as e:
        print(f"⚠️ Erro na detecção da sirene: {e}")
        return False