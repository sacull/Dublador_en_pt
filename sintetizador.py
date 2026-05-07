"""
sintetizador.py
───────────────
Responsabilidade: sintetizar a fala traduzida usando Coqui XTTS-v2 com
clonagem de voz, ajustar cada segmento ao seu slot de tempo e montar
a trilha de áudio final sincronizada com o vídeo.
"""

import os
import tempfile
from pathlib import Path

import ffmpeg
from TTS.api import TTS
from pydub import AudioSegment
from tqdm import tqdm

from config import Config, XTTS_LANG


class Sintetizador:
    """
    Gera o áudio dublado segmento por segmento usando o Coqui XTTS-v2.

    Funcionalidades:
    - Clonagem de voz a partir de um trecho de referência do vídeo original
    - Checkpoint por segmento: retoma de onde parou se o processo for interrompido
    - Sincronização via filtro atempo do FFmpeg (preserva pitch, ajusta velocidade)
    - Fallback para silêncio se um segmento falhar, sem abortar o processo
    """

    def __init__(self, cfg: Config, pasta_temp: Path, device: str):
        self.cfg    = cfg
        self.tmp    = pasta_temp
        self.device = device

    def carregar_modelo(self) -> TTS:
        """Carrega o modelo XTTS-v2. Na 1ª vez faz download de ~1.8 GB."""
        print("\n[4/6] Carregando Coqui XTTS-v2…")
        print("      (1ª execução faz download de ~1.8 GB — aguarde)")
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def sintetizar(self, segmentos: list[dict], modelo: TTS) -> Path:
        """
        Sintetiza todos os segmentos e retorna o caminho da trilha final (WAV).
        """
        print(f"\n[5/6] Sintetizando {len(segmentos)} segmentos…")

        lang      = XTTS_LANG.get(self.cfg.idioma_destino, "pt")
        ref_wav   = str(self.tmp / "voz_referencia.wav")
        dur_total = int(segmentos[-1]["end"] * 1000) + 500   # ms
        trilha    = AudioSegment.silent(duration=dur_total)

        for i, seg in enumerate(tqdm(segmentos, desc="  TTS")):
            texto  = seg.get("texto_traduzido", "").strip()
            inicio = int(seg["start"] * 1000)
            fim    = int(seg["end"]   * 1000)
            slot   = fim - inicio

            if not texto:
                continue

            seg_audio = self._gerar_segmento(i, texto, lang, ref_wav, modelo, slot)
            seg_audio = self._ajustar_duracao(seg_audio, slot)
            trilha    = trilha.overlay(seg_audio, position=inicio)

        saida = self.tmp / "audio_dublado.wav"
        trilha.export(str(saida), format="wav")
        print(f"      Trilha final: {saida}")
        return saida

    # ── Métodos privados ──────────────────────────────────────────────────────

    def _gerar_segmento(
        self,
        indice: int,
        texto: str,
        lang: str,
        ref_wav: str,
        modelo: TTS,
        slot_ms: int,
    ) -> AudioSegment:
        """
        Gera (ou reutiliza do checkpoint) o áudio de um único segmento.
        Se falhar, retorna silêncio com a duração do slot.
        """
        seg_wav = self.tmp / f"seg_{indice:04d}.wav"

        # Checkpoint: se o arquivo já existe e é válido, reutiliza sem reprocessar
        if seg_wav.exists() and seg_wav.stat().st_size > 1000:
            return AudioSegment.from_wav(str(seg_wav))

        try:
            if self.cfg.clonar_voz:
                modelo.tts_to_file(
                    text=texto,
                    language=lang,
                    speaker_wav=ref_wav,
                    file_path=str(seg_wav),
                )
            else:
                modelo.tts_to_file(
                    text=texto,
                    language=lang,
                    file_path=str(seg_wav),
                )
            return AudioSegment.from_wav(str(seg_wav))

        except Exception as erro:
            print(f"\n  ⚠️  Segmento {indice} falhou ({erro}) — usando silêncio.")
            return AudioSegment.silent(duration=slot_ms)

    def _ajustar_duracao(self, audio: AudioSegment, slot_ms: int) -> AudioSegment:
        """
        Ajusta a duração do segmento gerado para caber exatamente no slot de tempo.

        - modo "stretch": usa o filtro atempo do FFmpeg para esticar/comprimir
          o áudio sem alterar o pitch (tom de voz).
        - modo "fill": corta o excesso ou preenche com silêncio.
        """
        atual = len(audio)
        if abs(atual - slot_ms) < 80:
            return audio   # diferença desprezível, não precisa ajustar

        if self.cfg.modo_sync == "stretch":
            return self._stretch(audio, slot_ms)
        else:
            return self._fill(audio, slot_ms)

    def _stretch(self, audio: AudioSegment, slot_ms: int) -> AudioSegment:
        """Altera a velocidade de reprodução via FFmpeg atempo, preservando o pitch."""
        fator = max(0.5, min(2.0, len(audio) / slot_ms))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fi:
            audio.export(fi.name, format="wav")
            fo = fi.name.replace(".wav", "_stretched.wav")

            (
                ffmpeg.input(fi.name)
                .filter("atempo", fator)
                .output(fo, acodec="pcm_s16le")
                .overwrite_output()
                .run(quiet=True)
            )

            resultado = AudioSegment.from_wav(fo)
            os.unlink(fi.name)
            os.unlink(fo)

        return resultado

    def _fill(self, audio: AudioSegment, slot_ms: int) -> AudioSegment:
        """Corta o áudio se for maior que o slot, ou adiciona silêncio se for menor."""
        if len(audio) > slot_ms:
            return audio[:slot_ms]
        return audio + AudioSegment.silent(duration=slot_ms - len(audio))