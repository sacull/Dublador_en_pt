"""
extrator.py
───────────
Responsabilidade: extrair o áudio do vídeo de entrada em formato WAV,
pronto para transcrição e para uso como referência de clonagem de voz.
"""

from pathlib import Path

import ffmpeg

from config import Config


class Extrator:
    """
    Extrai duas faixas de áudio a partir do vídeo:

    - audio_original.wav  → áudio completo em mono 22050 Hz (entrada do Whisper)
    - voz_referencia.wav  → trecho inicial usado pelo XTTS para clonar a voz
    """

    def __init__(self, cfg: Config, pasta_temp: Path):
        self.cfg = cfg
        self.tmp = pasta_temp

    def extrair(self) -> Path:
        """
        Executa a extração e retorna o caminho do áudio completo (WAV).
        """
        print("\n[1/6] Extraindo áudio do vídeo…")

        audio_completo = self.tmp / "audio_original.wav"
        voz_referencia = self.tmp / "voz_referencia.wav"

        self._extrair_completo(audio_completo)
        self._extrair_referencia(voz_referencia)

        print(f"      Áudio completo : {audio_completo}")
        print(f"      Referência voz : {voz_referencia}  ({self.cfg.ref_duracao_s}s)")

        return audio_completo

    # ── Métodos privados ──────────────────────────────────────────────────────

    def _extrair_completo(self, saida: Path) -> None:
        """Converte o áudio inteiro para WAV mono 22050 Hz (exigido pelo XTTS)."""
        (
            ffmpeg.input(self.cfg.video_entrada)
            .output(str(saida), ac=1, ar=22050, acodec="pcm_s16le")
            .overwrite_output()
            .run(quiet=True)
        )

    def _extrair_referencia(self, saida: Path) -> None:
        """Extrai apenas os primeiros N segundos para usar como referência de voz."""
        (
            ffmpeg.input(self.cfg.video_entrada, ss=0, t=self.cfg.ref_duracao_s)
            .output(str(saida), ac=1, ar=22050, acodec="pcm_s16le")
            .overwrite_output()
            .run(quiet=True)
        )