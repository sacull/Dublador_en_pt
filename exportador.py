"""
exportador.py
─────────────
Responsabilidade: combinar o stream de vídeo original com a nova trilha de
áudio dublada, gerando o arquivo de saída final via FFmpeg.
"""

from pathlib import Path

import ffmpeg

from config import Config


class Exportador:
    """
    Monta o vídeo final substituindo completamente o áudio original pela
    trilha dublada, sem recodificar o stream de vídeo desnecessariamente.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def exportar(self, audio_dublado: Path) -> None:
        """
        Combina o vídeo de entrada com o áudio dublado e salva o arquivo final.
        """
        print(f"\n[6/6] Montando vídeo final…")

        (
            ffmpeg.output(
                ffmpeg.input(self.cfg.video_entrada).video,
                ffmpeg.input(str(audio_dublado)).audio,
                self.cfg.video_saida,
                vcodec=self.cfg.video_codec,
                acodec=self.cfg.audio_codec,
                audio_bitrate=self.cfg.audio_bitrate,
                crf=self.cfg.crf,
                preset="fast",
                map_metadata=0,
                shortest=None,
            )
            .overwrite_output()
            .run(quiet=False)
        )

        print(f"\n✅  Pronto! Vídeo dublado salvo em: {self.cfg.video_saida}")