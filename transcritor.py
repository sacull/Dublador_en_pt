"""
transcritor.py
──────────────
Responsabilidade: transcrever o áudio com Whisper (OpenAI), gerando segmentos
com timestamps precisos de início e fim para cada trecho de fala.
"""

import json
from pathlib import Path

import whisper

from config import Config


class Transcritor:
    """
    Usa o modelo Whisper para transcrever o áudio e retornar uma lista de
    segmentos, cada um com: start, end, text (e timestamps por palavra).

    O resultado é salvo em transcricao.json na pasta temporária para debug
    e para permitir reuso sem nova transcrição.
    """

    def __init__(self, cfg: Config, pasta_temp: Path, device: str):
        self.cfg    = cfg
        self.tmp    = pasta_temp
        self.device = device

    def transcrever(self, wav: Path) -> list[dict]:
        """
        Transcreve o áudio e retorna a lista de segmentos.
        """
        print(f"\n[2/6] Transcrevendo com Whisper ({self.cfg.whisper_model})…")

        modelo    = whisper.load_model(self.cfg.whisper_model, device=self.device)
        resultado = modelo.transcribe(
            str(wav),
            language=self.cfg.idioma_origem,
            word_timestamps=True,
            verbose=False,
        )

        segmentos = resultado["segments"]
        print(f"      {len(segmentos)} segmentos encontrados.")

        self._salvar_cache(segmentos)
        segmentos = self._mesclar_curtos(segmentos)

        return segmentos

    # ── Métodos privados ──────────────────────────────────────────────────────

    def _salvar_cache(self, segmentos: list[dict]) -> None:
        """Salva a transcrição em JSON para debug ou reuso."""
        cache = self.tmp / "transcricao.json"
        cache.write_text(json.dumps(segmentos, ensure_ascii=False, indent=2))

    def _mesclar_curtos(self, segmentos: list[dict]) -> list[dict]:
        """
        Une segmentos com duração menor que cfg.min_seg_s ao segmento seguinte.
        Evita que o XTTS receba textos de 1–2 palavras, que geram áudio de má qualidade.
        """
        mesclados: list[dict] = []
        buffer = None

        for seg in segmentos:
            duracao = seg["end"] - seg["start"]

            if buffer is None:
                buffer = dict(seg)
            elif duracao < self.cfg.min_seg_s:
                buffer["end"]  = seg["end"]
                buffer["text"] = buffer["text"].rstrip() + " " + seg["text"].lstrip()
            else:
                mesclados.append(buffer)
                buffer = dict(seg)

        if buffer:
            mesclados.append(buffer)

        removidos = len(segmentos) - len(mesclados)
        if removidos:
            print(f"      {removidos} segmentos curtos mesclados → {len(mesclados)} no total.")

        return mesclados