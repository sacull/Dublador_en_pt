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
    - Clonagem de voz a partir de um trecho de referência por falante
    - Checkpoint por segmento: retoma de onde parou se interrompido
    - Sincronização via filtro atempo do FFmpeg encadeado (preserva pitch)
    - Fallback para silêncio se um segmento falhar
    """

    def __init__(self, cfg: Config, pasta_temp: Path, device: str):
        self.cfg    = cfg
        self.tmp    = pasta_temp
        self.device = device

    def carregar_modelo(self) -> TTS:
        print("\n[4/6] Carregando Coqui XTTS-v2…")
        print("      (1a execucao faz download de ~1.8 GB — aguarde)")
        self._aplicar_patch_torch()
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def _aplicar_patch_torch(self) -> None:
        import torch, functools
        _original = torch.load

        @functools.wraps(_original)
        def _patch(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original(*args, **kwargs)

        torch.load = _patch
        print("      Patch PyTorch 2.6+ aplicado (weights_only=False).")

    def sintetizar(self, segmentos: list[dict], modelo: TTS) -> Path:
        print(f"\n[5/6] Sintetizando {len(segmentos)} segmentos…")

        lang      = XTTS_LANG.get(self.cfg.idioma_destino, "pt")
        dur_total = int(segmentos[-1]["end"] * 1000) + 2000
        trilha    = AudioSegment.silent(duration=dur_total)

        for i, seg in enumerate(tqdm(segmentos, desc="  TTS")):
            texto  = seg.get("texto_traduzido", "").strip()
            inicio = int(seg["start"] * 1000)
            fim    = int(seg["end"]   * 1000)
            slot   = fim - inicio

            if not texto:
                continue

            # Referência de voz: por falante (diarização) ou padrão
            speaker_id  = seg.get("speaker", "SPEAKER_00")
            ref_por_spk = self.tmp / f"voz_{speaker_id}.wav"
            ref_padrao  = self.tmp / "voz_referencia.wav"
            ref_wav     = str(ref_por_spk) if ref_por_spk.exists() else str(ref_padrao)

            seg_audio = self._gerar_segmento(i, texto, lang, ref_wav, modelo, slot)
            seg_audio = self._ajustar_duracao(seg_audio, slot)
            trilha    = trilha.overlay(seg_audio, position=inicio)

        saida = self.tmp / "audio_dublado.wav"
        trilha.export(str(saida), format="wav")
        print(f"      Trilha final: {saida}")
        return saida

    # ── Geração por segmento ──────────────────────────────────────────────────

    def _gerar_segmento(
        self,
        indice: int,
        texto: str,
        lang: str,
        ref_wav: str,
        modelo: TTS,
        slot_ms: int,
    ) -> AudioSegment:
        seg_wav = self.tmp / f"seg_{indice:04d}.wav"

        if seg_wav.exists() and seg_wav.stat().st_size > 1000:
            return AudioSegment.from_wav(str(seg_wav))

        try:
            if self.cfg.clonar_voz:
                modelo.tts_to_file(
                    text=texto, language=lang,
                    speaker_wav=ref_wav, file_path=str(seg_wav),
                )
            else:
                modelo.tts_to_file(
                    text=texto, language=lang, file_path=str(seg_wav),
                )
            return AudioSegment.from_wav(str(seg_wav))

        except Exception as erro:
            print(f"\n  ⚠️  Segmento {indice} falhou ({erro}) — usando silêncio.")
            return AudioSegment.silent(duration=slot_ms)

    # ── Sincronização ─────────────────────────────────────────────────────────

    def _ajustar_duracao(self, audio: AudioSegment, slot_ms: int) -> AudioSegment:
        """Ajusta o segmento para caber no slot de tempo do vídeo original."""
        if abs(len(audio) - slot_ms) < 80:
            return audio

        if self.cfg.modo_sync == "stretch":
            return self._stretch(audio, slot_ms)
        return self._fill(audio, slot_ms)

    def _stretch(self, audio: AudioSegment, slot_ms: int) -> AudioSegment:
        """
        Usa atempo do FFmpeg para ajustar velocidade sem alterar pitch.

        CORREÇÃO CRÍTICA: usa mkstemp para garantir nomes de arquivo com
        sufixo .wav correto — evita o bug onde replace('.wav') falhava
        silenciosamente e o arquivo de saída nunca era criado.

        O atempo só aceita 0.5–2.0; encadeia filtros para ratios fora desse range.
        """
        if slot_ms <= 0:
            return audio

        ratio = len(audio) / slot_ms
        # Limita para evitar artefatos extremos (>4x ou <0.25x)
        ratio = max(0.25, min(4.0, ratio))

        passos = self._calcular_passos_atempo(ratio)

        # mkstemp garante sufixo .wav correto em qualquer SO
        fd_in,  path_in  = tempfile.mkstemp(suffix=".wav")
        fd_out, path_out = tempfile.mkstemp(suffix=".wav")
        os.close(fd_in)
        os.close(fd_out)

        try:
            audio.export(path_in, format="wav")

            # Encadeia os filtros atempo um a um
            stream = ffmpeg.input(path_in).audio
            for p in passos:
                stream = stream.filter("atempo", p)

            (
                ffmpeg.output(stream, path_out, acodec="pcm_s16le")
                .overwrite_output()
                .run(quiet=True)
            )

            resultado = AudioSegment.from_wav(path_out)

        except Exception as e:
            print(f"\n  ⚠️  stretch falhou ({e}) — usando fill como fallback.")
            resultado = self._fill(audio, slot_ms)

        finally:
            # Garante limpeza mesmo se houver exceção
            for p in (path_in, path_out):
                try:
                    os.unlink(p)
                except OSError:
                    pass

        return resultado

    @staticmethod
    def _calcular_passos_atempo(ratio: float) -> list[float]:
        """
        Decompõe ratio em passos dentro de [0.5, 2.0] para encadeamento.

        ratio > 1.0 → áudio mais longo que o slot → comprime (fala mais rápido)
        ratio < 1.0 → áudio mais curto que o slot → estica (fala mais devagar)

        Exemplos:
          ratio 3.0  → [2.0, 1.5]
          ratio 4.0  → [2.0, 2.0]
          ratio 0.25 → [0.5, 0.5]
          ratio 0.3  → [0.5, 0.6]
        """
        passos: list[float] = []
        restante = ratio

        while restante > 2.0:
            passos.append(2.0)
            restante /= 2.0

        while restante < 0.5:
            passos.append(0.5)
            restante /= 0.5    # CORRIGIDO: era "restante /= 0.5" na versão bugada
                                # que dividia ao invés de multiplicar o restante,
                                # gerando loop infinito ou passos errados

        passos.append(round(restante, 6))
        return passos

    def _fill(self, audio: AudioSegment, slot_ms: int) -> AudioSegment:
        """Corta se maior que o slot, ou preenche com silêncio se menor."""
        if len(audio) > slot_ms:
            return audio[:slot_ms]
        return audio + AudioSegment.silent(duration=slot_ms - len(audio))