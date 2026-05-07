"""
Dublagem profissional de vídeo — 100% OFFLINE e gratuito
=========================================================
Stack (tudo roda localmente, sem internet após instalação):
  • Whisper        → transcrição com timestamps precisos
  • Argos Translate → tradução offline EN → PT (sem API, sem internet)
  • Coqui XTTS-v2  → TTS com clonagem de voz, 100% local
  • FFmpeg          → sincronização e exportação final

Dependências Python:
    pip install openai-whisper argostranslate TTS pydub ffmpeg-python tqdm torch

Sistema:
    ffmpeg instalado e no PATH
    Python 3.10 ou 3.11 (obrigatório para TTS/Coqui)

NENHUMA chave de API necessária. NENHUMA conexão com internet necessária após
o primeiro download dos modelos (feito automaticamente na 1ª execução).
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-INSTALAÇÃO DE DEPENDÊNCIAS
# ══════════════════════════════════════════════════════════════════════════════
DEPS = [
    "openai-whisper",
    "argostranslate",
    "TTS",
    "pydub",
    "ffmpeg-python",
    "tqdm",
    "torch",
]

def instalar_deps():
    for dep in DEPS:
        pkg = dep.replace("-", "_").lower()
        try:
            __import__(pkg)
        except ImportError:
            print(f"[setup] Instalando {dep}…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])

instalar_deps()

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import torch
import whisper
import argostranslate.package
import argostranslate.translate
from TTS.api import TTS
from pydub import AudioSegment
from tqdm import tqdm
import ffmpeg


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES — edite aqui antes de rodar
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # ── Arquivos ──────────────────────────────────────────────────────────────
    video_entrada:   str   = "video.mp4"
    video_saida:     str   = "video_dublado.mp4"

    # ── Idiomas ───────────────────────────────────────────────────────────────
    # Códigos Argos: en, es, fr, de, it, ja, zh, ru, pl, nl, ar, ko, pt…
    idioma_origem:   str   = "en"
    idioma_destino:  str   = "pt"   # pt = português (Argos não diferencia BR/PT)

    # ── Whisper ───────────────────────────────────────────────────────────────
    # tiny (rápido) | small | medium (recomendado) | large (mais preciso, lento)
    whisper_model:   str   = "medium"

    # ── Coqui XTTS-v2 ────────────────────────────────────────────────────────
    # True  → clona a voz extraída do próprio vídeo (recomendado)
    # False → usa voz padrão multilíngue do XTTS
    clonar_voz:      bool  = True

    # Duração do trecho inicial usado como referência de clonagem (mínimo 3s)
    # Aumente para 15–20s se o início do vídeo tiver ruído ou música
    ref_duracao_s:   float = 10.0

    # ── Sincronização ─────────────────────────────────────────────────────────
    # "stretch" → ajusta velocidade do áudio (sem alterar pitch) para caber no slot
    # "fill"    → corta ou preenche com silêncio
    modo_sync:       str   = "stretch"

    # Segmentos com duração < min_seg_s são mesclados ao próximo
    min_seg_s:       float = 1.5

    # ── Hardware ──────────────────────────────────────────────────────────────
    usar_gpu:        bool  = torch.cuda.is_available()

    # ── Exportação ────────────────────────────────────────────────────────────
    video_codec:     str   = "libx264"
    audio_codec:     str   = "aac"
    audio_bitrate:   str   = "192k"
    crf:             int   = 18


# ══════════════════════════════════════════════════════════════════════════════
# MAPEAMENTO DE IDIOMAS Argos → XTTS
# ══════════════════════════════════════════════════════════════════════════════
XTTS_LANG = {
    "pt": "pt", "en": "en", "es": "es",
    "fr": "fr", "de": "de", "it": "it",
    "ja": "ja", "zh": "zh-cn", "pl": "pl",
    "nl": "nl", "ru": "ru",   "tr": "tr",
    "ko": "ko", "ar": "ar",   "cs": "cs",
    "hu": "hu", "ro": "ro",
}


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
class Dublador:

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.tmp    = Path(tempfile.mkdtemp(prefix="dub_"))
        self.device = "cuda" if cfg.usar_gpu else "cpu"
        print(f"[•] Dispositivo : {self.device.upper()}")
        print(f"[•] Pasta temp  : {self.tmp}")

    # ── 1. Extrair áudio WAV ──────────────────────────────────────────────────
    def extrair_audio(self) -> Path:
        print("\n[1/6] Extraindo áudio do vídeo…")
        saida = self.tmp / "audio_original.wav"
        ref   = self.tmp / "voz_referencia.wav"

        (
            ffmpeg.input(self.cfg.video_entrada)
            .output(str(saida), ac=1, ar=22050, acodec="pcm_s16le")
            .overwrite_output().run(quiet=True)
        )
        (
            ffmpeg.input(self.cfg.video_entrada, ss=0, t=self.cfg.ref_duracao_s)
            .output(str(ref), ac=1, ar=22050, acodec="pcm_s16le")
            .overwrite_output().run(quiet=True)
        )
        print(f"      Áudio completo : {saida}")
        print(f"      Referência voz : {ref}  ({self.cfg.ref_duracao_s}s)")
        return saida

    # ── 2. Transcrever com Whisper ────────────────────────────────────────────
    def transcrever(self, wav: Path) -> list[dict]:
        print(f"\n[2/6] Transcrevendo com Whisper ({self.cfg.whisper_model})…")
        model  = whisper.load_model(self.cfg.whisper_model, device=self.device)
        result = model.transcribe(
            str(wav),
            language=self.cfg.idioma_origem,
            word_timestamps=True,
            verbose=False,
        )
        segs = result["segments"]
        print(f"      {len(segs)} segmentos encontrados.")
        (self.tmp / "transcricao.json").write_text(
            json.dumps(segs, ensure_ascii=False, indent=2)
        )
        return segs

    # ── 2b. Mesclar segmentos muito curtos ────────────────────────────────────
    def mesclar_curtos(self, segs: list[dict]) -> list[dict]:
        mesclados, buf = [], None
        for seg in segs:
            dur = seg["end"] - seg["start"]
            if buf is None:
                buf = dict(seg)
            elif dur < self.cfg.min_seg_s:
                buf["end"]  = seg["end"]
                buf["text"] = buf["text"].rstrip() + " " + seg["text"].lstrip()
            else:
                mesclados.append(buf)
                buf = dict(seg)
        if buf:
            mesclados.append(buf)
        removidos = len(segs) - len(mesclados)
        if removidos:
            print(f"      {removidos} segmentos curtos mesclados → {len(mesclados)} no total.")
        return mesclados

    # ── 3. Traduzir com Argos Translate (100% offline) ────────────────────────
    def preparar_argos(self):
        """
        Baixa o pacote de tradução EN→PT na 1ª execução e instala localmente.
        Nas execuções seguintes detecta que já está instalado e não baixa nada.
        """
        from_code = self.cfg.idioma_origem
        to_code   = self.cfg.idioma_destino

        # Verifica se o par já está instalado
        instalados = argostranslate.translate.get_installed_languages()
        pares_ok   = [
            l for l in instalados
            if l.code == from_code and any(t.to_lang.code == to_code for t in l.translations_from)
        ]

        if pares_ok:
            print(f"      Pacote {from_code}→{to_code} já instalado.")
            return

        print(f"      Baixando pacote Argos {from_code}→{to_code} (~100 MB, apenas 1 vez)…")
        argostranslate.package.update_package_index()
        pacotes     = argostranslate.package.get_available_packages()
        pacote_alvo = next(
            (p for p in pacotes if p.from_code == from_code and p.to_code == to_code),
            None,
        )
        if pacote_alvo is None:
            raise RuntimeError(
                f"Par de idiomas {from_code}→{to_code} não encontrado no Argos. "
                f"Verifique os códigos em: https://github.com/argosopentech/argos-translate"
            )
        argostranslate.package.install_from_path(pacote_alvo.download())
        print("      Pacote instalado com sucesso.")

    def traduzir(self, segs: list[dict]) -> list[dict]:
        print(f"\n[3/6] Traduzindo {len(segs)} segmentos com Argos Translate (offline)…")
        self.preparar_argos()

        from_code  = self.cfg.idioma_origem
        to_code    = self.cfg.idioma_destino
        instalados = argostranslate.translate.get_installed_languages()
        lang_from  = next(l for l in instalados if l.code == from_code)
        lang_to    = next(l for l in instalados if l.code == to_code)
        tradutor   = lang_from.get_translation(lang_to)

        for seg in tqdm(segs, desc="  Traduzindo"):
            texto = seg["text"].strip()
            seg["texto_traduzido"] = tradutor.translate(texto) if texto else ""

        # Mostra amostra da tradução
        for seg in segs[:3]:
            print(f"  [{seg['start']:.1f}s→{seg['end']:.1f}s]  {seg['text'][:55]}")
            print(f"   ↳ {seg['texto_traduzido'][:55]}")

        return segs

    # ── 4. Carregar Coqui XTTS-v2 ────────────────────────────────────────────
    def carregar_tts(self) -> TTS:
        print("\n[4/6] Carregando Coqui XTTS-v2…")
        print("      (1ª execução faz download de ~1.8 GB — aguarde)")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        return tts

    # ── 5. Sintetizar cada segmento ───────────────────────────────────────────
    def sintetizar(self, segs: list[dict], tts: TTS) -> Path:
        print(f"\n[5/6] Sintetizando {len(segs)} segmentos…")
        lang      = XTTS_LANG.get(self.cfg.idioma_destino, "pt")
        ref_wav   = str(self.tmp / "voz_referencia.wav")
        dur_total = int(segs[-1]["end"] * 1000) + 500
        trilha    = AudioSegment.silent(duration=dur_total)

        for i, seg in enumerate(tqdm(segs, desc="  TTS")):
            texto  = seg.get("texto_traduzido", "").strip()
            inicio = int(seg["start"] * 1000)
            fim    = int(seg["end"]   * 1000)
            slot   = fim - inicio

            if not texto:
                continue

            seg_wav = self.tmp / f"seg_{i:04d}.wav"

            # Checkpoint: reutiliza segmentos já gerados
            if seg_wav.exists() and seg_wav.stat().st_size > 1000:
                seg_audio = AudioSegment.from_wav(str(seg_wav))
            else:
                try:
                    if self.cfg.clonar_voz:
                        tts.tts_to_file(
                            text=texto,
                            language=lang,
                            speaker_wav=ref_wav,
                            file_path=str(seg_wav),
                        )
                    else:
                        tts.tts_to_file(
                            text=texto,
                            language=lang,
                            file_path=str(seg_wav),
                        )
                    seg_audio = AudioSegment.from_wav(str(seg_wav))
                except Exception as e:
                    print(f"\n  ⚠️  Segmento {i} falhou ({e}) — usando silêncio.")
                    seg_audio = AudioSegment.silent(duration=slot)

            seg_audio = self._ajustar_duracao(seg_audio, slot)
            trilha    = trilha.overlay(seg_audio, position=inicio)

        saida = self.tmp / "audio_dublado.wav"
        trilha.export(str(saida), format="wav")
        print(f"      Trilha final: {saida}")
        return saida

    # ── Ajuste de duração via FFmpeg atempo (preserva pitch) ──────────────────
    def _ajustar_duracao(self, audio: AudioSegment, slot_ms: int) -> AudioSegment:
        atual = len(audio)
        if abs(atual - slot_ms) < 80:
            return audio

        if self.cfg.modo_sync == "stretch":
            fator = max(0.5, min(2.0, atual / slot_ms))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fi:
                audio.export(fi.name, format="wav")
                fo = fi.name.replace(".wav", "_s.wav")
                (
                    ffmpeg.input(fi.name)
                    .filter("atempo", fator)
                    .output(fo, acodec="pcm_s16le")
                    .overwrite_output().run(quiet=True)
                )
                resultado = AudioSegment.from_wav(fo)
                os.unlink(fi.name)
                os.unlink(fo)
            return resultado
        else:
            if atual > slot_ms:
                return audio[:slot_ms]
            return audio + AudioSegment.silent(duration=slot_ms - atual)

    # ── 6. Montar vídeo final ─────────────────────────────────────────────────
    def combinar(self, audio_dublado: Path):
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

    # ── Pipeline completa ─────────────────────────────────────────────────────
    def dublar(self):
        wav  = self.extrair_audio()
        segs = self.transcrever(wav)
        segs = self.mesclar_curtos(segs)
        segs = self.traduzir(segs)
        tts  = self.carregar_tts()
        aud  = self.sintetizar(segs, tts)
        self.combinar(aud)


# ══════════════════════════════════════════════════════════════════════════════
# PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = Config(
        video_entrada  = "video.mp4",
        video_saida    = "video_dublado.mp4",
        idioma_origem  = "en",
        idioma_destino = "pt",
        whisper_model  = "medium",
        clonar_voz     = True,
        ref_duracao_s  = 10.0,
        modo_sync      = "stretch",
    )

    Dublador(cfg).dublar()