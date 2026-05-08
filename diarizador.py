"""
diarizador.py
─────────────
Diarização de falantes 100% offline usando:
  - resemblyzer  → embeddings de voz por segmento (modelo local, ~17 MB)
  - scikit-learn → KMeans para agrupar embeddings em clusters de falantes

Sem token, sem internet em runtime, sem servidor externo.

Instalação (única vez, baixa o modelo resemblyzer):
    pip install resemblyzer scikit-learn soundfile
"""

from pathlib import Path

import numpy as np

from config import Config

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    import soundfile as sf
    RESEMBLYZER_DISPONIVEL = True
except ImportError:
    RESEMBLYZER_DISPONIVEL = False


# Número máximo de falantes esperados no vídeo.
# O algoritmo escolhe automaticamente o melhor N até esse limite.
MAX_FALANTES = 6

# Duração mínima de um segmento (em segundos) para gerar embedding confiável.
# Segmentos muito curtos produzem embeddings instáveis.
MIN_DUR_EMBEDDING = 1.0


class Diarizador:
    """
    Identifica falantes por similaridade de voz e anota cada segmento
    com um ID de falante (SPEAKER_00, SPEAKER_01, …).

    Fluxo:
        1. Carrega o áudio completo
        2. Extrai um embedding de 256 dims por segmento via resemblyzer
        3. Agrupa embeddings com KMeans (n escolhido pelo menor inércia relativa)
        4. Mapeia cluster → SPEAKER_XX e anota os segmentos
        5. Extrai o melhor trecho de referência de cada falante para o XTTS
    """

    def __init__(self, cfg: Config, pasta_temp: Path):
        self.cfg = cfg
        self.tmp = pasta_temp

    def diarizar(self, wav: Path, segmentos: list[dict]) -> list[dict]:
        """
        Anota cada segmento com o campo 'speaker' e extrai referências de voz.
        Retorna os segmentos anotados.
        """
        if not RESEMBLYZER_DISPONIVEL:
            print("\n  ℹ️  resemblyzer/scikit-learn não instalados — diarização desativada.")
            print("       Instale com: pip install resemblyzer scikit-learn soundfile")
            print("       Todos os segmentos usarão a mesma voz de referência.")
            return self._anotar_falante_unico(segmentos)

        print("\n[1b/6] Diarizando falantes (offline, resemblyzer + KMeans)…")

        try:
            audio, sr = sf.read(str(wav))
            encoder   = VoiceEncoder()

            embeddings, indices_validos = self._extrair_embeddings(
                audio, sr, segmentos, encoder
            )

            n_falantes = self._estimar_n_falantes(embeddings)
            labels     = self._clusterizar(embeddings, n_falantes)
            segmentos  = self._anotar_segmentos(segmentos, indices_validos, labels)
            self._extrair_referencias(segmentos)

            falantes = {s.get("speaker", "SPEAKER_00") for s in segmentos}
            print(f"      {len(falantes)} falante(s) identificado(s): {sorted(falantes)}")

        except Exception as erro:
            print(f"\n  ⚠️  Diarização falhou ({erro}) — usando falante único.")
            segmentos = self._anotar_falante_unico(segmentos)

        return segmentos

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _extrair_embeddings(
        self,
        audio: np.ndarray,
        sr: int,
        segmentos: list[dict],
        encoder: "VoiceEncoder",
    ) -> tuple[np.ndarray, list[int]]:
        """
        Extrai um embedding por segmento. Ignora segmentos muito curtos.
        Retorna (matriz de embeddings, lista de índices dos segmentos válidos).
        """
        embeddings: list[np.ndarray] = []
        indices_validos: list[int]   = []

        for i, seg in enumerate(segmentos):
            duracao = seg["end"] - seg["start"]
            if duracao < MIN_DUR_EMBEDDING:
                continue

            inicio_amostra = int(seg["start"] * sr)
            fim_amostra    = int(seg["end"]   * sr)
            trecho         = audio[inicio_amostra:fim_amostra]

            if len(trecho) < int(sr * MIN_DUR_EMBEDDING):
                continue

            # resemblyzer espera mono float32 a 16 kHz
            trecho_proc = preprocess_wav(trecho, source_sr=sr)
            embedding   = encoder.embed_utterance(trecho_proc)
            embeddings.append(embedding)
            indices_validos.append(i)

        return np.array(embeddings), indices_validos

    # ── Estimativa do número de falantes ─────────────────────────────────────

    def _estimar_n_falantes(self, embeddings: np.ndarray) -> int:
        """
        Escolhe o número de clusters via elbow method simplificado:
        aumenta N enquanto a queda de inércia relativa for > 15%.
        Limita entre 1 e MAX_FALANTES.
        """
        n_max = min(MAX_FALANTES, len(embeddings))
        if n_max <= 1:
            return 1

        X = normalize(embeddings)
        inercias: list[float] = []

        for n in range(1, n_max + 1):
            km = KMeans(n_clusters=n, n_init=10, random_state=42)
            km.fit(X)
            inercias.append(km.inertia_)

        # Primeiro N onde a melhoria relativa cai abaixo de 15%
        for n in range(1, len(inercias)):
            melhoria = (inercias[n - 1] - inercias[n]) / (inercias[0] + 1e-9)
            if melhoria < 0.15:
                return n

        return n_max

    # ── Clusterização ─────────────────────────────────────────────────────────

    def _clusterizar(self, embeddings: np.ndarray, n: int) -> np.ndarray:
        """Agrupa os embeddings em N clusters e retorna os labels."""
        X  = normalize(embeddings)
        km = KMeans(n_clusters=n, n_init=10, random_state=42)
        return km.fit_predict(X)

    # ── Anotação ──────────────────────────────────────────────────────────────

    def _anotar_segmentos(
        self,
        segmentos: list[dict],
        indices_validos: list[int],
        labels: np.ndarray,
    ) -> list[dict]:
        """
        Propaga o label do cluster para o campo 'speaker' de cada segmento.
        Segmentos sem embedding herdam o falante do vizinho anterior.
        """
        mapa: dict[int, str] = {
            idx: f"SPEAKER_{label:02d}"
            for idx, label in zip(indices_validos, labels)
        }

        ultimo = "SPEAKER_00"
        for i in range(len(segmentos)):
            if i in mapa:
                ultimo = mapa[i]
            segmentos[i]["speaker"] = ultimo

        return segmentos

    # ── Extração de referências ───────────────────────────────────────────────

    def _extrair_referencias(self, segmentos: list[dict]) -> None:
        """
        Para cada falante, pega o segmento mais longo e salva como WAV
        de referência (voz_SPEAKER_XX.wav) para o XTTS usar na clonagem.
        Usa o audio_original.wav já extraído pelo Extrator.
        """
        try:
            import ffmpeg as _ffmpeg
        except ImportError:
            print("  ⚠️  ffmpeg-python não disponível — referências não extraídas.")
            return

        # Escolhe o segmento mais longo de cada falante como referência
        melhores: dict[str, dict] = {}
        for seg in segmentos:
            spk = seg.get("speaker", "SPEAKER_00")
            dur = seg["end"] - seg["start"]
            if spk not in melhores or dur > (melhores[spk]["end"] - melhores[spk]["start"]):
                melhores[spk] = seg

        wav_original = self.tmp / "audio_original.wav"

        for spk, seg in melhores.items():
            saida = self.tmp / f"voz_{spk}.wav"
            if saida.exists():
                continue

            duracao = min(seg["end"] - seg["start"], 15.0)

            try:
                (
                    _ffmpeg.input(str(wav_original), ss=seg["start"], t=duracao)
                    .output(str(saida), ac=1, ar=22050, acodec="pcm_s16le")
                    .overwrite_output()
                    .run(quiet=True)
                )
                print(f"      Ref. de voz: {saida.name} "
                      f"({seg['start']:.1f}s → {seg['start'] + duracao:.1f}s)")
            except Exception as e:
                print(f"  ⚠️  Não foi possível salvar referência para {spk}: {e}")

    # ── Modo degradado ────────────────────────────────────────────────────────

    @staticmethod
    def _anotar_falante_unico(segmentos: list[dict]) -> list[dict]:
        """Todos os segmentos recebem SPEAKER_00 (modo sem diarização)."""
        for seg in segmentos:
            seg.setdefault("speaker", "SPEAKER_00")
        return segmentos