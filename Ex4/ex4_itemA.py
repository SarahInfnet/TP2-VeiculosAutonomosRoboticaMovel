from pathlib import Path
import csv
import time

import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

from detector_faixas import (
    DetectorFaixas,
    configuracao_metadrive,
    configuracao_imagem_real,
)


# ================= CONFIGURACAO =================

RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "dados"
SAIDA = RAIZ / "saidas" / "itemA"

SAIDA.mkdir(parents=True, exist_ok=True)

TOTAL_STEPS = 200
LARGURA = 1280
ALTURA = 720
SEMENTE = 42

IMAGEM_REAL = DADOS / "imagem_real.png"

CAMPOS_CSV = [
    "step",
    "episodio",
    "direcao",
    "aceleracao",
    "desvio_px",
    "desvio_m",
    "raio_curvatura_m",
    "raio_curvatura_px",
    "curvatura_px_inversa",
    "velocidade_kmh",
    "faixas_detectadas",
    "tempo_detector_ms",
]


# ================= METADRIVE =================

def criar_ambiente():
    """Configura o simulador com camera frontal."""
    config = {
        "use_render": True,
        "image_observation": True,
        "norm_pixel": False,
        "stack_size": 1,
        "map": "SSS",
        "traffic_density": 0.15,
        "start_seed": SEMENTE,
        "num_scenarios": 1,
        "sensors": {
            "rgb_camera": (
                RGBCamera,
                LARGURA,
                ALTURA,
            )
        },
        "vehicle_config": {
            "image_source": "rgb_camera"
        },
        "log_level": 50,
    }

    return MetaDriveEnv(config)


def obter_frame(observacao):
    """Converte o frame RGB do MetaDrive para BGR."""
    if (
        not isinstance(observacao, dict)
        or "image" not in observacao
    ):
        raise RuntimeError(
            "Observacao sem imagem da camera frontal."
        )

    imagem = np.asarray(observacao["image"])

    if imagem.ndim == 4:
        imagem = imagem[:, :, :, -1]

    if imagem.ndim != 3:
        raise RuntimeError(
            f"Formato inesperado: {imagem.shape}"
        )

    if np.issubdtype(imagem.dtype, np.floating):
        if imagem.max() <= 1.0:
            imagem = imagem * 255.0

    imagem = np.clip(
        imagem,
        0,
        255,
    ).astype(np.uint8)

    if imagem.shape[2] == 4:
        return cv2.cvtColor(
            imagem,
            cv2.COLOR_RGBA2BGR,
        )

    if imagem.shape[2] == 3:
        return cv2.cvtColor(
            imagem,
            cv2.COLOR_RGB2BGR,
        )

    raise RuntimeError(
        f"Numero de canais inesperado: {imagem.shape}"
    )


def obter_velocidade(ambiente):
    """Velocidade do veiculo informada pelo MetaDrive."""
    return float(ambiente.agent.speed)


# ================= POLITICA ALEATORIA =================

def gerar_acao(gerador):
    """Gera direcao aleatoria e aceleracao positiva.

    A politica e aleatoria, mas usa intervalos restritos
    para favorecer o deslocamento do veiculo.
    """
    direcao = gerador.uniform(-0.1, 0.1)
    aceleracao = gerador.uniform(0.7, 1.0)

    return np.array(
        [direcao, aceleracao],
        dtype=np.float32,
    )


# ================= SOBREPOSICAO =================

def desenhar_informacoes(
    frame,
    step,
    episodio,
    velocidade,
    medidas,
    tempo_ms,
):
    """Adiciona os dados da execucao sobre a imagem."""
    resultado = frame.copy()

    cv2.rectangle(
        resultado,
        (0, 15),
        (650, 70),
        (20, 30, 45),
        -1,
    )

    cv2.putText(
        resultado,
        (
            f"Step: {step:03d}/{TOTAL_STEPS} | "
            f"Episodio: {episodio} | "
            f"Velocidade: {velocidade:.1f} km/h"
        ),
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        resultado,
        f"Detector: {tempo_ms:.1f} ms",
        (35, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    if medidas["detectado"]:
        status = "Faixas detectadas"
        cor = (0, 255, 0)
    else:
        status = "Faixas nao identificadas"
        cor = (0, 0, 255)

    cv2.putText(
        resultado,
        status,
        (35, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        cor,
        2,
    )

    return resultado


# ================= COMPARACAO VISUAL =================

def gerar_comparacao(frame_simulado):
    """Compara a fotografia real com o frame do MetaDrive."""
    if not IMAGEM_REAL.is_file():
        print(
            "Comparacao nao gerada. "
            f"Arquivo ausente: {IMAGEM_REAL}"
        )
        return

    imagem_real = cv2.imread(str(IMAGEM_REAL))

    if imagem_real is None:
        print(
            f"Erro ao abrir a imagem: {IMAGEM_REAL}"
        )
        return

    detector_real = DetectorFaixas(
        configuracao_imagem_real()
    )

    detector_simulado = DetectorFaixas(
        configuracao_metadrive()
    )

    resultado_real, medidas_real, painel_real = (
        detector_real.processar_frame(imagem_real)
    )

    resultado_simulado, medidas_simuladas, painel_simulado = (
        detector_simulado.processar_frame(frame_simulado)
    )

    resultado_real = cv2.resize(
        resultado_real,
        (640, 360),
    )

    resultado_simulado = cv2.resize(
        resultado_simulado,
        (640, 360),
    )

    # Indica explicitamente se houve deteccao.
    for imagem, medidas in (
        (resultado_real, medidas_real),
        (resultado_simulado, medidas_simuladas),
    ):
        if medidas["detectado"]:
            status = "DETECTADO"
            cor = (50, 220, 50)
        else:
            status = "NAO DETECTADO"
            cor = (50, 50, 240)

        cv2.rectangle(
            imagem,
            (0, 315),
            (640, 360),
            (20, 30, 45),
            -1,
        )

        cv2.putText(
            imagem,
            status,
            (15, 344),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            cor,
            2,
        )

    comparacao = np.hstack(
        (resultado_real, resultado_simulado)
    )

    cv2.rectangle(
        comparacao,
        (0, 0),
        (1280, 45),
        (20, 30, 45),
        -1,
    )

    cv2.putText(
        comparacao,
        "FOTOGRAFIA REAL",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        comparacao,
        "METADRIVE - SIMULADO",
        (660, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    caminho = SAIDA / "comparacao_real_simulada.png"

    cv2.imwrite(
        str(caminho),
        comparacao,
    )

    cv2.imwrite(
        str(SAIDA / "painel_imagem_real.png"),
        painel_real,
    )

    cv2.imwrite(
        str(SAIDA / "painel_imagem_simulada.png"),
        painel_simulado,
    )

    print(f"\nComparacao salva: {caminho}")
    print(
        "Fotografia real: "
        f"detectado={medidas_real['detectado']}"
    )
    print(
        "MetaDrive: "
        f"detectado={medidas_simuladas['detectado']}"
    )


# ================= CSV =================

def salvar_csv(metricas):
    """Salva as medidas coletadas durante a simulacao."""
    caminho = SAIDA / "metricas_200_steps.csv"

    with caminho.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:
        escritor = csv.DictWriter(
            arquivo,
            fieldnames=CAMPOS_CSV,
        )

        escritor.writeheader()
        escritor.writerows(metricas)

    return caminho


# ================= EXECUCAO PRINCIPAL =================

def main():
    print("=== EXERCICIO 4 - ITEM A ===")
    print("Iniciando MetaDrive...")

    detector = DetectorFaixas(
        configuracao_metadrive()
    )

    ambiente = criar_ambiente()
    gerador = np.random.default_rng(SEMENTE)

    metricas = []
    deteccoes = 0
    episodio = 1

    frame_comparacao = None
    primeiro_frame = None
    video = None

    try:
        observacao, info = ambiente.reset(
            seed=SEMENTE
        )

        caminho_video = (
            SAIDA / "simulacao_faixas.mp4"
        )

        video = cv2.VideoWriter(
            str(caminho_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (LARGURA, ALTURA),
        )

        if not video.isOpened():
            raise RuntimeError(
                "Nao foi possivel criar o video."
            )

        for step in range(1, TOTAL_STEPS + 1):
            # Politica aleatoria com aceleracao positiva.
            acao = gerar_acao(gerador)

            observacao, recompensa, terminou, truncou, info = (
                ambiente.step(acao)
            )

            frame = obter_frame(observacao)

            if primeiro_frame is None:
                primeiro_frame = frame.copy()

                cv2.imwrite(
                    str(SAIDA / "frame_simulado.png"),
                    primeiro_frame,
                )

            inicio = time.perf_counter()

            resultado, medidas, painel = (
                detector.processar_frame(frame)
            )

            tempo_ms = (
                time.perf_counter() - inicio
            ) * 1000

            velocidade = obter_velocidade(ambiente)

            if medidas["detectado"]:
                deteccoes += 1

                if frame_comparacao is None:
                    frame_comparacao = frame.copy()

            metricas.append({
                "step": step,
                "episodio": episodio,
                "direcao": float(acao[0]),
                "aceleracao": float(acao[1]),
                "desvio_px": medidas["desvio_px"],
                "desvio_m": medidas["desvio_m"],
                "raio_curvatura_m": (
                    medidas["raio_curvatura_m"]
                ),
                "raio_curvatura_px": (
                    medidas["raio_curvatura_px"]
                ),
                "curvatura_px_inversa": (
                    medidas["curvatura_px_inversa"]
                ),
                "velocidade_kmh": velocidade,
                "faixas_detectadas": medidas["detectado"],
                "tempo_detector_ms": round(
                    tempo_ms,
                    3,
                ),
            })

            resultado = desenhar_informacoes(
                resultado,
                step,
                episodio,
                velocidade,
                medidas,
                tempo_ms,
            )

            video.write(resultado)

            visualizacao = cv2.resize(
                resultado,
                (960, 540),
            )

            cv2.imshow(
                "MetaDrive - Detector de faixas",
                visualizacao,
            )

            if step in (1, 100, 200):
                cv2.imwrite(
                    str(
                        SAIDA / f"frame_{step:03d}.png"
                    ),
                    resultado,
                )

                cv2.imwrite(
                    str(
                        SAIDA / f"painel_{step:03d}.png"
                    ),
                    painel,
                )

            if medidas["desvio_m"] is not None:
                texto_desvio = (
                    f"{medidas['desvio_m']:+.3f} m"
                )
            elif medidas["desvio_px"] is not None:
                texto_desvio = (
                    f"{medidas['desvio_px']:+.1f} px"
                )
            else:
                texto_desvio = "indisponivel"

            print(
                f"Step {step:03d}/{TOTAL_STEPS}"
                f" | Episodio {episodio}"
                f" | Desvio: {texto_desvio}"
                f" | Velocidade: {velocidade:.1f} km/h"
                f" | Detector: {tempo_ms:.1f} ms"
            )

            # ESC permite interromper a execucao.
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 27:
                print(
                    "Execucao interrompida pelo usuario."
                )
                break

            # Reinicia o episodio quando necessario.
            if terminou or truncou:
                if step < TOTAL_STEPS:
                    episodio += 1
                    observacao, info = ambiente.reset()

        if frame_comparacao is None:
            frame_comparacao = primeiro_frame

        if frame_comparacao is not None:
            gerar_comparacao(frame_comparacao)

    finally:
        # Preserva os registros mesmo em caso de erro.
        caminho_csv = salvar_csv(metricas)

        if video is not None:
            video.release()

        ambiente.close()
        cv2.destroyAllWindows()

        print("\n=== RESULTADOS ===")
        print(f"Steps registrados: {len(metricas)}")

        print(
            "Curvatura em pixels: estimativa no "
            "plano BEV, nao em metros."
        )

        print(
            f"Deteccoes validas: "
            f"{deteccoes}/{len(metricas)}"
        )

        print(f"CSV: {caminho_csv}")
        print(f"Resultados: {SAIDA}")

        if len(metricas) == TOTAL_STEPS:
            print("Validacao de 200 steps: OK")
        else:
            print(
                "Validacao de 200 steps: INCOMPLETA"
            )


if __name__ == "__main__":
    main()