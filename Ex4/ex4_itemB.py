from pathlib import Path
import csv
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from ultralytics import YOLO

# =========================================================
# CONFIGURACOES
# =========================================================

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "saidas" / "itemB"
SAIDA.mkdir(parents=True, exist_ok=True)

TOTAL_STEPS = 500
SEMENTE = 42

LARGURA = 1280
ALTURA = 720

MODELO_YOLO = "yolov8n.pt"
CONFIANCA_MINIMA = 0.25
RESOLUCAO_YOLO = 640

# 1 = YOLO em todos os frames.
INTERVALO_YOLO = 1

# Exibir as janelas durante as duas execucoes.
MOSTRAR_JANELA = True

# Politica aleatoria restrita, como no Item A.
DIRECAO_MIN = -0.1
DIRECAO_MAX = 0.1
ACELERACAO_MIN = 0.7
ACELERACAO_MAX = 1.0

CATEGORIAS = {
    "car": "veiculos",
    "truck": "veiculos",
    "bus": "veiculos",
    "motorcycle": "veiculos",
    "person": "pedestres",
    "stop sign": "sinais",
    "traffic light": "sinais",
}

CORES = {
    "veiculos": (0, 220, 0),
    "pedestres": (255, 160, 0),
    "sinais": (0, 220, 255),
}

CAMPOS_CSV = [
    "step",
    "episodio",
    "veiculos",
    "pedestres",
    "sinais",
    "total_objetos",
    "inferencia_executada",
    "tempo_yolo_ms",
    "velocidade_kmh",
]

# =========================================================
# AMBIENTE METADRIVE
# =========================================================

def criar_ambiente():
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
    if (
        not isinstance(observacao, dict)
        or "image" not in observacao
    ):
        raise RuntimeError(
            "A observacao nao contem imagem da camera."
        )

    imagem = np.asarray(observacao["image"])

    if imagem.ndim == 4:
        imagem = imagem[:, :, :, -1]

    if imagem.ndim != 3:
        raise RuntimeError(
            f"Formato de imagem inesperado: {imagem.shape}"
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


# =========================================================
# POLITICA ALEATORIA
# =========================================================

def gerar_acoes():
    """Gera a mesma sequencia para os dois testes."""
    gerador = np.random.default_rng(SEMENTE)

    acoes = []

    for _ in range(TOTAL_STEPS):
        direcao = gerador.uniform(
            DIRECAO_MIN,
            DIRECAO_MAX,
        )

        aceleracao = gerador.uniform(
            ACELERACAO_MIN,
            ACELERACAO_MAX,
        )

        acoes.append(
            np.array(
                [direcao, aceleracao],
                dtype=np.float32,
            )
        )

    return acoes


# =========================================================
# YOLO
# =========================================================

def detectar_objetos(modelo, frame):
    """Aplica YOLO e retorna deteccoes por categoria."""
    inicio = time.perf_counter()

    resultados = modelo.predict(
        source=frame,
        imgsz=RESOLUCAO_YOLO,
        conf=CONFIANCA_MINIMA,
        verbose=False,
    )

    tempo_ms = (
        time.perf_counter() - inicio
    ) * 1000

    contagens = {
        "veiculos": 0,
        "pedestres": 0,
        "sinais": 0,
    }

    deteccoes = []

    for resultado in resultados:
        if resultado.boxes is None:
            continue

        for box in resultado.boxes:
            classe_id = int(box.cls[0].item())
            confianca = float(box.conf[0].item())

            nome_classe = modelo.names[classe_id]

            categoria = CATEGORIAS.get(nome_classe)

            if categoria is None:
                continue

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            contagens[categoria] += 1

            deteccoes.append({
                "caixa": (x1, y1, x2, y2),
                "classe": nome_classe,
                "categoria": categoria,
                "confianca": confianca,
            })

    return deteccoes, contagens, tempo_ms


def desenhar_deteccoes(frame, deteccoes):
    resultado = frame.copy()

    for deteccao in deteccoes:
        x1, y1, x2, y2 = deteccao["caixa"]

        categoria = deteccao["categoria"]
        classe = deteccao["classe"]
        confianca = deteccao["confianca"]

        cor = CORES[categoria]

        cv2.rectangle(
            resultado,
            (x1, y1),
            (x2, y2),
            cor,
            3,
        )

        rotulo = (
            f"{classe} {confianca:.2f}"
        )

        tamanho, _ = cv2.getTextSize(
            rotulo,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2,
        )

        topo = max(y1 - 32, 0)

        cv2.rectangle(
            resultado,
            (x1, topo),
            (x1 + tamanho[0] + 12, topo + 30),
            cor,
            -1,
        )

        cv2.putText(
            resultado,
            rotulo,
            (x1 + 6, topo + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
        )

    return resultado


# =========================================================
# INTERFACE
# =========================================================

def desenhar_painel(
    frame,
    step,
    episodio,
    com_yolo,
    contagens,
    tempo_yolo_ms,
    fps_atual,
    inferencia_executada,
):
    resultado = frame.copy()

    cv2.rectangle(
        resultado,
        (0, 0),
        (790, 185),
        (25, 30, 40),
        -1,
    )

    modo = "COM YOLO" if com_yolo else "SEM YOLO"

    linhas = [
        f"MetaDrive - {modo}",
        f"Step: {step:03d}/{TOTAL_STEPS} | Episodio: {episodio}",
        (
            f"Veiculos: {contagens['veiculos']} | "
            f"Pedestres: {contagens['pedestres']} | "
            f"Sinais: {contagens['sinais']}"
        ),
        f"FPS observado: {fps_atual:.1f}",
    ]

    if com_yolo:
        if inferencia_executada:
            linhas.append(
                f"Inferencia YOLO: {tempo_yolo_ms:.1f} ms"
            )
        else:
            linhas.append(
                "YOLO: frame ignorado (otimizacao)"
            )
    else:
        linhas.append("YOLO desativado")

    for indice, linha in enumerate(linhas):
        cv2.putText(
            resultado,
            linha,
            (15, 30 + indice * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )

    return resultado


# =========================================================
# EXECUCAO DO EXPERIMENTO
# =========================================================

def executar_experimento(
    com_yolo,
    acoes,
    modelo=None,
):
    modo = "com_yolo" if com_yolo else "sem_yolo"

    print(f"\n=== INICIANDO {modo.upper()} ===")

    ambiente = criar_ambiente()

    registros = []
    episodio = 1

    ultima_deteccoes = []
    ultima_contagem = {
        "veiculos": 0,
        "pedestres": 0,
        "sinais": 0,
    }

    total_inferencias = 0
    soma_tempo_yolo_ms = 0.0

    inicio_experimento = None

    try:
        observacao, info = ambiente.reset(
            seed=SEMENTE
        )

        inicio_experimento = time.perf_counter()

        for indice, acao in enumerate(acoes):
            step = indice + 1

            (
                observacao,
                recompensa,
                terminou,
                truncou,
                info,
            ) = ambiente.step(acao)

            frame = obter_frame(observacao)

            inferencia_executada = False
            tempo_yolo_ms = 0.0

            if com_yolo:
                inferencia_executada = (
                    indice % INTERVALO_YOLO == 0
                )

                if inferencia_executada:
                    (
                        ultima_deteccoes,
                        ultima_contagem,
                        tempo_yolo_ms,
                    ) = detectar_objetos(
                        modelo,
                        frame,
                    )

                    total_inferencias += 1
                    soma_tempo_yolo_ms += tempo_yolo_ms

                resultado = desenhar_deteccoes(
                    frame,
                    ultima_deteccoes,
                )
            else:
                resultado = frame.copy()

            contagens = (
                ultima_contagem.copy()
                if com_yolo
                else {
                    "veiculos": 0,
                    "pedestres": 0,
                    "sinais": 0,
                }
            )

            velocidade = float(
                ambiente.agent.speed
            )

            registros.append({
                "step": step,
                "episodio": episodio,
                "veiculos": contagens["veiculos"],
                "pedestres": contagens["pedestres"],
                "sinais": contagens["sinais"],
                "total_objetos": sum(
                    contagens.values()
                ),
                "inferencia_executada": (
                    inferencia_executada
                ),
                "tempo_yolo_ms": round(
                    tempo_yolo_ms,
                    3,
                ),
                "velocidade_kmh": velocidade,
            })

            tempo_decorrido = (
                time.perf_counter()
                - inicio_experimento
            )

            fps_atual = (
                step / tempo_decorrido
                if tempo_decorrido > 0
                else 0.0
            )

            resultado = desenhar_painel(
                resultado,
                step,
                episodio,
                com_yolo,
                contagens,
                tempo_yolo_ms,
                fps_atual,
                inferencia_executada,
            )

            if MOSTRAR_JANELA:
                cv2.imshow(
                    f"MetaDrive - {modo}",
                    cv2.resize(
                        resultado,
                        (960, 540),
                    ),
                )

                tecla = cv2.waitKey(1) & 0xFF

                if tecla == 27:
                    print(
                        "Execucao interrompida pelo usuario."
                    )
                    break

            # Evidencias visuais.
            if com_yolo and step in (1, 100, 250, 500):
                cv2.imwrite(
                    str(
                        SAIDA
                        / f"yolo_step_{step:03d}.png"
                    ),
                    resultado,
                )

            if not com_yolo and step == 100:
                cv2.imwrite(
                    str(
                        SAIDA / "sem_yolo_step_100.png"
                    ),
                    resultado,
                )

            print(
                f"{modo} | Step {step:03d}/{TOTAL_STEPS}"
                f" | FPS: {fps_atual:.1f}"
                f" | Objetos: {sum(contagens.values())}"
            )

            if terminou or truncou:
                if step < TOTAL_STEPS:
                    episodio += 1

                    # Limpa deteccoes anteriores ao reset.
                    ultima_deteccoes = []
                    ultima_contagem = {
                        "veiculos": 0,
                        "pedestres": 0,
                        "sinais": 0,
                    }

                    observacao, info = ambiente.reset(
                        seed=SEMENTE
                    )

        tempo_total = (
            time.perf_counter()
            - inicio_experimento
        )

    finally:
        ambiente.close()
        cv2.destroyAllWindows()

    quantidade = len(registros)

    fps_medio = (
        quantidade / tempo_total
        if tempo_total > 0
        else 0.0
    )

    media_inferencia = (
        soma_tempo_yolo_ms / total_inferencias
        if total_inferencias > 0
        else 0.0
    )

    return {
        "modo": modo,
        "fps": fps_medio,
        "tempo_total_s": tempo_total,
        "steps": quantidade,
        "episodios": episodio,
        "inferencias": total_inferencias,
        "tempo_medio_yolo_ms": media_inferencia,
        "registros": registros,
    }


# =========================================================
# CSV
# =========================================================

def salvar_csv(registros, nome):
    caminho = SAIDA / nome

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
        escritor.writerows(registros)

    return caminho


# =========================================================
# GRAFICO TEMPORAL
# =========================================================

def gerar_grafico(registros):
    steps = [
        linha["step"]
        for linha in registros
    ]

    veiculos = [
        linha["veiculos"]
        for linha in registros
    ]

    pedestres = [
        linha["pedestres"]
        for linha in registros
    ]

    sinais = [
        linha["sinais"]
        for linha in registros
    ]

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.plot(
        steps,
        veiculos,
        label="Veiculos",
        linewidth=1.8,
    )

    ax.plot(
        steps,
        pedestres,
        label="Pedestres",
        linewidth=1.8,
    )

    ax.plot(
        steps,
        sinais,
        label="Sinais",
        linewidth=1.8,
    )

    ax.set_title(
        "Deteccoes YOLOv8n ao longo de 500 steps"
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Quantidade de objetos detectados")

    ax.set_xlim(1, TOTAL_STEPS)
    ax.set_ylim(bottom=0)

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()
    fig.tight_layout()

    caminho = (
        SAIDA / "serie_temporal_deteccoes.png"
    )

    fig.savefig(
        caminho,
        dpi=180,
    )

    plt.close(fig)

    return caminho


# =========================================================
# COMPARACAO DE FPS
# =========================================================

def comparar_fps(sem_yolo, com_yolo):
    fps_sem = sem_yolo["fps"]
    fps_com = com_yolo["fps"]

    if fps_sem > 0:
        degradacao = (
            (fps_sem - fps_com)
            / fps_sem
        ) * 100
    else:
        degradacao = 0.0

    print("\n" + "=" * 55)
    print("COMPARACAO DE DESEMPENHO")
    print("=" * 55)

    print(
        f"FPS medio sem YOLO: {fps_sem:.2f}"
    )

    print(
        f"FPS medio com YOLO: {fps_com:.2f}"
    )

    print(
        f"Degradacao de FPS: {degradacao:.2f}%"
    )

    print(
        "Tempo medio por inferencia YOLO: "
        f"{com_yolo['tempo_medio_yolo_ms']:.2f} ms"
    )

    print(
        f"Inferencias realizadas: "
        f"{com_yolo['inferencias']}"
    )

    print(
        f"Steps sem YOLO: {sem_yolo['steps']}"
    )

    print(
        f"Steps com YOLO: {com_yolo['steps']}"
    )

    caminho = (
        SAIDA / "comparacao_fps.txt"
    )

    with caminho.open(
        "w",
        encoding="utf-8",
    ) as arquivo:
        arquivo.write(
            f"FPS sem YOLO: {fps_sem:.4f}\n"
        )

        arquivo.write(
            f"FPS com YOLO: {fps_com:.4f}\n"
        )

        arquivo.write(
            f"Degradacao percentual: "
            f"{degradacao:.4f}%\n"
        )

        arquivo.write(
            f"Tempo medio de inferencia: "
            f"{com_yolo['tempo_medio_yolo_ms']:.4f} ms\n"
        )

        arquivo.write(
            f"Quantidade de inferencias: "
            f"{com_yolo['inferencias']}\n"
        )

        arquivo.write(
            f"Intervalo YOLO: "
            f"{INTERVALO_YOLO}\n"
        )

        arquivo.write(
            f"Resolucao YOLO: "
            f"{RESOLUCAO_YOLO}\n"
        )

        arquivo.write(
            f"Steps sem YOLO: {sem_yolo['steps']}\n"
        )

        arquivo.write(
            f"Steps com YOLO: {com_yolo['steps']}\n"
        )

    return caminho


# =========================================================
# MAIN
# =========================================================

def main():
    print("=== EXERCICIO 4 - ITEM B ===")
    print("Detector YOLOv8n no MetaDrive")
    print(f"Steps por teste: {TOTAL_STEPS}")

    acoes = gerar_acoes()

    # Carrega antes da medicao, para nao incluir
    # o tempo de inicializacao do modelo no FPS.
    print("\nCarregando YOLOv8n...")
    modelo = YOLO(MODELO_YOLO)

    # Aquecimento do modelo antes do benchmark.
    imagem_teste = np.zeros(
        (ALTURA, LARGURA, 3),
        dtype=np.uint8,
    )

    modelo.predict(
        source=imagem_teste,
        imgsz=RESOLUCAO_YOLO,
        conf=CONFIANCA_MINIMA,
        verbose=False,
    )

    # Teste 1: simulacao sem YOLO.
    sem_yolo = executar_experimento(
        com_yolo=False,
        acoes=acoes,
    )

    # Teste 2: simulacao com YOLO.
    com_yolo = executar_experimento(
        com_yolo=True,
        acoes=acoes,
        modelo=modelo,
    )

    caminho_csv = salvar_csv(
        com_yolo["registros"],
        "deteccoes_500_steps.csv",
    )

    caminho_grafico = gerar_grafico(
        com_yolo["registros"]
    )

    caminho_fps = comparar_fps(
        sem_yolo,
        com_yolo,
    )

    print("\n=== ARQUIVOS GERADOS ===")
    print(f"CSV: {caminho_csv}")
    print(f"Grafico: {caminho_grafico}")
    print(f"FPS: {caminho_fps}")
    print(f"Pasta de saida: {SAIDA}")

    if (
        sem_yolo["steps"] == TOTAL_STEPS
        and com_yolo["steps"] == TOTAL_STEPS
    ):
        print("\nValidacao de 500 steps: OK")
    else:
        print("\nValidacao de 500 steps: INCOMPLETA")


if __name__ == "__main__":
    main()
