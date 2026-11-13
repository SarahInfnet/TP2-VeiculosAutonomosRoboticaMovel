from pathlib import Path
import csv
import time

import cv2
import numpy as np


BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
SAIDA = BASE / "saidas" / "itemB"
SAIDA.mkdir(parents=True, exist_ok=True)

N_PONTOS = 200

PARAMETROS_LK = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        30, 0.01
    )
}

def detectar_harris(cinza):
    inicio = time.perf_counter()

    resposta = cv2.cornerHarris(
        np.float32(cinza),
        blockSize=2,
        ksize=3,
        k=0.04
    )

    # Seleciona maximos locais, evitando varios
    # pixels praticamente iguais no mesmo canto.
    maximos = cv2.dilate(
        resposta, np.ones((3, 3), np.uint8)
    )

    mascara = (
        (resposta == maximos) &
        (resposta > 0.01 * resposta.max())
    )

    ys, xs = np.where(mascara)

    if len(xs) == 0:
        pontos = np.empty((0, 2), np.float32)
    else:
        ordem = np.argsort(
            resposta[ys, xs]
        )[::-1]

        selecionados = []

        for indice in ordem:
            x = int(xs[indice])
            y = int(ys[indice])

            # Distancia minima entre keypoints.
            if any(
                (x - px) ** 2 + (y - py) ** 2 < 8 ** 2
                for px, py in selecionados
            ):
                continue

            selecionados.append((x, y))

            if len(selecionados) == N_PONTOS:
                break

        pontos = np.array(
            selecionados, dtype=np.float32
        ).reshape(-1, 2)

    tempo_ms = (
        time.perf_counter() - inicio
    ) * 1000

    return pontos, tempo_ms


def detectar_shi_tomasi(cinza):
    inicio = time.perf_counter()

    pontos = cv2.goodFeaturesToTrack(
        cinza,
        maxCorners=N_PONTOS,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7
    )

    tempo_ms = (
        time.perf_counter() - inicio
    ) * 1000

    if pontos is None:
        pontos = np.empty(
            (0, 1, 2), np.float32
        )

    return pontos, tempo_ms


def desenhar_pontos(imagem, pontos, cor):
    visual = imagem.copy()

    for x, y in pontos.reshape(-1, 2):
        cv2.circle(
            visual,
            (round(float(x)), round(float(y))),
            3, cor, -1, cv2.LINE_AA
        )

    return visual


def comparar_detectores():
    arquivos = sorted(
        caminho for caminho in DADOS.iterdir()
        if caminho.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    )

    if not arquivos:
        raise RuntimeError(
            "Nenhuma imagem encontrada em Ex3/dados."
        )

    registros = []

    print("\nCOMPARACAO HARRIS X SHI-TOMASI")
    print("-" * 65)

    for caminho in arquivos:
        imagem = cv2.imread(str(caminho))

        if imagem is None:
            continue

        cinza = cv2.cvtColor(
            imagem, cv2.COLOR_BGR2GRAY
        )

        harris, tempo_harris = detectar_harris(cinza)
        shi, tempo_shi = detectar_shi_tomasi(cinza)

        print(f"\nImagem: {caminho.name}")
        print(
            f"Harris: {len(harris)} pontos | "
            f"{tempo_harris:.3f} ms"
        )
        print(
            f"Shi-Tomasi: {len(shi)} pontos | "
            f"{tempo_shi:.3f} ms"
        )

        visual_harris = desenhar_pontos(
            imagem, harris, (0, 0, 255)
        )

        visual_shi = desenhar_pontos(
            imagem, shi, (0, 255, 0)
        )

        cv2.imwrite(
            str(SAIDA / f"{caminho.stem}_harris.png"),
            visual_harris
        )

        cv2.imwrite(
            str(SAIDA / f"{caminho.stem}_shi_tomasi.png"),
            visual_shi
        )

        painel = np.hstack([
            cv2.resize(visual_harris, (640, 360)),
            cv2.resize(visual_shi, (640, 360))
        ])

        cv2.putText(
            painel, "Harris",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 2
        )

        cv2.putText(
            painel, "Shi-Tomasi",
            (655, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 2
        )

        cv2.imwrite(
            str(SAIDA / f"{caminho.stem}_comparacao.png"),
            painel
        )

        cv2.imshow(
            f"Keypoints - {caminho.name}", painel
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        registros.append([
            caminho.name,
            len(harris),
            round(tempo_harris, 3),
            len(shi),
            round(tempo_shi, 3)
        ])

    with open(
        SAIDA / "comparacao_keypoints.csv",
        "w", newline="", encoding="utf-8"
    ) as arquivo:
        escritor = csv.writer(arquivo)

        escritor.writerow([
            "imagem",
            "harris_pontos",
            "harris_ms",
            "shi_pontos",
            "shi_ms"
        ])

        escritor.writerows(registros)

    if registros:
        print("\nMEDIAS DOS TEMPOS")
        print(
            "Harris:",
            round(np.mean([r[2] for r in registros]), 3),
            "ms"
        )
        print(
            "Shi-Tomasi:",
            round(np.mean([r[4] for r in registros]), 3),
            "ms"
        )


def desenhar_fluxo(imagem, antigos, novos):
    visual = imagem.copy()

    for (x, y), (u, v) in zip(antigos, novos):
        origem = (
            round(float(x)), round(float(y))
        )
        destino = (
            round(float(u)), round(float(v))
        )

        cv2.arrowedLine(
            visual, origem, destino,
            (0, 255, 255), 2,
            cv2.LINE_AA, tipLength=0.3
        )

        cv2.circle(
            visual, destino,
            3, (0, 0, 255), -1
        )

    return visual


def rastrear_fluxo_optico():
    caminho = DADOS / "trafego.mp4"

    captura = cv2.VideoCapture(str(caminho))

    if not captura.isOpened():
        raise RuntimeError(
            f"Nao foi possivel abrir: {caminho}"
        )

    fps = captura.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0
        print(
            "FPS nao informado; usando 30 apenas "
            "para salvar o video."
        )

    sucesso, anterior = captura.read()

    if not sucesso:
        captura.release()
        raise RuntimeError(
            "Nao foi possivel ler o primeiro frame."
        )

    altura, largura = anterior.shape[:2]

    gravador = cv2.VideoWriter(
        str(SAIDA / "fluxo_optico.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (largura, altura)
    )

    if not gravador.isOpened():
        captura.release()
        raise RuntimeError(
            "Nao foi possivel criar o video de saida."
        )

    cinza_anterior = cv2.cvtColor(
        anterior, cv2.COLOR_BGR2GRAY
    )

    # Pontos Shi-Tomasi usados como entrada do LK.
    pontos, _ = detectar_shi_tomasi(
        cinza_anterior
    )

    registros = []
    numero_frame = 0

    print("\nRASTREAMENTO LUCAS-KANADE")
    print("-" * 65)

    while True:
        sucesso, atual = captura.read()

        if not sucesso:
            break

        numero_frame += 1

        cinza_atual = cv2.cvtColor(
            atual, cv2.COLOR_BGR2GRAY
        )

        if len(pontos) == 0:
            pontos, _ = detectar_shi_tomasi(
                cinza_anterior
            )

        antigos = np.empty((0, 2), np.float32)
        novos = np.empty((0, 2), np.float32)
        tempo_ms = 0.0

        if len(pontos) > 0:
            inicio = time.perf_counter()

            proximos, status, erro = (
                cv2.calcOpticalFlowPyrLK(
                    cinza_anterior,
                    cinza_atual,
                    pontos,
                    None,
                    **PARAMETROS_LK
                )
            )

            tempo_ms = (
                time.perf_counter() - inicio
            ) * 1000

            if proximos is not None and status is not None:
                validos = status.reshape(-1) == 1

                antigos = pontos.reshape(
                    -1, 2
                )[validos]

                novos = proximos.reshape(
                    -1, 2
                )[validos]

        visual = desenhar_fluxo(
            atual, antigos, novos
        )

        if len(novos) > 0:
            deslocamentos = np.linalg.norm(
                novos - antigos, axis=1
            )

            media_pixels = float(
                deslocamentos.mean()
            )

            # Velocidade aparente em pixels/s.
            # Nao equivale diretamente a m/s.
            velocidade_aparente = (
                media_pixels * fps
            )
        else:
            media_pixels = 0.0
            velocidade_aparente = 0.0

        cv2.putText(
            visual,
            f"Frame: {numero_frame} | "
            f"Vetores: {len(novos)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        print(
            f"Frame {numero_frame:04d}: "
            f"vetores={len(novos):3d} | "
            f"deslocamento={media_pixels:.2f} px | "
            f"velocidade_aparente="
            f"{velocidade_aparente:.2f} px/s | "
            f"LK={tempo_ms:.3f} ms"
        )

        registros.append([
            numero_frame,
            len(novos),
            round(media_pixels, 3),
            round(velocidade_aparente, 3),
            round(tempo_ms, 3)
        ])

        gravador.write(visual)

        # Redimensiona 
        altura, largura = visual.shape[:2]

        escala = min(960 / largura, 540 / altura, 1.0)

        nova_largura = int(largura * escala)
        nova_altura = int(altura * escala)

        visual_reduzido = cv2.resize(
            visual,
            (nova_largura, nova_altura),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow("Fluxo optico Lucas-Kanade", visual_reduzido)

        # Pressione Q para interromper.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cinza_anterior = cinza_atual

        # Mantem os pontos rastreados.
        pontos = novos.reshape(
            -1, 1, 2
        ).astype(np.float32)

        # Reabastece os keypoints quando poucos restarem.
        if len(pontos) < 60:
            pontos, _ = detectar_shi_tomasi(
                cinza_anterior
            )

    captura.release()
    gravador.release()
    cv2.destroyAllWindows()

    with open(
        SAIDA / "resultados_fluxo.csv",
        "w", newline="", encoding="utf-8"
    ) as arquivo:
        escritor = csv.writer(arquivo)

        escritor.writerow([
            "frame",
            "vetores_validos",
            "deslocamento_medio_px",
            "velocidade_aparente_px_s",
            "tempo_lk_ms"
        ])

        escritor.writerows(registros)

    print("\nRESUMO DO FLUXO OPTICO")
    print("Frames processados:", numero_frame)
    print(
        "Video salvo em:",
        SAIDA / "fluxo_optico.mp4"
    )

# Discussao de velocidade relativa:

# O fluxo optico permite acompanhar o movimento de pontos
# da imagem entre um frame e outro. Com o FPS, conseguimos
# calcular a velocidade aparente desses pontos em pixels
# por segundo (px/s).

# Mas, para saber a velocidade de um obstaculo em metros
# por segundo (m/s), precisamos conhecer a calibracao da
# camera, a distancia do objeto e considerar que o proprio
# veiculo tambem pode estar se movimentando.

# Por isso, o fluxo optico sozinho nao e suficiente
# para descobrir a velocidade real de um obstaculo.


def main():
    comparar_detectores()
    rastrear_fluxo_optico()

    print("\nEXERCICIO FINALIZADO")
    print("Resultados em:", SAIDA)


if __name__ == "__main__":
    main()
