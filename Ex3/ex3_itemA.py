from pathlib import Path
import csv
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
SAIDA = BASE / "saidas" / "itemA"
SAIDA.mkdir(parents=True, exist_ok=True)

CONFIGURACOES = {
    "sensivel": {
        "dp": 1.0, "minDist": 25,
        "param1": 100, "param2": 20
    },
    "intermediaria": {
        "dp": 1.2, "minDist": 35,
        "param1": 135, "param2": 32
    },
    "conservadora": {
        "dp": 1.5, "minDist": 50,
        "param1": 160, "param2": 45
    }
}

RAIO_MIN = 8
RAIO_MAX = 100

# Limites iniciais para as imagens utilizadas.
MIN_SUPORTE_BORDA = 0.35
MIN_SUPORTE_VERMELHO = 0.30


def preparar_imagem(imagem):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cinza, (7, 7), 1.5)


def mascara_vermelha(imagem):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    vermelho_1 = cv2.inRange(
        hsv, (0, 75, 65), (12, 255, 255)
    )
    vermelho_2 = cv2.inRange(
        hsv, (170, 75, 65), (179, 255, 255)
    )

    return cv2.bitwise_or(vermelho_1, vermelho_2)


def pontos_circunferencia(x, y, raio, altura, largura):
    angulos = np.linspace(
        0, 2 * np.pi, 180, endpoint=False
    )

    xs = np.rint(
        x + raio * np.cos(angulos)
    ).astype(int)

    ys = np.rint(
        y + raio * np.sin(angulos)
    ).astype(int)

    validos = (
        (xs >= 0) & (xs < largura) &
        (ys >= 0) & (ys < altura)
    )

    return xs[validos], ys[validos]


def suporte_circular(mascara, x, y, raio):
    altura, largura = mascara.shape

    xs, ys = pontos_circunferencia(
        x, y, raio, altura, largura
    )

    if len(xs) == 0:
        return 0.0

    return float(np.mean(mascara[ys, xs] > 0))


def calcular_confianca(bordas, vermelho, x, y, raio):
    # Tolera pequenas diferencas entre a borda detectada e a circunferencia estimada.
    bordas_dilatadas = cv2.dilate(
        bordas, np.ones((3, 3), np.uint8)
    )

    vermelho_dilatado = cv2.dilate(
        vermelho, np.ones((5, 5), np.uint8)
    )

    suporte_borda = suporte_circular(
        bordas_dilatadas, x, y, raio
    )

    # A borda vermelha pode estar ligeiramente para dentro ou para fora do raio encontrado.
    suporte_vermelho = max(
        suporte_circular(
            vermelho_dilatado, x, y, max(1, raio - 2)
        ),
        suporte_circular(
            vermelho_dilatado, x, y, raio
        ),
        suporte_circular(
            vermelho_dilatado, x, y, raio + 2
        )
    )

    confianca = (
        0.5 * suporte_borda +
        0.5 * suporte_vermelho
    )

    return (
        suporte_borda,
        suporte_vermelho,
        confianca
    )


def detectar_circulos(cinza, configuracao):
    encontrados = cv2.HoughCircles(
        cinza,
        cv2.HOUGH_GRADIENT,
        dp=configuracao["dp"],
        minDist=configuracao["minDist"],
        param1=configuracao["param1"],
        param2=configuracao["param2"],
        minRadius=RAIO_MIN,
        maxRadius=RAIO_MAX
    )

    if encontrados is None:
        return []

    return np.rint(
        encontrados[0]
    ).astype(int).tolist()


def filtrar_circulos(circulos, bordas, vermelho):
    aprovados = []

    for x, y, raio in circulos:
        suporte_borda, suporte_vermelho, confianca = (
            calcular_confianca(
                bordas, vermelho, x, y, raio
            )
        )

        if suporte_borda < MIN_SUPORTE_BORDA:
            continue

        if suporte_vermelho < MIN_SUPORTE_VERMELHO:
            continue

        aprovados.append({
            "x": x,
            "y": y,
            "raio": raio,
            "confianca": confianca,
            "suporte_borda": suporte_borda,
            "suporte_vermelho": suporte_vermelho
        })

    # Remove circulos muito semelhantes que representam o mesmo objeto.
    aprovados.sort(
        key=lambda item: item["confianca"],
        reverse=True
    )

    resultado = []

    for candidato in aprovados:
        duplicado = False

        for existente in resultado:
            distancia = np.hypot(
                candidato["x"] - existente["x"],
                candidato["y"] - existente["y"]
            )

            raio_referencia = min(
                candidato["raio"],
                existente["raio"]
            )

            if (
                distancia < 0.45 * raio_referencia
                and abs(
                    candidato["raio"] - existente["raio"]
                ) < 0.35 * max(
                    candidato["raio"],
                    existente["raio"]
                )
            ):
                duplicado = True
                break

        if not duplicado:
            resultado.append(candidato)

    return resultado


def detectar_linhas(imagem):
    altura, largura = imagem.shape[:2]

    cinza = cv2.cvtColor(
        imagem, cv2.COLOR_BGR2GRAY
    )

    suave = cv2.GaussianBlur(
        cinza, (5, 5), 0
    )

    bordas = cv2.Canny(
        suave, 60, 150
    )

    # Regiao de interesse: parte inferior da cena.
    mascara = np.zeros_like(bordas)

    poligono = np.array([[
        (0, altura - 1),
        (0, int(altura * 0.55)),
        (largura, int(altura * 0.55)),
        (largura - 1, altura - 1)
    ]], dtype=np.int32)

    cv2.fillPoly(mascara, poligono, 255)

    bordas_roi = cv2.bitwise_and(
        bordas, mascara
    )

    linhas = cv2.HoughLinesP(
        bordas_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=45,
        maxLineGap=30
    )

    if linhas is None:
        return []

    resultado = []

    for linha in linhas.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, linha)

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        comprimento = np.hypot(dx, dy)

        if comprimento < 45:
            continue

        # Descarta segmentos quase horizontais.
        if dy < 0.35 * max(dx, 1):
            continue

        resultado.append((x1, y1, x2, y2))

    return resultado


def desenhar_resultado(imagem, circulos, linhas):
    visual = imagem.copy()

    # Azul: segmentos candidatos a faixas.
    for x1, y1, x2, y2 in linhas:
        cv2.line(
            visual,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    # Verde: circulos aprovados pelos filtros.
    for indice, item in enumerate(circulos, 1):
        x = item["x"]
        y = item["y"]
        raio = item["raio"]
        confianca = item["confianca"]

        cv2.circle(
            visual, (x, y), raio,
            (0, 255, 0), 2, cv2.LINE_AA
        )

        cv2.circle(
            visual, (x, y), 3,
            (0, 255, 255), -1
        )

        cv2.putText(
            visual,
            f"{indice}: {confianca:.2f}",
            (max(0, x - raio), max(20, y - raio - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return visual


def montar_painel(imagens):
    quadros = []

    for titulo, imagem in imagens:
        quadro = cv2.resize(
            imagem, (480, 320)
        )

        cv2.rectangle(
            quadro,
            (0, 0), (480, 36),
            (25, 25, 25), -1
        )

        cv2.putText(
            quadro, titulo, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        quadros.append(quadro)

    return np.vstack([
        np.hstack(quadros[:2]),
        np.hstack(quadros[2:])
    ])


def main():
    extensoes = {
        ".jpg", ".jpeg", ".png", ".bmp"
    }

    arquivos = sorted(
        caminho for caminho in DADOS.iterdir()
        if caminho.suffix.lower() in extensoes
    ) if DADOS.exists() else []

    if len(arquivos) < 5:
        raise RuntimeError(
            "Coloque pelo menos cinco imagens "
            f"na pasta {DADOS}"
        )

    registros = []
    total_processadas = 0
    total_bruto = 0
    total_filtrado = 0

    for caminho in arquivos:
        imagem = cv2.imread(str(caminho))

        if imagem is None:
            print(
                f"Imagem ignorada: {caminho.name}"
            )
            continue

        total_processadas += 1

        cinza = preparar_imagem(imagem)

        bordas = cv2.Canny(
            cinza, 80, 160
        )

        vermelho = mascara_vermelha(imagem)
        linhas = detectar_linhas(imagem)

        print(f"\nIMAGEM: {caminho.name}")
        print(
            f"Linhas detectadas: {len(linhas)}"
        )

        imagens_painel = [
            ("Original", imagem.copy())
        ]

        for nome, config in CONFIGURACOES.items():
            circulos_brutos = detectar_circulos(
                cinza, config
            )

            circulos = filtrar_circulos(
                circulos_brutos,
                bordas,
                vermelho
            )

            total_bruto += len(circulos_brutos)
            total_filtrado += len(circulos)

            visual = desenhar_resultado(
                imagem, circulos, linhas
            )

            print(f"\nConfiguracao: {nome}")
            print(
                f"dp={config['dp']}, "
                f"minDist={config['minDist']}, "
                f"param1={config['param1']}, "
                f"param2={config['param2']}"
            )

            print(
                "Circulos candidatos:",
                len(circulos_brutos)
            )

            print(
                "Circulos apos filtragem:",
                len(circulos)
            )

            for indice, item in enumerate(
                circulos, 1
            ):
                print(
                    f"  Circulo {indice}: "
                    f"centro=({item['x']},{item['y']}), "
                    f"raio={item['raio']}, "
                    f"confianca_estimada="
                    f"{item['confianca']:.3f}, "
                    f"suporte_borda="
                    f"{item['suporte_borda']:.3f}, "
                    f"suporte_vermelho="
                    f"{item['suporte_vermelho']:.3f}"
                )

                registros.append([
                    caminho.name,
                    nome,
                    config["dp"],
                    config["minDist"],
                    config["param1"],
                    config["param2"],
                    indice,
                    item["x"],
                    item["y"],
                    item["raio"],
                    round(item["confianca"], 4),
                    round(item["suporte_borda"], 4),
                    round(item["suporte_vermelho"], 4)
                ])

            cv2.imwrite(
                str(
                    SAIDA /
                    f"{caminho.stem}_{nome}.png"
                ),
                visual
            )

            imagens_painel.append(
                (nome, visual)
            )

        painel = montar_painel(
            imagens_painel
        )

        cv2.imwrite(
            str(
                SAIDA /
                f"{caminho.stem}_comparacao.png"
            ),
            painel
        )

        cv2.imshow(
            f"Comparacao - {caminho.name}",
            painel
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    with open(
        SAIDA / "resultados_circulos.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:
        escritor = csv.writer(arquivo)

        escritor.writerow([
            "imagem",
            "configuracao",
            "dp",
            "minDist",
            "param1",
            "param2",
            "circulo",
            "centro_x",
            "centro_y",
            "raio",
            "confianca_estimada",
            "suporte_borda",
            "suporte_vermelho"
        ])

        escritor.writerows(registros)

    print("\nRESUMO FINAL")
    print(
        "Imagens processadas:",
        total_processadas
    )
    print(
        "Configuracoes por imagem:",
        len(CONFIGURACOES)
    )
    print(
        "Circulos candidatos:",
        total_bruto
    )
    print(
        "Circulos apos filtragem:",
        total_filtrado
    )
    print(
        "Resultados salvos em:",
        SAIDA
    )


if __name__ == "__main__":
    main()
