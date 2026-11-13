from pathlib import Path
import time
import csv
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
SAIDA = BASE / "saidas" / "itemA"
SAIDA.mkdir(parents=True, exist_ok=True)


def ler(nome="pista_bev.png"):
    caminho = DADOS / nome
    img = cv2.imread(str(caminho))

    if img is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {caminho}")

    return img


# Cenários adaptados dos exemplos da Aula 5
def sombra(img):
    a = img.astype(np.float32).copy()
    h, w = img.shape[:2]

    mascara = np.zeros((h, w), np.uint8)
    pontos = np.array([
        [0, int(h * .16)],
        [w, int(h * .36)],
        [w, int(h * .59)],
        [0, int(h * .39)]
    ], np.int32)

    cv2.fillConvexPoly(mascara, pontos, 255)
    a[mascara > 0] *= .38

    ruido = np.random.default_rng(41).normal(0, 2.5, a.shape)
    return np.clip(a + ruido, 0, 255).astype(np.uint8)


def molhado(img):
    # Simulacao de reflexos, nao asfalto molhado real.
    h, w = img.shape[:2]
    a = (img.astype(np.float32) * .68).astype(np.uint8)
    overlay = np.zeros_like(a)

    cv2.ellipse(
        overlay,
        (int(w * .50), int(h * .73)),
        (int(w * .39), int(h * .10)),
        0, 0, 360, (175, 181, 189), -1
    )

    cv2.line(
        overlay,
        (int(w * .27), int(h * .57)),
        (int(w * .76), int(h * .89)),
        (255, 235, 226), 6
    )

    cv2.line(
        overlay,
        (int(w * .47), int(h * .62)),
        (int(w * .66), int(h * .97)),
        (225, 224, 240), 4
    )

    a = cv2.addWeighted(a, .83, overlay, .17, 0)
    ruido = np.random.default_rng(27).normal(0, 6, a.shape)

    return np.clip(
        a.astype(np.float32) + ruido, 0, 255
    ).astype(np.uint8)


# Operadores de borda
def sobel(g):
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)

    return np.uint8(magnitude >= 95) * 255


def log(g):
    suave = cv2.GaussianBlur(g, (5, 5), 1.2)
    lap = cv2.Laplacian(suave, cv2.CV_64F)

    return np.uint8(np.abs(lap) >= 12) * 255


def canny_baixo(g):
    return cv2.Canny(g, 50, 110)


def canny_alto(g):
    return cv2.Canny(g, 100, 200)


def dog(g):
    g1 = cv2.GaussianBlur(g, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(g, (0, 0), 2.2)

    diferenca = cv2.absdiff(g1, g2)
    return np.uint8(diferenca >= 8) * 255


OPERADORES = {
    "Sobel": sobel,
    "LoG": log,
    "Canny 50/110": canny_baixo,
    "Canny 100/200": canny_alto,
    "DoG": dog
}


def medir_tempo(funcao, imagem, repeticoes=30):
    # Aquecimento para reduzir o efeito da primeira chamada
    for _ in range(3):
        funcao(imagem)

    tempos = []

    for _ in range(repeticoes):
        inicio = time.perf_counter()
        funcao(imagem)
        tempos.append((time.perf_counter() - inicio) * 1000)

    return float(np.median(tempos))


def criar_painel(imagens, colunas=3):
    largura, altura = 400, 260
    quadros = []

    for titulo, img in imagens:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        quadro = cv2.resize(img, (largura, altura))
        cv2.rectangle(
            quadro, (0, 0), (largura, 35),
            (30, 30, 30), -1
        )
        cv2.putText(
            quadro, titulo, (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            .65, (255, 255, 255), 2
        )
        quadros.append(quadro)

    while len(quadros) % colunas:
        quadros.append(
            np.zeros((altura, largura, 3), np.uint8)
        )

    linhas = [
        np.hstack(quadros[i:i + colunas])
        for i in range(0, len(quadros), colunas)
    ]

    return np.vstack(linhas)


def executar():
    original = ler()

    cenarios = {
        "Normal": original,
        "Sombra": sombra(original),
        "Reflexos": molhado(original)
    }

    resultados = []

    print("\nCOMPARACAO DE OPERADORES DE BORDA")
    print("-" * 76)
    print(
        f"{'Cenario':<12} {'Operador':<18} "
        f"{'Pixels ativos':>15} {'Tempo (ms)':>14}"
    )
    print("-" * 76)

    for nome_cenario, imagem in cenarios.items():
        cinza = cv2.cvtColor(
            imagem, cv2.COLOR_BGR2GRAY
        )

        painel = [(nome_cenario, imagem)]

        for nome_operador, funcao in OPERADORES.items():
            bordas = funcao(cinza)

            pixels = int(cv2.countNonZero(bordas))
            tempo_ms = medir_tempo(funcao, cinza)

            resultados.append({
                "cenario": nome_cenario,
                "operador": nome_operador,
                "pixels_ativos": pixels,
                "tempo_ms": round(tempo_ms, 3)
            })

            print(
                f"{nome_cenario:<12} {nome_operador:<18} "
                f"{pixels:>15} {tempo_ms:>14.3f}"
            )

            nome_arquivo = (
                nome_operador.lower()
                .replace(" ", "_")
                .replace("/", "_")
            )

            cv2.imwrite(
                str(SAIDA / f"{nome_cenario.lower()}_{nome_arquivo}.png"),
                bordas
            )

            painel.append((nome_operador, bordas))

        comparacao = criar_painel(painel)

        cv2.imwrite(
            str(SAIDA / f"comparacao_{nome_cenario.lower()}.png"),
            comparacao
        )

        cv2.imshow(
            f"Operadores - {nome_cenario}", comparacao
        )

    # Salvar tabela em CSV
    with open(
        SAIDA / "comparacao_operadores.csv",
        "w", newline="", encoding="utf-8"
    ) as arquivo:
        escritor = csv.DictWriter(
            arquivo,
            fieldnames=[
                "cenario", "operador",
                "pixels_ativos", "tempo_ms"
            ]
        )
        escritor.writeheader()
        escritor.writerows(resultados)

    print("\nTabela salva em:", SAIDA / "comparacao_operadores.csv")

    print("\nCONCLUSAO:")
    print("Compare a continuidade das faixas e as bordas falsas.")
    print("Canny e um candidato, mas a melhor configuracao")
    print("deve ser escolhida apos analisar as imagens geradas.")
    print("Sombra e reflexos utilizados sao simulados.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    executar()
