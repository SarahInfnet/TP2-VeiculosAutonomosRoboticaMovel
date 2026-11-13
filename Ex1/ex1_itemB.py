from pathlib import Path
import cv2
import numpy as np

# Pastas do exercicio
BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
SAIDA = BASE / "saidas" / "itemB"
SAIDA.mkdir(parents=True, exist_ok=True)


def ler(nome):
    caminho = DADOS / nome
    imagem = cv2.imread(str(caminho))

    if imagem is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {caminho}")

    return imagem


def binarizar(img):
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return cv2.adaptiveThreshold(
        cinza,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -12
    )


def morfologia(mascara, forma, tamanho=(5, 7)):
    kernel = cv2.getStructuringElement(forma, tamanho)

    # Opening: erosao seguida de dilatacao
    abertura = cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, kernel
    )

    # Closing: dilatacao seguida de erosao
    fechamento = cv2.morphologyEx(
        abertura, cv2.MORPH_CLOSE, kernel
    )

    return abertura, fechamento, kernel


def selecionar_componentes(img, mascara,
                           area_min=100, razao_min=1.5):

    n, rotulos, stats, centros = (
        cv2.connectedComponentsWithStats(mascara, 8)
    )

    resultado = img.copy()
    aceitos = []

    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        razao = h / max(w, 1)

        if area >= area_min and razao >= razao_min:
            aceitos.append((x, y, w, h, area, razao))

            cv2.rectangle(
                resultado,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    return resultado, aceitos


def criar_painel(imagens):
    quadros = []

    for titulo, img in imagens:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        quadro = cv2.resize(img, (400, 300))

        cv2.rectangle(
            quadro, (0, 0), (400, 35), (30, 30, 30), -1
        )

        cv2.putText(
            quadro, titulo, (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2
        )

        quadros.append(quadro)

    return np.hstack(quadros)


# 1. Carregar imagem BEV
bev = ler("bev_marcacoes.png")

# 2. Aplicar threshold adaptativo
mascara = binarizar(bev)

cv2.imwrite(str(SAIDA / "threshold_adaptativo.png"), mascara)

# 3. Comparar tres formatos de kernel
formas = [
    ("Retangular", cv2.MORPH_RECT),
    ("Eliptico", cv2.MORPH_ELLIPSE),
    ("Cruz", cv2.MORPH_CROSS)
]

comparacao = [("BEV original", bev), ("Threshold", mascara)]

for nome, forma in formas:

    abertura, fechamento, kernel = morfologia(
        mascara, forma, (5, 7)
    )

    # 4. Extrair componentes conectados e filtrar
    resultado, aceitos = selecionar_componentes(
        bev, fechamento,
        area_min=100,
        razao_min=1.5
    )

    # 5. Calcular estatisticas
    quantidade = len(aceitos)

    print(f"\n--- Kernel {nome} ---")
    print("Matriz do kernel:\n", kernel)
    print("Marcacoes detectadas:", quantidade)

    if quantidade > 0:
        largura_media = np.mean([v[2] for v in aceitos])
        altura_media = np.mean([v[3] for v in aceitos])
        area_media = np.mean([v[4] for v in aceitos])

        print(f"Largura media: {largura_media:.2f} pixels")
        print(f"Altura media: {altura_media:.2f} pixels")
        print(f"Area media: {area_media:.2f} pixels quadrados")

    else:
        print("Nenhuma marcacao passou pelos filtros.")

    # 6. Salvar resultados de cada kernel
    nome_arquivo = nome.lower()

    cv2.imwrite(
        str(SAIDA / f"{nome_arquivo}_opening.png"),
        abertura
    )

    cv2.imwrite(
        str(SAIDA / f"{nome_arquivo}_closing.png"),
        fechamento
    )

    cv2.imwrite(
        str(SAIDA / f"{nome_arquivo}_componentes.png"),
        resultado
    )

    # Painel individual: opening, closing e deteccao
    painel = criar_painel([
        (f"{nome} - Opening", abertura),
        (f"{nome} - Closing", fechamento),
        (f"{nome} - Componentes", resultado)
    ])

    cv2.imwrite(
        str(SAIDA / f"{nome_arquivo}_painel.png"),
        painel
    )

    comparacao.append((nome, fechamento))

    cv2.imshow(f"Kernel {nome}", painel)

# 7. Comparacao visual dos tres kernels
painel_comparacao = criar_painel(comparacao)

cv2.imwrite(
    str(SAIDA / "comparacao_kernels.png"),
    painel_comparacao
)

cv2.imshow("Comparacao dos kernels", painel_comparacao)

print("\nResultados salvos em:", SAIDA)

cv2.waitKey(0)
cv2.destroyAllWindows()
