
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
SAIDA = BASE / "saidas" / "itemB"
SAIDA.mkdir(parents=True, exist_ok=True)


def normalizar(img):
    return cv2.normalize(
        np.abs(img), None, 0, 255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)


def convolucao_manual(img, kernel):
    # Convolucao matematica: inverte o kernel.
    kernel = np.flip(kernel, axis=(0, 1))
    h, w = img.shape
    kh, kw = kernel.shape

    py, px = kh // 2, kw // 2

    # BORDER_REFLECT_101 e o padrao do cv2.Sobel.
    padded = cv2.copyMakeBorder(
        img.astype(np.float64),
        py, py, px, px,
        cv2.BORDER_REFLECT_101
    )

    saida = np.zeros((h, w), dtype=np.float64)

    # Multiplicacao e soma feitas com NumPy.
    for y in range(kh):
        for x in range(kw):
            saida += (
                kernel[y, x] *
                padded[y:y + h, x:x + w]
            )

    return saida


def painel(imagens, colunas=3):
    largura, altura = 400, 280
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
            0.65, (255, 255, 255), 2
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


# 1. Carregar imagem BEV
caminho = DADOS / "pista_bev.png"
imagem = cv2.imread(str(caminho))

if imagem is None:
    raise FileNotFoundError(f"Imagem nao encontrada: {caminho}")

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# 2. Kernel Sobel X 3x3
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

# A funcao manual faz convolucao matematica.
# Por isso, invertemos o Sobel X antes da chamada
# para reproduzir a convencao de correlacao do OpenCV.
manual = convolucao_manual(cinza, np.flip(sobel_x, (0, 1)))

opencv = cv2.Sobel(
    cinza, cv2.CV_64F,
    1, 0, ksize=3,
    borderType=cv2.BORDER_REFLECT_101
)

# 3. Comparacao pixel a pixel
diferenca = np.abs(manual - opencv)

erro_maximo = float(diferenca.max())
erro_medio = float(diferenca.mean())
pixels_diferentes = int(np.count_nonzero(diferenca))

print("\nVALIDACAO SOBEL MANUAL X OPENCV")
print("Erro maximo:", erro_maximo)
print("Erro medio:", erro_medio)
print("Pixels diferentes:", pixels_diferentes)
print("Total de pixels:", cinza.size)
print("Comparacao exata:", np.array_equal(manual, opencv))

if not np.array_equal(manual, opencv):
    raise AssertionError(
        "A convolucao manual nao corresponde ao cv2.Sobel."
    )

comparacao = painel([
    ("Sobel manual", normalizar(manual)),
    ("cv2.Sobel", normalizar(opencv)),
    ("Diferenca", normalizar(diferenca))
])

cv2.imwrite(
    str(SAIDA / "comparacao_sobel.png"), comparacao
)

cv2.imshow("Validacao Sobel", comparacao)

# 4. Banco de filtros direcionais
# Os angulos representam a direcao do gradiente.
sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float64)

# Gradientes X e Y calculados com OpenCV
gx = cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(cinza, cv2.CV_64F, 0, 1, ksize=3)

# Direcoes equivalentes a kernels orientados
angulos = [0, 45, 90, 135]
respostas = {}

for angulo in angulos:
    rad = np.deg2rad(angulo)

    # Resposta direcional do gradiente
    resposta = np.abs(
        np.cos(rad) * gx +
        np.sin(rad) * gy
    )

    respostas[angulo] = resposta

    cv2.imwrite(
        str(SAIDA / f"direcional_{angulo}.png"),
        normalizar(resposta)
    )

# 5. Selecionar regiao das marcacoes
# A mascara utiliza o contraste das faixas em HLS.
hls = cv2.cvtColor(imagem, cv2.COLOR_BGR2HLS)
l = hls[:, :, 1]
s = hls[:, :, 2]

amarelo = (s >= 90) & (l >= 75)
branco = (l >= 165) & (s < 95)

mascara = np.uint8((amarelo | branco) * 255)

# Usa bordas das marcacoes para avaliar os gradientes.
bordas_marcacoes = cv2.morphologyEx(
    mascara, cv2.MORPH_GRADIENT,
    np.ones((3, 3), np.uint8)
) > 0

if not np.any(bordas_marcacoes):
    raise RuntimeError(
        "Nenhuma borda de marcacao foi encontrada."
    )

# 6. Calcular resposta media de cada angulo
medias = {}

print("\nBANCO DE FILTROS DIRECIONAIS")
print("-" * 42)

for angulo, resposta in respostas.items():
    media = float(resposta[bordas_marcacoes].mean())
    medias[angulo] = media

    print(f"Angulo {angulo:>3} graus: {media:.3f}")

melhor_angulo = max(medias, key=medias.get)

print("\nAngulo de maior resposta:", melhor_angulo, "graus")
print(f"Resposta media maxima: {medias[melhor_angulo]:.3f}")

# 7. Painel das quatro respostas direcionais
painel_direcional = painel([
    (f"Filtro {angulo} graus", normalizar(respostas[angulo]))
    for angulo in angulos
], colunas=2)

cv2.imwrite(
    str(SAIDA / "comparacao_direcional.png"),
    painel_direcional
)

cv2.imwrite(
    str(SAIDA / "mascara_marcacoes.png"),
    mascara
)

cv2.imshow("Filtros direcionais", painel_direcional)

# 8. Relacao com CNNs
# Cada filtro produz um mapa de caracteristicas.
# Empilhando quatro mapas, obtemos um tensor H x W x 4.
# Em uma CNN, os coeficientes dos filtros sao aprendidos
# durante o treinamento, em vez de definidos manualmente.
features = np.stack(
    [respostas[a] for a in angulos],
    axis=-1
).astype(np.float32)

print("\nRELACAO COM CNN")
print("Entrada H x W x 1:", cinza.shape + (1,))
print("Saida H x W x 4:", features.shape)

np.save(SAIDA / "mapas_direcionais.npy", features)

print("\nResultados salvos em:", SAIDA)

cv2.waitKey(0)
cv2.destroyAllWindows()
