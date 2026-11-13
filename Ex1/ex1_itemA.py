
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
SAIDA = BASE / "saidas"
SAIDA.mkdir(exist_ok=True)

# 1. Carregar duas imagens sobrepostas
img1 = cv2.imread(str(DADOS / "visao_01.png"))
img2 = cv2.imread(str(DADOS / "visao_02.png"))

if img1 is None or img2 is None:
    raise FileNotFoundError(
        "Verifique visao_01.png e visao_02.png na pasta dados."
    )

# 2. Detectar keypoints e descritores com ORB
orb = cv2.ORB_create(nfeatures=2500, fastThreshold=12)

kp1, des1 = orb.detectAndCompute(
    cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None
)
kp2, des2 = orb.detectAndCompute(
    cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None
)

if des1 is None or des2 is None:
    raise RuntimeError("ORB nao encontrou descritores suficientes.")

# 3. Matching com BFMatcher e teste da razao de Lowe
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

pares = bf.knnMatch(des2, des1, k=2)
bons = [
    m for par in pares if len(par) == 2
    for m, n in [par] if m.distance < 0.80 * n.distance
]

if len(bons) < 4:
    raise RuntimeError("Sao necessarios pelo menos 4 matches.")

# 4. Calcular homografia usando RANSAC
src = np.float32(
    [kp2[m.queryIdx].pt for m in bons]
).reshape(-1, 1, 2)

dst = np.float32(
    [kp1[m.trainIdx].pt for m in bons]
).reshape(-1, 1, 2)

H, mascara = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)

if H is None or mascara is None:
    raise RuntimeError("Nao foi possivel calcular a homografia.")

inliers = mascara.ravel().astype(bool)

# 5. Calcular erro medio de reprojecao dos inliers
projetados = cv2.perspectiveTransform(src, H)
erros = np.linalg.norm(
    (projetados - dst).reshape(-1, 2), axis=1
)

erro_medio = float(np.mean(erros[inliers]))

print("Keypoints imagem 1:", len(kp1))
print("Keypoints imagem 2:", len(kp2))
print("Matches apos Lowe:", len(bons))
print("Inliers RANSAC:", int(np.sum(inliers)))
print(f"Erro medio de reprojecao: {erro_medio:.3f} pixels")
print("Matriz de homografia:\n", H)

# 6. Determinar dimensoes do mosaico panoramico
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

cantos1 = np.float32([
    [0, 0], [w1, 0], [w1, h1], [0, h1]
]).reshape(-1, 1, 2)

cantos2 = np.float32([
    [0, 0], [w2, 0], [w2, h2], [0, h2]
]).reshape(-1, 1, 2)

cantos2_transformados = cv2.perspectiveTransform(cantos2, H)

todos = np.concatenate(
    [cantos1, cantos2_transformados], axis=0
).reshape(-1, 2)

xmin, ymin = np.floor(todos.min(axis=0)).astype(int)
xmax, ymax = np.ceil(todos.max(axis=0)).astype(int)

T = np.array([
    [1, 0, -xmin],
    [0, 1, -ymin],
    [0, 0, 1]
], dtype=np.float64)

largura = int(xmax - xmin)
altura = int(ymax - ymin)

# 7. Aplicar warpPerspective nas duas imagens
warp1 = cv2.warpPerspective(img1, T, (largura, altura))
warp2 = cv2.warpPerspective(img2, T @ H, (largura, altura))

# Mascaras para identificar pixels validos
mask1 = cv2.warpPerspective(
    np.full((h1, w1), 255, dtype=np.uint8),
    T, (largura, altura)
) > 0

mask2 = cv2.warpPerspective(
    np.full((h2, w2), 255, dtype=np.uint8),
    T @ H, (largura, altura)
) > 0

# 8. Construir o mosaico e combinar a sobreposicao
mosaico = warp1.copy()
mosaico[mask2 & ~mask1] = warp2[mask2 & ~mask1]

comum = mask1 & mask2

mosaico[comum] = (
    0.5 * warp1[comum].astype(np.float32)
    + 0.5 * warp2[comum].astype(np.float32)
).astype(np.uint8)

# 9. Salvar e exibir o resultado
caminho = SAIDA / "mosaico_panoramico.png"
cv2.imwrite(str(caminho), mosaico)

print("Mosaico salvo em:", caminho)

cv2.imshow("Mosaico panoramico - Item A", mosaico)
cv2.waitKey(0)
cv2.destroyAllWindows()