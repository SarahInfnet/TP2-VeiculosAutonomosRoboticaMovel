from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class ConfiguracaoFaixas:
    largura: int = 1280
    altura: int = 720

    src: tuple = (
        (570, 430),
        (710, 430),
        (1150, 710),
        (130, 710),
    )

    dst: tuple = (
        (290, 0),
        (990, 0),
        (990, 719),
        (290, 719),
    )

    metros_por_pixel_x: float | None = None
    metros_por_pixel_y: float | None = None

    largura_minima_px: float = 350
    largura_maxima_px: float = 900

    margem_janela: int = 100
    numero_janelas: int = 9
    minimo_pixels_janela: int = 30
    minimo_pixels_faixa: int = 180

    limiar_branco: int = 155
    limiar_amarelo: int = 90
    limiar_sobel: int = 23


def configuracao_tp1():
    """Configuracao original do TP1."""
    return ConfiguracaoFaixas(
        metros_por_pixel_x=3.7 / 700,
        metros_por_pixel_y=30 / 720,
    )


def configuracao_metadrive():
    """Perspectiva estimada para a camera do MetaDrive."""
    return ConfiguracaoFaixas(
        src=(
            (620, 290),
            (660, 290),
            (830, 520),
            (500, 520),
        ),
        largura_minima_px=350,
        largura_maxima_px=900,
        margem_janela=95,
    )


def configuracao_imagem_real():
    """Perspectiva estimada para a fotografia real."""
    return ConfiguracaoFaixas(
        src=(
            (663, 359),
            (815, 359),
            (1249, 693),
            (701, 693),
        ),
        largura_minima_px=350,
        largura_maxima_px=900,
        margem_janela=110,
    )


class DetectorFaixas:
    def __init__(self, configuracao=None):
        self.config = configuracao or configuracao_tp1()
        c = self.config

        self.m = cv2.getPerspectiveTransform(
            np.float32(c.src),
            np.float32(c.dst),
        )

        self.m_inversa = cv2.getPerspectiveTransform(
            np.float32(c.dst),
            np.float32(c.src),
        )

    def criar_mascara(self, bev):
        """Segmentacao HLS e Sobel X, baseada no TP1."""
        c = self.config

        hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
        l = hls[:, :, 1]
        s = hls[:, :, 2]

        cinza = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)

        gradiente = np.abs(
            cv2.Sobel(
                cinza,
                cv2.CV_64F,
                1,
                0,
                ksize=3,
            )
        )

        maximo = gradiente.max()

        if maximo > 0:
            sobel = np.uint8(255 * gradiente / maximo)
        else:
            sobel = np.zeros_like(cinza)

        amarelo = (s >= c.limiar_amarelo) & (l >= 75)
        branco = (l >= c.limiar_branco) & (s < 105)
        bordas = sobel >= c.limiar_sobel

        mascara = np.uint8(
            (amarelo | branco | (bordas & (l >= 85))) * 255
        )

        mascara = cv2.morphologyEx(
            mascara,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
        )

        origem = np.full(
            (c.altura, c.largura),
            255,
            dtype=np.uint8,
        )

        regiao_valida = cv2.warpPerspective(
            origem,
            self.m,
            (c.largura, c.altura),
        )

        mascara[regiao_valida < 250] = 0
        return mascara

    def detectar_segmentos(self, mascara):
        """Canny e HoughLinesP, preservados do TP1."""
        arestas = cv2.Canny(mascara, 60, 150)

        segmentos = cv2.HoughLinesP(
            arestas,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=35,
            maxLineGap=70,
        )

        visualizacao = cv2.cvtColor(
            mascara,
            cv2.COLOR_GRAY2BGR,
        )

        if segmentos is not None:
            for x1, y1, x2, y2 in segmentos.reshape(-1, 4):
                cv2.line(
                    visualizacao,
                    (x1, y1),
                    (x2, y2),
                    (0, 140, 255),
                    2,
                )

        return segmentos, visualizacao

    def ajustar_faixas(self, mascara, segmentos=None):
        """Ajuste polinomial com busca por janelas deslizantes."""
        c = self.config
        altura, largura = mascara.shape

        histograma = np.sum(
            mascara[altura // 2:, :] > 0,
            axis=0,
        )

        centro = largura // 2

        a = max(0, centro - 500)
        b = max(1, centro - 80)
        d = min(largura - 1, centro + 80)
        e = min(largura, centro + 500)

        if a >= b or d >= e:
            return None, None

        if histograma[a:b].max() < 5:
            return None, None

        if histograma[d:e].max() < 5:
            return None, None

        pos_esq = int(a + np.argmax(histograma[a:b]))
        pos_dir = int(d + np.argmax(histograma[d:e]))

        ys, xs = np.nonzero(mascara)

        altura_janela = max(
            1,
            altura // c.numero_janelas,
        )

        indices_esq = []
        indices_dir = []

        for janela in range(c.numero_janelas):
            y_alto = max(
                0,
                altura - (janela + 1) * altura_janela,
            )

            y_baixo = altura - janela * altura_janela
            margem = c.margem_janela

            encontrados_esq = np.where(
                (ys >= y_alto)
                & (ys < y_baixo)
                & (xs >= pos_esq - margem)
                & (xs < pos_esq + margem)
            )[0]

            encontrados_dir = np.where(
                (ys >= y_alto)
                & (ys < y_baixo)
                & (xs >= pos_dir - margem)
                & (xs < pos_dir + margem)
            )[0]

            indices_esq.append(encontrados_esq)
            indices_dir.append(encontrados_dir)

            if len(encontrados_esq) >= c.minimo_pixels_janela:
                pos_esq = int(
                    np.median(xs[encontrados_esq])
                )

            if len(encontrados_dir) >= c.minimo_pixels_janela:
                pos_dir = int(
                    np.median(xs[encontrados_dir])
                )

        indices_esq = np.concatenate(indices_esq)
        indices_dir = np.concatenate(indices_dir)

        if (
            len(indices_esq) < c.minimo_pixels_faixa
            or len(indices_dir) < c.minimo_pixels_faixa
        ):
            return None, None

        try:
            esquerda = np.polyfit(
                ys[indices_esq],
                xs[indices_esq],
                2,
            )

            direita = np.polyfit(
                ys[indices_dir],
                xs[indices_dir],
                2,
            )
        except (ValueError, np.linalg.LinAlgError):
            return None, None

        if not self.validar_faixas(esquerda, direita):
            return None, None

        return esquerda, direita

    def validar_faixas(self, esquerda, direita):
        """Verifica se as linhas formam uma faixa plausivel."""
        c = self.config

        alturas = np.array(
            [180, 300, 420, 540, 650],
            dtype=np.float64,
        )

        x_esq = np.polyval(esquerda, alturas)
        x_dir = np.polyval(direita, alturas)

        if not (
            np.all(np.isfinite(x_esq))
            and np.all(np.isfinite(x_dir))
        ):
            return False

        larguras = x_dir - x_esq

        if np.any(larguras < c.largura_minima_px):
            return False

        if np.any(larguras > c.largura_maxima_px):
            return False

        if np.ptp(larguras) > 220:
            return False

        if (
            np.any(x_esq < -100)
            or np.any(x_dir > c.largura + 100)
        ):
            return False

        return True

    def calcular_medidas(self, esquerda, direita):
        """Calcula desvio e curvatura no plano BEV."""
        c = self.config
        y = c.altura - 1

        x_esq = float(np.polyval(esquerda, y))
        x_dir = float(np.polyval(direita, y))

        centro_faixa = (x_esq + x_dir) / 2
        centro_veiculo = c.largura / 2

        desvio_px = centro_veiculo - centro_faixa

        def raio_px(coeficientes):
            a, b, _ = coeficientes
            denominador = abs(2 * a)

            if denominador < 1e-10:
                return float("inf")

            return float(
                (1 + (2 * a * y + b) ** 2) ** 1.5
                / denominador
            )

        raios = [
            raio_px(esquerda),
            raio_px(direita),
        ]

        raio_medio_px = float(np.mean(raios))

        if np.isinf(raio_medio_px):
            curvatura_px_inversa = 0.0
        else:
            curvatura_px_inversa = 1.0 / raio_medio_px

        medidas = {
            "desvio_px": float(desvio_px),
            "desvio_m": None,
            "raio_curvatura_px": raio_medio_px,
            "curvatura_px_inversa": curvatura_px_inversa,
            "raio_curvatura_m": None,
        }

        mx = c.metros_por_pixel_x
        my = c.metros_por_pixel_y

        # Sem calibracao nao calculamos medidas em metros.
        if (
            mx is None
            or my is None
            or mx <= 0
            or my <= 0
        ):
            return medidas

        medidas["desvio_m"] = float(desvio_px * mx)

        def raio_m(coeficientes):
            a = coeficientes[0] * mx / (my ** 2)
            b = coeficientes[1] * mx / my
            y_m = y * my

            return (
                (1 + (2 * a * y_m + b) ** 2) ** 1.5
                / (abs(2 * a) + 1e-12)
            )

        medidas["raio_curvatura_m"] = float(
            (
                raio_m(esquerda)
                + raio_m(direita)
            ) / 2
        )

        return medidas

    def preencher_faixa(self, frame, esquerda, direita):
        """Projeta a area detectada sobre a imagem original."""
        c = self.config

        ys = np.arange(
            c.altura,
            dtype=np.float64,
        )

        xs_esq = np.clip(
            np.polyval(esquerda, ys),
            0,
            c.largura - 1,
        )

        xs_dir = np.clip(
            np.polyval(direita, ys),
            0,
            c.largura - 1,
        )

        pontos_esq = np.column_stack(
            (xs_esq, ys)
        ).astype(np.int32)

        pontos_dir = np.column_stack(
            (xs_dir, ys)
        ).astype(np.int32)

        camada = np.zeros_like(frame)

        poligono = np.vstack(
            (pontos_esq, pontos_dir[::-1])
        )

        cv2.fillPoly(
            camada,
            [poligono],
            (30, 190, 45),
        )

        cv2.polylines(
            camada,
            [pontos_esq],
            False,
            (255, 255, 255),
            5,
        )

        cv2.polylines(
            camada,
            [pontos_dir],
            False,
            (0, 235, 255),
            5,
        )

        camada_original = cv2.warpPerspective(
            camada,
            self.m_inversa,
            (c.largura, c.altura),
        )

        resultado = frame.copy()

        valido = np.any(
            camada_original > 0,
            axis=2,
        )

        mistura = cv2.addWeighted(
            frame,
            0.65,
            camada_original,
            0.35,
            0,
        )

        resultado[valido] = mistura[valido]
        return resultado

    def criar_painel(self, frame, bev, mascara, hough):
        """Painel visual das quatro etapas do processamento."""
        imagens = [
            ("1. ORIGINAL", frame),
            ("2. BEV", bev),
            ("3. HLS + SOBEL", mascara),
            ("4. CANNY + HOUGH", hough),
        ]

        blocos = []

        for titulo, imagem in imagens:
            if imagem.ndim == 2:
                imagem = cv2.cvtColor(
                    imagem,
                    cv2.COLOR_GRAY2BGR,
                )

            imagem = cv2.resize(
                imagem,
                (640, 360),
            )

            cv2.rectangle(
                imagem,
                (0, 0),
                (640, 38),
                (20, 30, 45),
                -1,
            )

            cv2.putText(
                imagem,
                titulo,
                (15, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            blocos.append(imagem)

        return np.vstack(
            (
                np.hstack(blocos[:2]),
                np.hstack(blocos[2:]),
            )
        )

    def processar_frame(self, frame):
        """Retorna imagem anotada, medidas e painel."""
        c = self.config

        frame = cv2.resize(
            frame,
            (c.largura, c.altura),
        )

        bev = cv2.warpPerspective(
            frame,
            self.m,
            (c.largura, c.altura),
        )

        mascara = self.criar_mascara(bev)

        segmentos, hough = self.detectar_segmentos(
            mascara
        )

        esquerda, direita = self.ajustar_faixas(
            mascara,
            segmentos,
        )

        medidas = {
            "detectado": False,
            "desvio_px": None,
            "desvio_m": None,
            "raio_curvatura_px": None,
            "curvatura_px_inversa": None,
            "raio_curvatura_m": None,
        }

        resultado = frame.copy()

        if esquerda is not None and direita is not None:
            medidas.update(
                self.calcular_medidas(esquerda, direita)
            )

            medidas["detectado"] = True

            resultado = self.preencher_faixa(
                frame,
                esquerda,
                direita,
            )

        if medidas["detectado"]:
            if medidas["desvio_m"] is not None:
                texto_desvio = (
                    f"Desvio lateral: "
                    f"{medidas['desvio_m']:+.3f} m"
                )
            else:
                texto_desvio = (
                    f"Desvio lateral: "
                    f"{medidas['desvio_px']:+.1f} px "
                    "(sem calibracao)"
                )

            if medidas["raio_curvatura_m"] is not None:
                texto_raio = (
                    "Raio de curvatura: "
                    f"{medidas['raio_curvatura_m']:.1f} m"
                )
            else:
                raio = medidas["raio_curvatura_px"]

                if np.isinf(raio):
                    valor = "infinito"
                else:
                    valor = f"{raio:.0f}"

                texto_raio = (
                    f"Raio de curvatura BEV: {valor} px"
                )

            mensagens = [
                texto_desvio,
                texto_raio,
            ]
        else:
            mensagens = [
                "Faixas nao identificadas"
            ]

        for i, mensagem in enumerate(mensagens):
            cv2.putText(
                resultado,
                mensagem,
                (35, 100 + 42 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

        painel = self.criar_painel(
            frame,
            bev,
            mascara,
            hough,
        )

        return resultado, medidas, painel