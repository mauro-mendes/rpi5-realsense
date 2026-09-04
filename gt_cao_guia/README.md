# gt_cao_guia — ground truth por RealSense fixa + ArUco

Rastreamento da trajetória de uma pessoa (e/ou de uma **bola verde** no capacete) para servir
de *ground truth* nos testes com o robô cão-guia. A RealSense fica **parada** (mesa/tripé,
nivelada), um marcador ArUco no chão define a origem do mundo, e o alvo é localizado por
profundidade + deprojeção.

É a versão corrigida do script original, que está preservado em
[`referencia/original_colega.py`](referencia/original_colega.py). O diagnóstico completo do
que estava errado — incluindo a origem do deslocamento de ~60 cm — está em
[`referencia/DIAGNOSTICO.md`](referencia/DIAGNOSTICO.md).

---

## Instalação

```bash
pip install -r requirements.txt
```

`pyrealsense2` não tem wheel para ARM64: no Raspberry Pi é preciso compilar da fonte
(passo a passo em [`../docs/notes.md`](../docs/notes.md)). Em x86_64 o `pip install` resolve.

Para rastrear pessoa é preciso o `yolov8n.pt` na pasta do script (uso offline).

---

## Antes de rodar: quatro medidas com trena

Sem elas o sistema roda, mas você não tem como saber se a escala está certa — e um erro de
escala desloca **toda** a trajetória de forma rígida.

| Medida | Vira |
|---|---|
| Lado do **quadrado preto** do marcador (borda preta inclusa, margem branca fora) | `--marker-size` |
| Altura da câmera até o chão, no **centro da lente** | `--cam-height` |
| Distância horizontal da câmera até o centro do marcador de referência | `--cam-dist` |
| Altura da bola no capacete, **por participante**, de pé | `--ball-height` |

As duas últimas são conferências independentes: o script compara o que o ArUco calcula com o
que você mediu e, se discordarem, diz o `--marker-size` correto.

### Onde colocar a câmera

A bola sai do quadro quando `|altura_bola − altura_câmera| / tan(meio-FOV)` fica maior que a
distância de trabalho. Com a câmera na mesa (0,78 m) e a bola a 1,78 m, a bola **só aparece
além de ~2,5 m**. Ponha a câmera no **meio da faixa de alturas das bolas** dos participantes:
o problema some para todos, e o marcador no chão deixa de ser visto em ângulo rasante (o que
melhora muito a pose).

Nivele a câmera com [`../tools/level_camera.py`](../tools/level_camera.py) (usa o IMU da
D435i). Com ela nivelada, a flag `--level` passa a valer.

---

## Rodar

```bash
python realsense_gt.py --ref-id 30 --marker-size 0.1400 \
                       --cam-height 0.78 --cam-dist 2.20 \
                       --ball-height 1.78 --level --prefix P01
```

| Tecla | |
|---|---|
| `ESPAÇO` | inicia / encerra uma passada (1 CSV + 1 `.meta.json` por passada) |
| `B` | liga/desliga a máscara HSV, para ajustar `--hsv` até só a bola acender |
| `R` | refaz o travamento da pose (se mexeram na câmera) |
| `Q` | sai |

Opções úteis: `--track ball|person|both` (default `both`; `ball` nem carrega o YOLO),
`--every-n 2` (roda o YOLO a cada 2 frames), `--bag sessao.bag` (grava o stream inteiro,
~1–2 GB/min, permite reprocessar tudo depois).

### Confira o bloco `[TRAVADO]`

No arranque o script imprime a posição da câmera no referencial do marcador e compara com as
suas trenas. **Se ele disser `*** ESCALA ERRADA`, rode de novo com o `--marker-size` que ele
sugere antes de coletar qualquer dado.**

---

## Marcadores

[`arucos/ARUCOS_140mm_A4_IMPRIMIR.pdf`](arucos/ARUCOS_140mm_A4_IMPRIMIR.pdf) — 4 páginas A4,
IDs 30–33, DICT_4X4_50, quadrado preto de 140,0 mm.

**Imprima em "Tamanho real" / 100%, sem "ajustar à página".** Cada folha traz uma régua de
100 mm: confira com a trena depois de imprimir. Se não medir 100 mm, a impressora escalou.

Cole em papelão rígido ou madeirinha — marcador ondulado destrói a pose, porque o `solvePnP`
assume os 4 cantos coplanares.

Outros tamanhos/IDs: `python gerar_arucos.py --ids 40 41 --size-mm 120`

---

## O que sai

`trajetorias/<prefix>_<NN>_<timestamp>.csv` + o `.meta.json` ao lado.

O CSV mantém as colunas do script original (`timestamp_iso, track_id, x_world, y_world,
z_world`) e acrescenta o que faltava para reprocessar depois sem repetir a coleta:

| coluna | para quê |
|---|---|
| `p_cam_x/y/z` | o ponto no referencial da **câmera**, antes da transformação para o mundo. Se a pose ou a escala do ArUco estiverem erradas, dá para recalcular tudo offline |
| `wall` | epoch, para alinhar com a câmera do teto |
| `n_px`, `depth_std` | quantos pixels sustentaram a medida e a dispersão deles — é o critério de qualidade para descartar amostra ruim |
| `bx1..by2`, `u_px`, `v_px`, `clipped` | refazer a extração do ponto do zero, offline |
| `target` | separa linha de bola e de pessoa no mesmo arquivo |

O `.meta.json` guarda a pose travada, os intrínsecos, os parâmetros da sessão e o **commit do
código** que gerou aquela passada — o script também imprime o commit no arranque.

---

## Ciclo de trabalho

```
laboratório: git pull  →  roda  →  reporta (com o commit do arranque)
      PC:    corrige   →  push
```

Ao reportar um problema, mande sempre: o bloco `[TRAVADO]` do terminal, um `.csv` e o
`.meta.json` da mesma passada. Com o commit no `.meta.json` dá para saber exatamente qual
versão rodou.

---

## Dados

A pasta `dados/` é **ignorada pelo git** de propósito: contém dados de teste e imagens de
participantes identificáveis, que não devem ir para um repositório remoto.
