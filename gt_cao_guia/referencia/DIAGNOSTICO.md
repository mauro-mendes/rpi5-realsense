# Diagnóstico do script original

Análise de [`original_colega.py`](original_colega.py) e do CSV do teste de 03/09/2026
(`trajetoria_pessoa_1788443957.csv`, 1565 linhas, 80,8 s). Sintoma relatado: **todo o
rastreamento da pessoa ~60 cm mais próximo da origem do que deveria.**

---

## 1. O erro de 60 cm é de escala, não de ruído

O CSV contém um experimento natural: a pessoa fica **parada nos primeiros 44 s**.

| Fase parada (43 s, 625 amostras) | |
|---|---|
| Desvio em x / y / z | 0,60 / 1,11 / 0,50 cm |
| Deriva entre a 1ª e a 2ª metade | **6,6 mm** |

Pose que deriva 6,6 mm em 43 s é excelente. **Estável, porém não necessariamente correta**:
um marcador visto em ângulo rasante dá uma pose muito repetível e sistematicamente errada.
Logo o problema é **viés**, e viés de escala tem uma assinatura própria.

### A matemática

Para um alvo planar, se você informa pontos de objeto escalados por `s = tamanho_assumido /
tamanho_real`, a solução do PnP é exatamente `R' = R` (a rotação **não** muda) e `t' = s·t`.
Escalar toda a configuração 3D não altera a projeção. Propagando para a posição da pessoa:

```
p_est  = Rᵀ(p_cam − s·t)
p_real = Rᵀ(p_cam −   t)
────────────────────────────
erro = (s − 1) · C          onde C = −Rᵀ·t = posição da câmera no frame do marcador
```

`p_cam` não aparece no erro: ele **não depende de onde a pessoa está**. É uma translação
rígida de toda a trajetória, ao longo da reta câmera↔origem. Em outras palavras:

> **O erro na posição da pessoa é exatamente o erro na posição estimada da câmera.**

A pessoa é localizada como `câmera + raio de profundidade`; o raio vem do sensor e está
correto. Se o ArUco põe a câmera 60 cm longe demais do marcador, tudo fica 60 cm deslocado.

Substituindo `C = C_est/s`, isso colapsa em:

```
|deslocamento| = | distância_estimada − distância_real |
```

### Verificação numérica

Simulando um marcador **real de 11,8 cm** com o código assumindo 15 cm (câmera a 0,78 m,
2,5 m do marcador):

```
round-trip da pose (sem erro):   erro 0.000000 m
altura da câmera reportada:      0.992 m   (real 0.780)
s recuperado pela altura:        1.271  ->  --marker-size 0.1180  (exato)
deslocamento da pessoa:          71.0 cm  ==  |dist_est − dist_real|
```

### Onde está a câmera (triangulada a partir do CSV)

O ruído de profundidade é **radial**, então o eixo principal do ruído aponta para a câmera.
Enquanto a pessoa anda, esse eixo gira e as retas se cruzam:

```
seg0  pessoa=(-0.38,-0.02)  eixo=[+0.214 +0.933]   92% da variância
seg1  pessoa=(-0.37,+0.56)  eixo=[+0.205 +0.961]   92%
seg2  pessoa=(+0.25,+0.90)  eixo=[+0.279 +0.937]   87%
seg3  pessoa=(+0.81,+0.95)  eixo=[+0.386 +0.900]   89%
seg4  pessoa=(+1.40,+1.20)  eixo=[+0.505 +0.861]   97%
seg5  pessoa=(+1.84,+1.01)  eixo=[+0.595 +0.801]   92%
                      ↓
        câmera ≈ (-0.79, -2.40) — 2,53 m da origem
```

A componente vertical do raio é quase nula (−0,06), confirmando câmera **nivelada olhando na
horizontal**; e `z` da pessoa constante em 1,08 m ±5 mm confirma o marcador **deitado no chão**.

**Teste decisivo:** medir com trena a distância horizontal câmera→marcador. O ArUco diz
2,53 m. Se der ~1,93 m, o marcador é ~11,4 cm; se der ~3,13 m, é ~18,6 cm.

**Causa mais provável:** o OpenCV quer o lado do **quadrado preto**. Um ArUco 4×4 tem 6
células (4 de dados + 1 de borda preta por lado). Medir o quadrado **branco impresso**, que
costuma trazer 1 célula de margem de silêncio de cada lado, dá 8 células — **33% a mais**,
`s ≈ 1,33`, deslocamento previsto de 0,63 m.

> ⚠️ **Hipótese alternativa não descartada:** se a câmera do teto rastreia o **robô** (ele tem
> um marcador verde) e a RealSense rastreia a **pessoa**, parte dos 60 cm é o comprimento da
> guia, não um bug. Teste: o caminho tem uma curva de ~110°; se a direção de
> `(estimado − referência)` **girar** junto com o rumo, é a guia; se ficar **fixa**, é a escala.

---

## 2. Ruído: não é o ArUco, é a extração do ponto da pessoa

| | Parada | Andando |
|---|---|---|
| Ruído 3D (rms) | **0,6 cm** | **3,8 cm** — 6× pior |
| Salto entre amostras | ~1 cm | mediana 5,7 cm, p90 14,3, **máx 33 cm** |
| Amostras com v > 2 m/s | 0 | **13%** (impossível andando) |

O ArUco é o mesmo nas duas fases; o que muda é a pessoa se mexer. O culpado é ler **um único
pixel** de profundidade no centro do bbox. Consequências no CSV:

- Comprimento do caminho cru: **55,3 m**. Com mediana k=9: **9,0 m** (0,19 m/s, coerente).
  O cru está **6× inflado** só por ruído.
- **15 tracks fantasma.** Doze têm exatamente **10 linhas com uma única posição repetida** —
  `MAX_MISSES = 10`. Track espúrio nasce, nunca mais casa, e o CSV grava a posição congelada
  10 vezes até morrer. Todos entre t=48 s e t=79 s, ou seja, só durante a caminhada.
- 7% das linhas do track 0 são cópias exatas da anterior.

---

## 3. Lista de defeitos e o que foi feito

| # | Defeito no original | Correção |
|---|---|---|
| 1 | Pose do ArUco recalculada **todo frame** | travada por mediana de N amostras, nunca mais recalculada |
| 2 | `rvecs[0]` = marcador arbitrário, podia trocar sozinho no meio da sessão | `--ref-id` explícito |
| 3 | `ARUCO_MARKER_LENGTH` fixo no código, sem conferência | `--marker-size` + checagem contra trena (`--cam-height`, `--cam-dist`) |
| 4 | Profundidade de **1 pixel** no centro do bbox | amostragem por superfícies: separa em clusters e fica com a maior (rejeita fundo **e** móvel na frente) |
| 5 | Grava todos os tracks ativos, inclusive os não atualizados | só grava track confirmado (`--min-hits`) e atualizado no frame |
| 6 | Sem gate de velocidade na associação | rejeita associação que exija velocidade impossível |
| 7 | `list(matched_detections)[list(matched_tracks).index(i)]` — indexa um `set` pela posição de um elemento em **outro** `set`; desenhava a caixa no track errado | associação por dicionário |
| 8 | `estimatePoseSingleMarkers` (removido no OpenCV ≥ 4.7) | `solvePnP` + `SOLVEPNP_IPPE_SQUARE` |
| 9 | `enable_stream(accel)` e `GRAVITY_SAMPLES` presentes mas **nunca lidos** (sobra de versão ROS) | removidos; a vertical vem do `--level` |
| 10 | Transformação retroativa aplicava a pose do instante da 1ª detecção a todo o histórico anterior | não existe mais: a pose trava antes de qualquer captura |
| 11 | CSV sem o ponto no frame da câmera → escala errada obrigava a recoletar | `p_cam_x/y/z` + sidecar `.meta.json` com a pose e os intrínsecos |

### Inclinação residual, não resolvida

O `z` cresce com a distância (r = +0,57): 1,00 m a 2,3 m → 1,13 m a 4,2 m. Ajusta um plano
inclinado de **5,4°**, mas 100% alinhado com o eixo de visada (coeficiente em x = 0,0025).
Duas explicações que os dados não separam:

- inclinação real do frame do marcador (é o grau de liberdade pior determinado de um
  marcador rasante) — atacada pela flag `--level`; ou
- artefato: o raio até o peito aponta ligeiramente para cima, então erro de profundidade
  empurra o ponto para longe **e** para cima, criando correlação espúria.

Não é recorte dos pés — recorte deixaria o `z` mais alto quando **perto**, e o dado mostra o
contrário.

---

## 4. Melhoria pendente: mapa de marcadores

O script ainda usa **um** marcador. Há 4 no chão. Usar os 4 juntos num `solvePnP` leva a base
de poucos centímetros (um marcador rasante) para **metros**.

O ganho maior é outro: com um mapa de posições medidas com trena, **a escala do mundo deixa
de vir do `--marker-size`** e passa a vir das distâncias entre os marcadores — o que elimina
o bug dos 60 cm pela raiz. É o que o [`../../config/corridors.yaml`](../../config/corridors.yaml)
faz neste repositório.

Para implementar, basta medir o centro (x, y) de cada um dos 4 marcadores num referencial
comum: 8 números.
