# Detector de Postura Corporal
### Visão Computacional com OpenCV + MediaPipe

---

## Descrição

Sistema de detecção e análise de postura em tempo real usando **OpenCV** e **MediaPipe Pose**.  
O programa acessa a câmera, detecta os pontos-chave do corpo e analisa automaticamente a postura da pessoa, identificando desvios e alertando visualmente.

---

## O que é detectado

| Métrica | Descrição |
|---|---|
| **Inclinação da Cabeça** | Detecta se a cabeça está inclinada lateralmente |
| **Nível dos Ombros** | Verifica se os ombros estão nivelados |
| **Cabeça Projetada p/ Frente** | Detecta projeção anterior da cabeça ("tech neck") |
| **Curvatura Lateral da Coluna** | Verifica alinhamento vertical do tronco |
| **Ângulo do Pescoço** | Mede o ângulo formado entre orelha, ombro e quadril |

---

## Requisitos

- Python 3.11
- pip

---

## Instalação e execução

### Primeira vez
Clique duas vezes em **`INICIAR.bat`**  
O script cria o ambiente virtual, instala todas as dependências e já abre o programa.

### Demais vezes
Clique duas vezes em **`RODAR.bat`**  
Abre o programa diretamente, sem reinstalar nada.

### Via terminal
```bash
# Ativar ambiente virtual
venv\Scripts\activate

# Rodar com webcam padrão
python posture_detector.py

# Rodar com câmera secundária
python posture_detector.py --source 1

# Rodar com arquivo de vídeo
python posture_detector.py --source meu_video.mp4

# Salvar resultado em vídeo
python posture_detector.py --save
```

---

## Controles

| Tecla | Ação |
|---|---|
| `Q` | Fechar o programa |

---

## Interface (HUD)

- **Esqueleto verde** — segmento sem problemas
- **Esqueleto vermelho** — segmento com problema detectado
- **Painel esquerdo** — barras de métricas em tempo real
- **Status geral** (canto superior direito):
  - 🟢 `BOA POSTURA` — nenhum problema
  - 🟡 `POSTURA REGULAR` — 1 problema detectado
  - 🔴 `POSTURA RUIM` — 2 ou mais problemas
- **Borda vermelha piscando** — alerta visual de postura incorreta
- **FPS** — taxa de quadros atual

---

## Ajuste de sensibilidade

Edite as constantes no início de `posture_detector.py`:

```python
THRESHOLDS = {
    "head_tilt":       10,   # graus de inclinação lateral da cabeça
    "shoulder_tilt":    8,   # graus de inclinação dos ombros
    "forward_head":    15,   # deslocamento horizontal orelha-ombro (% da largura)
    "spine_curvature": 12,   # desvio lateral do tronco (% da largura)
}
```

Diminua os valores para detecção mais sensível, aumente para menos sensível.

---

## Estrutura do projeto

```
posture_detector/
├── posture_detector.py   # Programa principal
├── demo_imagem.py        # Teste sem câmera
├── INICIAR.bat           # Setup completo + execução (primeira vez)
├── RODAR.bat             # Execução rápida (demais vezes)
├── requirements.txt      # Dependências
└── README.md             # Este arquivo
```

---

## Tecnologias

- **OpenCV** — captura de vídeo, desenho e exibição
- **MediaPipe Pose** — estimativa de pose corporal com 33 landmarks
- **NumPy** — cálculos vetoriais e trigonométricos

---

## Dicas para melhor detecção

- Fique a 1–2 metros da câmera
- Garanta boa iluminação no rosto e tronco
- Posicione a câmera na altura do peito ou levemente acima
- Vista roupas que contrastem com o fundo
