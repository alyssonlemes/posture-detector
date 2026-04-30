"""
demo_imagem.py — Testa o detector de postura em uma imagem sintética.
Útil para validar o pipeline sem câmera.
"""

import cv2
import numpy as np
from posture_detector import PostureDetector


def create_synthetic_person(w=960, h=720):
    """
    Cria um frame sintético com 'pessoa' representada por pontos-chave
    desenhados manualmente para testar o pipeline.
    Retorna frame BGR.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)   # fundo cinza escuro

    # Grade de fundo
    for x in range(0, w, 60):
        cv2.line(frame, (x, 0), (x, h), (45, 45, 45), 1)
    for y in range(0, h, 60):
        cv2.line(frame, (0, y), (w, y), (45, 45, 45), 1)

    cv2.putText(frame,
                "DEMO - Posicione-se em frente a camera e execute posture_detector.py",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    return frame


def main():
    detector = PostureDetector()
    frame    = create_synthetic_person()

    output, issues, metrics = detector.analyze_frame(frame)

    cv2.imshow("Demo - Detector de Postura (sem câmera)", output)
    cv2.imwrite("demo_output.png", output)
    print("Imagem salva em demo_output.png")
    print("Pressione qualquer tecla para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    detector.release()


if __name__ == "__main__":
    main()
