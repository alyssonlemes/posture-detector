"""
=============================================================
  Detector de Postura Corporal - OpenCV + MediaPipe
  Autor: Projeto de Visão Computacional
  Descrição: Detecta e analisa a postura da pessoa em tempo
             real usando câmera ou vídeo, identificando
             problemas como cabeça inclinada, ombros
             desnivelados e curvatura da coluna.
=============================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import argparse

# ─── Configurações ────────────────────────────────────────
COLORS = {
    "green":  (0, 200, 0),
    "red":    (0, 0, 220),
    "yellow": (0, 200, 220),
    "blue":   (220, 100, 0),
    "white":  (255, 255, 255),
    "black":  (0, 0, 0),
    "orange": (0, 140, 255),
    "gray":   (150, 150, 150),
}

# Limiares de ângulo para alertas
THRESHOLDS = {
    "head_tilt":        10,   # graus de inclinação lateral da cabeça
    "shoulder_tilt":    8,    # graus de inclinação dos ombros
    "forward_head":     15,   # graus de projeção da cabeça para frente
    "spine_curvature":  12,   # desvio lateral da coluna (pixels normalizados)
}

mp_pose      = mp.solutions.pose
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles


# ─── Funções auxiliares ───────────────────────────────────

def angle_between_points(p1, p2, p3=None):
    """
    Calcula ângulo entre dois pontos em relação ao eixo horizontal,
    ou o ângulo em p2 formado por p1-p2-p3.
    """
    if p3 is None:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(dy, dx))
    else:
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return math.degrees(math.acos(np.clip(cos_a, -1, 1)))


def get_landmark(landmarks, idx, w, h):
    """Retorna coordenadas do landmark em pixels."""
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1, alpha=0.7):
    """Desenha retângulo arredondado com transparência."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, thickness)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, thickness)
    cv2.circle(overlay, (x + r, y + r), r, color, thickness)
    cv2.circle(overlay, (x + w - r, y + r), r, color, thickness)
    cv2.circle(overlay, (x + r, y + h - r), r, color, thickness)
    cv2.circle(overlay, (x + w - r, y + h - r), r, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_status_bar(frame, label, value, max_val, x, y, bar_w, bar_h, color):
    """Desenha barra de status colorida."""
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), COLORS["gray"], -1)
    fill = int(min(value / max_val, 1.0) * bar_w)
    cv2.rectangle(frame, (x, y), (x + fill, y + bar_h), color, -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), COLORS["white"], 1)
    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["white"], 1)


def classify_posture(issues):
    """Classifica a postura geral com base nos problemas detectados."""
    if len(issues) == 0:
        return "BOA POSTURA", COLORS["green"]
    elif len(issues) == 1:
        return "POSTURA REGULAR", COLORS["yellow"]
    else:
        return "POSTURA RUIM", COLORS["red"]


# ─── Classe principal ─────────────────────────────────────

class PostureDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.alert_history = []   # histórico de frames com alertas
        self.fps_history   = []
        self.prev_time     = time.time()

    def analyze_frame(self, frame):
        """Processa um frame e retorna análise de postura."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        output   = frame.copy()
        issues   = []
        metrics  = {}

        if not results.pose_landmarks:
            cv2.putText(output, "Nenhuma pessoa detectada", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS["yellow"], 2)
            return output, [], {}

        lm = results.pose_landmarks.landmark

        # — Landmarks principais —
        nose         = get_landmark(lm, mp_pose.PoseLandmark.NOSE, w, h)
        left_ear     = get_landmark(lm, mp_pose.PoseLandmark.LEFT_EAR, w, h)
        right_ear    = get_landmark(lm, mp_pose.PoseLandmark.RIGHT_EAR, w, h)
        left_shoulder  = get_landmark(lm, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        right_shoulder = get_landmark(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        left_hip     = get_landmark(lm, mp_pose.PoseLandmark.LEFT_HIP, w, h)
        right_hip    = get_landmark(lm, mp_pose.PoseLandmark.RIGHT_HIP, w, h)
        left_knee    = get_landmark(lm, mp_pose.PoseLandmark.LEFT_KNEE, w, h)
        right_knee   = get_landmark(lm, mp_pose.PoseLandmark.RIGHT_KNEE, w, h)

        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                        (left_shoulder[1] + right_shoulder[1]) // 2)
        mid_hip      = ((left_hip[0] + right_hip[0]) // 2,
                        (left_hip[1] + right_hip[1]) // 2)
        mid_ear      = ((left_ear[0] + right_ear[0]) // 2,
                        (left_ear[1] + right_ear[1]) // 2)

        # ── 1. Inclinação da cabeça (lateral) ──────────────
        head_tilt = abs(angle_between_points(left_ear, right_ear))
        head_tilt = abs(head_tilt) if head_tilt < 90 else abs(180 - head_tilt)
        metrics["Inclinação da Cabeça"] = head_tilt
        if head_tilt > THRESHOLDS["head_tilt"]:
            issues.append(f"Cabeça inclinada ({head_tilt:.1f}°)")

        # ── 2. Ombros desnivelados ──────────────────────────
        shoulder_tilt = abs(angle_between_points(left_shoulder, right_shoulder))
        shoulder_tilt = shoulder_tilt if shoulder_tilt < 90 else abs(180 - shoulder_tilt)
        metrics["Nível dos Ombros"] = shoulder_tilt
        if shoulder_tilt > THRESHOLDS["shoulder_tilt"]:
            issues.append(f"Ombros desnivelados ({shoulder_tilt:.1f}°)")

        # ── 3. Cabeça projetada para frente ────────────────
        # (comparação horizontal entre orelha e ombro)
        ear_shoulder_dx = abs(mid_ear[0] - mid_shoulder[0])
        ear_shoulder_dx_norm = (ear_shoulder_dx / w) * 100   # % da largura
        metrics["Cabeça p/ Frente"] = ear_shoulder_dx_norm
        if ear_shoulder_dx_norm > THRESHOLDS["forward_head"]:
            issues.append(f"Cabeça projetada para frente ({ear_shoulder_dx_norm:.1f})")

        # ── 4. Curvatura lateral da coluna ──────────────────
        # Desvio horizontal: nariz → ombro → quadril (deve ser vertical)
        spine_dev = abs(mid_shoulder[0] - mid_hip[0])
        spine_dev_norm = (spine_dev / w) * 100
        metrics["Curvatura Coluna"] = spine_dev_norm
        if spine_dev_norm > THRESHOLDS["spine_curvature"]:
            issues.append(f"Coluna lateralmente desviada ({spine_dev_norm:.1f})")

        # ── 5. Ângulo pescoço–tronco ────────────────────────
        neck_angle = angle_between_points(mid_ear, mid_shoulder, mid_hip)
        metrics["Ângulo Pescoço"] = neck_angle

        # ── Desenhar esqueleto personalizado ───────────────
        self._draw_skeleton(output, {
            "nose": nose, "left_ear": left_ear, "right_ear": right_ear,
            "left_shoulder": left_shoulder, "right_shoulder": right_shoulder,
            "left_hip": left_hip, "right_hip": right_hip,
            "left_knee": left_knee, "right_knee": right_knee,
            "mid_shoulder": mid_shoulder, "mid_hip": mid_hip,
            "mid_ear": mid_ear,
        }, issues)

        # ── HUD principal ───────────────────────────────────
        self._draw_hud(output, issues, metrics, w, h)

        # ── Alertas visuais ─────────────────────────────────
        if issues:
            self._draw_alert_border(output, h, w)

        return output, issues, metrics

    def _draw_skeleton(self, frame, pts, issues):
        color_ok  = COLORS["green"]
        color_bad = COLORS["red"]

        has_head   = any("cabeça" in i.lower() or "Cabeça" in i for i in issues)
        has_shoulder = any("ombro" in i.lower() for i in issues)
        has_spine  = any("coluna" in i.lower() for i in issues)

        # Conexões do corpo
        connections = [
            (pts["left_ear"],      pts["right_ear"],     color_bad if has_head else color_ok),
            (pts["nose"],          pts["left_ear"],       color_bad if has_head else color_ok),
            (pts["nose"],          pts["right_ear"],      color_bad if has_head else color_ok),
            (pts["left_shoulder"], pts["right_shoulder"], color_bad if has_shoulder else color_ok),
            (pts["mid_shoulder"],  pts["mid_hip"],        color_bad if has_spine else color_ok),
            (pts["left_shoulder"], pts["left_hip"],       color_ok),
            (pts["right_shoulder"],pts["right_hip"],      color_ok),
            (pts["left_hip"],      pts["right_hip"],      color_ok),
            (pts["left_hip"],      pts["left_knee"],      color_ok),
            (pts["right_hip"],     pts["right_knee"],     color_ok),
            (pts["mid_ear"],       pts["mid_shoulder"],   color_bad if has_head else color_ok),
        ]

        for p1, p2, col in connections:
            cv2.line(frame, p1, p2, col, 2, cv2.LINE_AA)

        # Pontos
        for name, pt in pts.items():
            if "mid" not in name:
                cv2.circle(frame, pt, 5, COLORS["white"], -1)
                cv2.circle(frame, pt, 5, COLORS["blue"], 1)

    def _draw_hud(self, frame, issues, metrics, w, h):
        panel_w = 310
        panel_h = 30 + len(metrics) * 38 + 20 + len(issues) * 22 + 20
        px, py  = 10, 10

        # Fundo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h),
                      COLORS["blue"], 1)

        # Título
        cv2.putText(frame, "ANALISE DE POSTURA", (px + 8, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["blue"], 2)

        # Barras de métricas
        bar_configs = [
            ("Incl. Cabeça",    "Inclinação da Cabeça",  30,  COLORS["yellow"]),
            ("Nivel Ombros",    "Nível dos Ombros",      20,  COLORS["orange"]),
            ("Cabeca p/Frente", "Cabeça p/ Frente",      25,  COLORS["orange"]),
            ("Curv. Coluna",    "Curvatura Coluna",      25,  COLORS["red"]),
        ]

        y_offset = py + 40
        for label, key, max_v, col in bar_configs:
            if key in metrics:
                val = metrics[key]
                draw_status_bar(frame, f"{label}: {val:.1f}",
                                val, max_v, px + 8, y_offset, panel_w - 20, 14, col)
                y_offset += 38

        # Linha separadora
        cv2.line(frame, (px + 8, y_offset), (px + panel_w - 8, y_offset),
                 COLORS["gray"], 1)
        y_offset += 10

        # Lista de problemas
        if issues:
            cv2.putText(frame, "Problemas detectados:", (px + 8, y_offset + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["red"], 1)
            y_offset += 20
            for issue in issues:
                cv2.putText(frame, f"• {issue}", (px + 8, y_offset + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLORS["yellow"], 1)
                y_offset += 22
        else:
            cv2.putText(frame, "✓ Postura correta!", (px + 8, y_offset + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLORS["green"], 1)

        # Status geral (canto superior direito)
        status, status_color = classify_posture(issues)
        sx = w - 220
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (sx - 5, 10), (w - 10, 45), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, status, (sx, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

        # FPS
        now = time.time()
        fps = 1.0 / (now - self.prev_time + 1e-6)
        self.prev_time = now
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        cv2.putText(frame, f"FPS: {avg_fps:.0f}", (w - 90, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["gray"], 1)

        # Ângulo pescoço
        if "Ângulo Pescoço" in metrics:
            cv2.putText(frame, f"Angulo pescoco: {metrics['Ângulo Pescoço']:.1f}deg",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["gray"], 1)

    def _draw_alert_border(self, frame, h, w):
        thickness = 6
        alpha = 0.5 + 0.5 * abs(math.sin(time.time() * 3))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), COLORS["red"], thickness)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def release(self):
        self.pose.close()


# ─── Loop principal ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detector de Postura Corporal")
    parser.add_argument("--source", default="0",
                        help="Câmera (0,1,...) ou caminho de vídeo")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--save",   action="store_true",
                        help="Salvar saída em output.mp4")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a fonte: {source}")
        return

    detector = PostureDetector()
    writer   = None

    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, 30,
                                 (args.width, args.height))

    print("\n============================")
    print("  Detector de Postura Ativo ")
    print("  Pressione 'q' para sair   ")
    print("============================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou câmera desconectada.")
            break

        output, issues, metrics = detector.analyze_frame(frame)

        # Instruções na tela
        cv2.putText(output, "Pressione 'q' para sair",
                    (output.shape[1] - 220, output.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["gray"], 1)

        cv2.imshow("Detector de Postura", output)

        if writer:
            writer.write(output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Encerrado.")


if __name__ == "__main__":
    main()
