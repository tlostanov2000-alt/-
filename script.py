
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

import cv2
import sqlite3
import datetime
import math
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTextEdit, QGroupBox, QGridLayout, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

import mediapipe as mp


class DatabaseManager:
    """Класс для работы с базой данных"""

    def __init__(self):
        self.conn = None
        self.cursor = None
        self.init_database()

    def init_database(self):
        try:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)

            db_path = data_dir / "safety_system.db"
            self.conn = sqlite3.connect(str(db_path))
            self.cursor = self.conn.cursor()

            self.cursor.execute('''
                                CREATE TABLE IF NOT EXISTS incidents
                                (
                                    id
                                    INTEGER
                                    PRIMARY
                                    KEY
                                    AUTOINCREMENT,
                                    timestamp
                                    TEXT
                                    NOT
                                    NULL,
                                    incident_type
                                    TEXT
                                    NOT
                                    NULL,
                                    severity
                                    TEXT
                                    NOT
                                    NULL,
                                    description
                                    TEXT,
                                    screenshot_path
                                    TEXT
                                )
                                ''')

            self.cursor.execute('''
                                CREATE TABLE IF NOT EXISTS sessions
                                (
                                    id
                                    INTEGER
                                    PRIMARY
                                    KEY
                                    AUTOINCREMENT,
                                    start_time
                                    TEXT
                                    NOT
                                    NULL,
                                    end_time
                                    TEXT,
                                    total_incidents
                                    INTEGER
                                    DEFAULT
                                    0,
                                    max_severity
                                    TEXT
                                )
                                ''')

            self.conn.commit()
            print("Database initialized successfully")

        except Exception as e:
            print(f"Database error: {e}")

    def log_incident(self, incident_type, severity, description, screenshot_path=None):
        try:
            timestamp = datetime.datetime.now().isoformat()
            self.cursor.execute('''
                                INSERT INTO incidents (timestamp, incident_type, severity, description, screenshot_path)
                                VALUES (?, ?, ?, ?, ?)
                                ''', (timestamp, incident_type, severity, description, screenshot_path))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error logging incident: {e}")
            return False

    def get_incidents(self, limit=100):
        try:
            self.cursor.execute('''
                                SELECT timestamp, incident_type, severity, description, screenshot_path
                                FROM incidents
                                ORDER BY id DESC
                                    LIMIT ?
                                ''', (limit,))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error getting incidents: {e}")
            return []

    def clear_incidents(self):
        """Очистка всех инцидентов из базы данных"""
        try:
            self.cursor.execute('DELETE FROM incidents')
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error clearing incidents: {e}")
            return False

    def close(self):
        if self.conn:
            self.conn.close()


class SafetyMonitor:
    """Система мониторинга для фронтальной камеры ноутбука"""

    def __init__(self, db_manager):
        self.db_manager = db_manager

        # Инициализация MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Детектор позы
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Детектор лица
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        # Face Mesh для детекции глаз
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Счетчики для отслеживания состояний
        self.no_face_frames = 0
        self.no_face_threshold = 15

        self.eye_closed_frames = 0
        self.eye_closed_threshold = 10
        self.ear_threshold = 0.2

        # === Счетчики для искривления позвоночника ===
        self.spine_curvature_frames = 0
        self.spine_angle_threshold = 15  # градусов
        self.spine_warning_threshold = 40  # 2 секунды -> предупреждение
        self.spine_critical_threshold = 100  # 5 секунд -> критическое
        self.spine_warning_sent = False
        self.spine_critical_sent = False

        # === Счетчики для поворота головы ===
        self.head_turn_frames = 0
        self.head_turn_threshold = 0.05  # порог смещения носа от центра плеч
        self.head_turn_warning_threshold = 40  # 2 секунды -> предупреждение
        self.head_turn_critical_threshold = 100  # 5 секунд -> критическое
        self.head_turn_warning_sent = False
        self.head_turn_critical_sent = False
        self.last_head_direction = None

        # Для ограничения уведомлений
        self.last_alert_time = datetime.datetime.now()
        self.alert_cooldown = 2

        print("SafetyMonitor initialized for laptop camera")

    def calculate_ear(self, face_landmarks):
        """Расчет EAR (Eye Aspect Ratio) для детекции закрытых глаз"""
        try:
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            left_eye = [face_landmarks.landmark[i] for i in left_eye_indices]
            right_eye = [face_landmarks.landmark[i] for i in right_eye_indices]

            ear_left = self._eye_aspect_ratio(left_eye)
            ear_right = self._eye_aspect_ratio(right_eye)

            return (ear_left + ear_right) / 2
        except:
            return 1.0

    def _eye_aspect_ratio(self, eye_points):
        """Расчет EAR для одного глаза"""
        v1 = self._distance(eye_points[1], eye_points[5])
        v2 = self._distance(eye_points[2], eye_points[4])
        h = self._distance(eye_points[0], eye_points[3])

        if h == 0:
            return 1.0
        return (v1 + v2) / (2.0 * h)

    def _distance(self, p1, p2):
        """Расчет расстояния между двумя точками"""
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

    def calculate_shoulder_angle(self, pose_landmarks):
        """Расчет угла наклона плеч относительно горизонтали"""
        if pose_landmarks is None:
            return None

        try:
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            dy = left_shoulder.y - right_shoulder.y
            dx = left_shoulder.x - right_shoulder.x

            if dx == 0:
                return 0

            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)

            return abs(angle_deg)

        except Exception as e:
            return None

    def detect_spine_curvature(self, pose_landmarks):
        """
        Детекция искривления позвоночника
        2 сек -> предупреждение (warning)
        5 сек -> критическое (critical) + скриншот
        """
        if pose_landmarks is None:
            self.spine_curvature_frames = 0
            self.spine_warning_sent = False
            self.spine_critical_sent = False
            return False, None, None, False

        try:
            angle = self.calculate_shoulder_angle(pose_landmarks)

            if angle is None:
                self.spine_curvature_frames = 0
                self.spine_warning_sent = False
                self.spine_critical_sent = False
                return False, None, None, False

            if angle >= self.spine_angle_threshold:
                self.spine_curvature_frames += 1

                if self.spine_curvature_frames >= self.spine_warning_threshold:
                    if not self.spine_warning_sent:
                        self.spine_warning_sent = True
                        return True, angle, 'warning', False

                if self.spine_curvature_frames >= self.spine_critical_threshold:
                    if not self.spine_critical_sent:
                        self.spine_critical_sent = True
                        return True, angle, 'critical', True
                    return True, angle, 'critical', False

                return True, angle, None, False

            else:
                self.spine_curvature_frames = 0
                self.spine_warning_sent = False
                self.spine_critical_sent = False

        except Exception as e:
            pass

        return False, None, None, False

    def detect_head_turn(self, pose_landmarks):
        """
        Детекция поворота головы влево/вправо
        2 сек -> предупреждение (warning)
        5 сек -> критическое (critical) + скриншот
        """
        if pose_landmarks is None:
            self.head_turn_frames = 0
            self.head_turn_warning_sent = False
            self.head_turn_critical_sent = False
            self.last_head_direction = None
            return False, None, None, False

        try:
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2

            # Смещение носа от центра плеч
            turn_offset = nose.x - shoulder_center_x

            # Определяем направление
            if abs(turn_offset) > self.head_turn_threshold:
                direction = "RIGHT" if turn_offset > 0 else "LEFT"

                self.head_turn_frames += 1

                if self.head_turn_frames >= self.head_turn_warning_threshold:
                    if not self.head_turn_warning_sent:
                        self.head_turn_warning_sent = True
                        return True, direction, 'warning', False

                if self.head_turn_frames >= self.head_turn_critical_threshold:
                    if not self.head_turn_critical_sent:
                        self.head_turn_critical_sent = True
                        return True, direction, 'critical', True
                    return True, direction, 'critical', False

                return True, direction, None, False
            else:
                self.head_turn_frames = 0
                self.head_turn_warning_sent = False
                self.head_turn_critical_sent = False
                self.last_head_direction = None

        except Exception as e:
            pass

        return False, None, None, False

    def detect_fatigue(self, face_results):
        """Детекция отсутствия лица - ВОЗВРАЩАЕТ DANGER"""
        if face_results is None or not face_results.detections:
            self.no_face_frames += 1
            if self.no_face_frames > self.no_face_threshold:
                return True, "Operator not visible"
            return False, None

        self.no_face_frames = 0
        return False, None

    def detect_sleepy_eyes(self, face_landmarks):
        """Детекция сонливости по глазам - КРИТИЧЕСКОЕ"""
        if face_landmarks is None:
            return False, None

        ear = self.calculate_ear(face_landmarks)

        if ear < self.ear_threshold:
            self.eye_closed_frames += 1
            if self.eye_closed_frames > self.eye_closed_threshold:
                return True, f"Eyes closed (EAR: {ear:.2f})"
        else:
            self.eye_closed_frames = 0

        return False, None

    def process_frame(self, frame):
        """Обработка кадра"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        violations = []
        spine_angle = None
        head_direction = None
        is_danger = False
        need_screenshot = False
        spine_severity = None
        head_severity = None
        is_absent = False

        # Детекция позы
        pose_results = self.pose.process(rgb_frame)

        # Детекция лица
        face_results = self.face_detector.process(rgb_frame)

        # Face Mesh для детекции глаз
        face_mesh_results = self.face_mesh.process(rgb_frame)

        # === Детекция отсутствия ===
        absent, absent_type = self.detect_fatigue(face_results)
        is_absent = absent

        if absent:
            current_time = datetime.datetime.now()
            if (current_time - self.last_alert_time).seconds > self.alert_cooldown:
                violations.append({
                    'type': 'absence',
                    'severity': 'critical',
                    'description': f'ABSENCE: {absent_type}',
                    'frame': frame.copy()
                })
                need_screenshot = True
                self.last_alert_time = current_time

        # === Детекция искривления позвоночника ===
        spine_violation, angle, severity, spine_screenshot = self.detect_spine_curvature(
            pose_results.pose_landmarks
        )
        spine_angle = angle
        spine_severity = severity
        if spine_screenshot:
            need_screenshot = True

        if spine_violation:
            current_time = datetime.datetime.now()
            seconds = self.spine_curvature_frames / 20

            if spine_severity and (current_time - self.last_alert_time).seconds > self.alert_cooldown:
                if spine_severity == 'critical':
                    desc = f'Spine CRITICAL: {angle:.1f}° ({seconds:.1f}s)'
                elif spine_severity == 'warning':
                    desc = f'Spine WARNING: {angle:.1f}° ({seconds:.1f}s)'
                else:
                    desc = f'Spine: {angle:.1f}°'

                violations.append({
                    'type': 'spine_curvature',
                    'severity': spine_severity,
                    'description': desc,
                    'frame': frame.copy()
                })
                self.last_alert_time = current_time

        # === Детекция поворота головы ===
        head_violation, direction, head_sev, head_screenshot = self.detect_head_turn(
            pose_results.pose_landmarks
        )
        head_direction = direction
        head_severity = head_sev
        if head_screenshot:
            need_screenshot = True

        if head_violation:
            current_time = datetime.datetime.now()
            seconds = self.head_turn_frames / 20

            if head_severity and (current_time - self.last_alert_time).seconds > self.alert_cooldown:
                if head_severity == 'critical':
                    desc = f'Head CRITICAL: turned {direction} ({seconds:.1f}s)'
                elif head_severity == 'warning':
                    desc = f'Head WARNING: turned {direction} ({seconds:.1f}s)'
                else:
                    desc = f'Head turned {direction}'

                violations.append({
                    'type': 'head_turn',
                    'severity': head_severity,
                    'description': desc,
                    'frame': frame.copy()
                })
                self.last_alert_time = current_time

        # Проверка сонливости (КРИТИЧЕСКОЕ)
        if face_mesh_results and face_mesh_results.multi_face_landmarks:
            sleepy, sleepy_type = self.detect_sleepy_eyes(
                face_mesh_results.multi_face_landmarks[0]
            )
            if sleepy:
                current_time = datetime.datetime.now()
                if (current_time - self.last_alert_time).seconds > self.alert_cooldown:
                    violations.append({
                        'type': 'sleepy',
                        'severity': 'critical',
                        'description': f'Sleepy: {sleepy_type}',
                        'frame': frame.copy()
                    })
                    need_screenshot = True
                    self.last_alert_time = current_time

        # === ВИЗУАЛИЗАЦИЯ ===
        annotated_frame = frame.copy()
        h, w, _ = annotated_frame.shape

        # Рисуем скелет (только если есть)
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Рисуем линию между плечами
            left_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            lx = int(left_shoulder.x * w)
            ly = int(left_shoulder.y * h)
            rx = int(right_shoulder.x * w)
            ry = int(right_shoulder.y * h)

            # Цвет линии плеч
            if spine_severity == 'critical':
                cv2.line(annotated_frame, (lx, ly), (rx, ry), (0, 0, 255), 5)
            elif spine_severity == 'warning':
                cv2.line(annotated_frame, (lx, ly), (rx, ry), (0, 165, 255), 4)
            elif spine_violation:
                cv2.line(annotated_frame, (lx, ly), (rx, ry), (0, 255, 255), 3)
            else:
                cv2.line(annotated_frame, (lx, ly), (rx, ry), (0, 255, 255), 2)

            # Рисуем линию от центра плеч до носа (для визуализации поворота головы)
            nose = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            nx = int(nose.x * w)
            ny = int(nose.y * h)
            cx = int((lx + rx) / 2)
            cy = int((ly + ry) / 2)

            if head_violation:
                if head_severity == 'critical':
                    cv2.line(annotated_frame, (cx, cy), (nx, ny), (0, 0, 255), 3)
                elif head_severity == 'warning':
                    cv2.line(annotated_frame, (cx, cy), (nx, ny), (0, 165, 255), 3)
                else:
                    cv2.line(annotated_frame, (cx, cy), (nx, ny), (0, 255, 255), 2)
            else:
                cv2.line(annotated_frame, (cx, cy), (nx, ny), (0, 255, 0), 2)

        # Рисуем лицо
        if face_results and face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Полупрозрачный фон для текста
        overlay = annotated_frame.copy()

        # === СТАТУС ===
        # Определяем самый высокий уровень опасности
        if is_absent:
            status_text = "DANGER!"
            color = (0, 0, 255)
        elif spine_severity == 'critical' or head_severity == 'critical':
            status_text = "CRITICAL!"
            color = (0, 0, 255)
        elif spine_severity == 'warning' or head_severity == 'warning':
            status_text = "WARNING!"
            color = (0, 165, 255)
        elif spine_violation or head_violation:
            status_text = "DANGER!"
            color = (0, 255, 255)
        else:
            has_critical = any(v['severity'] == 'critical' for v in violations)
            if has_critical:
                status_text = "CRITICAL!"
                color = (0, 0, 255)
            else:
                status_text = "STATUS: OK" if not violations else "ALERT!"
                color = (0, 255, 0) if not violations else (0, 0, 255)

        cv2.rectangle(overlay, (5, 5), (350, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
        cv2.putText(annotated_frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # === ИНФОРМАЦИЯ ===
        cv2.rectangle(overlay, (5, 50), (350, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)

        info_y = 70

        # ABSENCE статус (если оператора нет)
        if is_absent:
            cv2.putText(annotated_frame, "OPERATOR: ABSENT", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            info_y += 20

        # SPINE статус
        if spine_severity == 'critical':
            cv2.putText(annotated_frame, "SPINE: CRITICAL", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif spine_severity == 'warning':
            cv2.putText(annotated_frame, "SPINE: WARNING", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        elif spine_violation:
            cv2.putText(annotated_frame, "SPINE: DANGER", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(annotated_frame, "SPINE: OK", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        info_y += 20

        # HEAD статус
        if head_severity == 'critical':
            cv2.putText(annotated_frame, f"HEAD: CRITICAL ({direction})", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif head_severity == 'warning':
            cv2.putText(annotated_frame, f"HEAD: WARNING ({direction})", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        elif head_violation:
            cv2.putText(annotated_frame, f"HEAD: TURNED {direction}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(annotated_frame, "HEAD: OK", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        info_y += 20

        # FACE статус
        if face_results and face_results.detections:
            cv2.putText(annotated_frame, "FACE: DETECTED", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(annotated_frame, "FACE: NOT DETECTED", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        info_y += 20

        # Время нарушений
        if spine_violation and self.spine_curvature_frames > 0:
            seconds = self.spine_curvature_frames / 20
            color_text = (0, 0, 255) if spine_severity == 'critical' else (0, 165,
                                                                           255) if spine_severity == 'warning' else (0,
                                                                                                                     255,
                                                                                                                     255)
            cv2.putText(annotated_frame, f"SPINE TIME: {seconds:.1f}s", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_text, 1)
            info_y += 15

        if head_violation and self.head_turn_frames > 0:
            seconds = self.head_turn_frames / 20
            color_text = (0, 0, 255) if head_severity == 'critical' else (0, 165,
                                                                          255) if head_severity == 'warning' else (0,
                                                                                                                   255,
                                                                                                                   255)
            cv2.putText(annotated_frame, f"HEAD TIME: {seconds:.1f}s", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_text, 1)
            info_y += 20

        # EAR
        if face_mesh_results and face_mesh_results.multi_face_landmarks:
            ear = self.calculate_ear(face_mesh_results.multi_face_landmarks[0])
            ear_y = info_y + 10
            cv2.rectangle(overlay, (5, ear_y - 5), (200, ear_y + 25), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
            cv2.putText(annotated_frame, f"EAR: {ear:.2f}", (10, ear_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255) if ear < self.ear_threshold else (255, 255, 255), 1)
            info_y += 30

        # Угол плеч
        if spine_angle is not None:
            angle_y = info_y + 5
            cv2.rectangle(overlay, (5, angle_y - 5), (250, angle_y + 25), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)

            if spine_severity == 'critical':
                cv2.putText(annotated_frame, f"ANGLE: {spine_angle:.1f}°", (10, angle_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif spine_severity == 'warning':
                cv2.putText(annotated_frame, f"ANGLE: {spine_angle:.1f}°", (10, angle_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                cv2.putText(annotated_frame, f"ANGLE: {spine_angle:.1f}°", (10, angle_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return annotated_frame, violations, need_screenshot


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager()
        self.monitor = SafetyMonitor(self.db_manager)

        self.camera = None
        self.camera_id = 0
        self.is_running = False
        self.current_frame = None

        self.init_ui()
        self.init_timer()

    def init_ui(self):
        self.setWindowTitle("Operator Safety System - Spine + Head Detection")
        self.setGeometry(100, 100, 1400, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Левая панель - видео
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid #333; background: #1a1a1a;")
        self.video_label.setText("Camera ready")
        left_layout.addWidget(self.video_label)

        # Панель управления
        controls_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.toggle_camera)
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        controls_layout.addWidget(self.start_btn)

        self.capture_btn = QPushButton("Screenshot")
        self.capture_btn.clicked.connect(self.capture_screenshot)
        self.capture_btn.setEnabled(False)
        controls_layout.addWidget(self.capture_btn)

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        controls_layout.addWidget(self.camera_combo)

        left_layout.addLayout(controls_layout)

        # Правая панель
        right_panel = QWidget()
        right_panel.setMaximumWidth(400)
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Статус
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)

        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
        status_layout.addWidget(self.status_label)

        self.fps_label = QLabel("FPS: 0")
        status_layout.addWidget(self.fps_label)

        right_layout.addWidget(status_group)

        # Уведомления
        notif_group = QGroupBox("Notifications")
        notif_layout = QVBoxLayout()
        notif_group.setLayout(notif_layout)

        self.notification_text = QTextEdit()
        self.notification_text.setReadOnly(True)
        self.notification_text.setMaximumHeight(100)
        self.notification_text.setStyleSheet("QTextEdit { background-color: #2b2b2b; color: #ffffff; }")
        notif_layout.addWidget(self.notification_text)

        right_layout.addWidget(notif_group)

        # Статистика
        stats_group = QGroupBox("Session Statistics")
        stats_layout = QGridLayout()
        stats_group.setLayout(stats_layout)

        self.total_violations_label = QLabel("0")
        self.total_violations_label.setFont(QFont("Arial", 14, QFont.Bold))
        stats_layout.addWidget(QLabel("Total violations:"), 0, 0)
        stats_layout.addWidget(self.total_violations_label, 0, 1)

        self.warning_count_label = QLabel("0")
        self.warning_count_label.setFont(QFont("Arial", 14, QFont.Bold))
        stats_layout.addWidget(QLabel("Warnings:"), 1, 0)
        stats_layout.addWidget(self.warning_count_label, 1, 1)

        self.critical_count_label = QLabel("0")
        self.critical_count_label.setFont(QFont("Arial", 14, QFont.Bold))
        stats_layout.addWidget(QLabel("Critical:"), 2, 0)
        stats_layout.addWidget(self.critical_count_label, 2, 1)

        right_layout.addWidget(stats_group)

        # Кнопка "Очистить статистику"
        clear_btn = QPushButton("🗑 Clear Statistics")
        clear_btn.clicked.connect(self.clear_statistics)
        clear_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; }")
        right_layout.addWidget(clear_btn)

        # История
        history_group = QGroupBox("Incident History")
        history_layout = QVBoxLayout()
        history_group.setLayout(history_layout)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["Time", "Type", "Severity", "Description"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setAlternatingRowColors(True)
        history_layout.addWidget(self.history_table)

        refresh_btn = QPushButton("Refresh History")
        refresh_btn.clicked.connect(self.refresh_history)
        history_layout.addWidget(refresh_btn)

        right_layout.addWidget(history_group)

        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)

    def clear_statistics(self):
        """Очистка статистики и предупреждений"""
        reply = QMessageBox.question(
            self,
            "Clear Statistics",
            "Are you sure you want to clear all statistics and notifications?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Очищаем базу данных
            if self.db_manager.clear_incidents():
                # Очищаем таблицу
                self.history_table.setRowCount(0)
                # Обнуляем счетчики
                self.violation_count = 0
                self.warning_count = 0
                self.critical_count = 0
                self.update_stats_display()
                # Очищаем уведомления
                self.notification_text.clear()
                QMessageBox.information(self, "Success", "Statistics cleared successfully!")
            else:
                QMessageBox.warning(self, "Error", "Failed to clear statistics!")

    def init_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(50)

        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_stats)
        self.fps_timer.setInterval(1000)
        self.fps_timer.start()

        self.frame_count = 0
        self.violation_count = 0
        self.warning_count = 0
        self.critical_count = 0

    def toggle_camera(self):
        if not self.is_running:
            try:
                self.camera_id = self.camera_combo.currentIndex()
                self.camera = cv2.VideoCapture(self.camera_id)
                if not self.camera.isOpened():
                    raise Exception("Could not open camera")

                self.is_running = True
                self.start_btn.setText("Stop Camera")
                self.start_btn.setStyleSheet(
                    "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; }")
                self.capture_btn.setEnabled(True)
                self.status_label.setText("Recording...")
                self.timer.start()
                self.violation_count = 0
                self.warning_count = 0
                self.critical_count = 0
                self.update_stats_display()

                QMessageBox.information(self, "Success", "Camera started successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not start camera: {e}")
        else:
            self.stop_camera()

    def stop_camera(self):
        self.is_running = False
        self.timer.stop()
        if self.camera:
            self.camera.release()
            self.camera = None
        self.start_btn.setText("Start Camera")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.capture_btn.setEnabled(False)
        self.status_label.setText("Stopped")
        self.video_label.clear()
        self.video_label.setText("Camera off")

    def update_frame(self):
        if not self.is_running or self.camera is None:
            return

        ret, frame = self.camera.read()
        if not ret:
            return

        self.frame_count += 1
        annotated_frame, violations, need_screenshot = self.monitor.process_frame(frame)
        self.current_frame = annotated_frame

        for violation in violations:
            self.violation_count += 1
            if violation['severity'] == 'critical':
                self.critical_count += 1
            elif violation['severity'] == 'warning':
                self.warning_count += 1

            self.db_manager.log_incident(
                violation['type'],
                violation['severity'],
                violation['description']
            )

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            notification = f"[{timestamp}] {violation['description']} (Severity: {violation['severity']})"
            self.notification_text.append(notification)

        # Скриншот при критическом нарушении
        if need_screenshot:
            self.capture_screenshot()

        self.update_stats_display()
        self.display_frame(annotated_frame)

    def display_frame(self, frame):
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.width(), self.video_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error displaying frame: {e}")

    def capture_screenshot(self):
        if self.current_frame is None:
            return

        try:
            screenshots_dir = Path("screenshots")
            screenshots_dir.mkdir(exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = screenshots_dir / f"screenshot_{timestamp}.jpg"

            cv2.imwrite(str(filename), self.current_frame)
            print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"Error saving screenshot: {e}")

    def update_stats(self):
        if self.is_running:
            fps = self.frame_count
            self.fps_label.setText(f"FPS: {fps}")
            self.frame_count = 0

    def update_stats_display(self):
        self.total_violations_label.setText(str(self.violation_count))
        self.warning_count_label.setText(str(self.warning_count))
        self.critical_count_label.setText(str(self.critical_count))

    def refresh_history(self):
        incidents = self.db_manager.get_incidents(50)
        self.history_table.setRowCount(len(incidents))

        for i, incident in enumerate(incidents):
            timestamp, inc_type, severity, description, screenshot = incident
            self.history_table.setItem(i, 0, QTableWidgetItem(timestamp[:16]))
            self.history_table.setItem(i, 1, QTableWidgetItem(inc_type))
            self.history_table.setItem(i, 2, QTableWidgetItem(severity))
            self.history_table.setItem(i, 3, QTableWidgetItem(description[:30] + "..."))

    def closeEvent(self, event):
        self.stop_camera()
        self.db_manager.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    font = QFont("Segoe UI", 9)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()