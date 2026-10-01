import os
import ctypes
from pathlib import Path

# Force legacy OpenGL path for fixed-function calls
os.environ.setdefault("PYGLET_LEGACY_OPENGL", "1")

import cv2
import numpy as np
import random
import trimesh
import pyglet
from pyglet import gl

# Minimal AR 3D car game: ArUco + OpenCV background + OpenGL model

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = None  # set to an int (e.g., 25) to lock to one marker
MARKER_LENGTH_M = 0.06
FLIP_HORIZONTAL = False

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
SUPPORTED_EXTS = {".obj", ".ply", ".glb", ".gltf"}


def pick_first_model():
    if SRC_DIR.exists():
        for p in sorted(SRC_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                return p
    raise RuntimeError("No models found in viewer/src")


def build_default_camera_matrix(width, height, fov_degrees=60.0):
    f = (width / 2.0) / np.tan(np.deg2rad(fov_degrees / 2.0))
    return np.array([[f, 0, width / 2.0],
                     [0, f, height / 2.0],
                     [0, 0, 1]], dtype=np.float32)


def load_mesh(model_path):
    mesh = trimesh.load(model_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.is_empty:
        raise RuntimeError("Mesh is empty")

    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    n = np.asarray(mesh.vertex_normals, dtype=np.float32) if hasattr(mesh, "vertex_normals") else None

    # Normalize model to marker size and center
    center = (v.min(axis=0) + v.max(axis=0)) / 2.0
    v = v - center
    v[:, 2] -= v[:, 2].min()

    ext = v.max(axis=0) - v.min(axis=0)
    scale = (MARKER_LENGTH_M * 0.8) / max(ext[0], ext[1], ext[2], 1e-6)
    v *= scale

    # Move onto marker center and lift a bit
    v[:, 0] += MARKER_LENGTH_M / 2.0
    v[:, 1] += MARKER_LENGTH_M / 2.0
    v[:, 2] += MARKER_LENGTH_M * 0.02

    colors = None
    if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) == len(v):
        vc = np.asarray(mesh.visual.vertex_colors, dtype=np.float32) / 255.0
        colors = vc[:, :3]

    return v, f, n, colors


def rvec_tvec_to_view_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3:] = t

    flip = np.eye(4, dtype=np.float32)
    flip[1, 1] = -1
    flip[2, 2] = -1

    M = flip @ M
    return M


class ARCarWindow(pyglet.window.Window):
    def __init__(self, width, height, vertices, faces, normals, colors, camera_matrix, dist_coeffs):
        super().__init__(width=width, height=height, caption="AR Car Game", resizable=False)

        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.colors = colors
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.keys = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.keys)

        # OpenCV capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.parameters = cv2.aruco.DetectorParameters()
        detector = getattr(cv2.aruco, "ArucoDetector", None)
        self.aruco_detector = detector(self.dictionary, self.parameters) if detector else None

        # Marker 3D corners
        self.objp = np.array([
            [0, 0, 0],
            [MARKER_LENGTH_M, 0, 0],
            [MARKER_LENGTH_M, MARKER_LENGTH_M, 0],
            [0, MARKER_LENGTH_M, 0],
        ], dtype=np.float32)

        # OpenGL setup
        gl.glEnable(gl.GL_DEPTH_TEST)
        

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glShadeModel(gl.GL_SMOOTH)
        gl.glClearColor(0, 0, 0, 1)

        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(0.5, 1.0, 1.0, 0.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (gl.GLfloat * 4)(0.2, 0.2, 0.2, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (gl.GLfloat * 4)(0.9, 0.9, 0.9, 1.0))

        # texture for webcam frame
        self.texture_id = gl.GLuint()
        gl.glGenTextures(1, self.texture_id)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        # car movement in marker plane (x, y) and yaw in degrees
        self.offset = np.array([0.0, 0.0], dtype=np.float32)
        self.yaw = 0.0
        self.speed = MARKER_LENGTH_M * 0.8
        self.turn_speed = 90.0

        self.rvec = None
        self.tvec = None
        self.frame = None

        # simple game state
        self.score = 0
        self.coins = []
        self._spawn_coins(5)

        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

    def _spawn_coins(self, n):
        limit = MARKER_LENGTH_M * 0.4
        for _ in range(n):
            x = random.uniform(-limit, limit)
            y = random.uniform(-limit, limit)
            self.coins.append(np.array([x, y], dtype=np.float32))

    def _collect_coins(self, limit):
        if not self.coins:
            return
        pos = self.offset
        keep = []
        for c in self.coins:
            if np.linalg.norm(c - pos) < MARKER_LENGTH_M * 0.08:
                self.score += 1
                x = random.uniform(-limit, limit)
                y = random.uniform(-limit, limit)
                keep.append(np.array([x, y], dtype=np.float32))
            else:
                keep.append(c)
        self.coins = keep

    def _draw_coin(self, x, y, r):
        gl.glColor3f(1.0, 0.85, 0.0)
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glVertex3f(x, y, 0.001)
        steps = 16
        for i in range(steps + 1):
            a = (i / steps) * 2.0 * np.pi
            gl.glVertex3f(x + r * np.cos(a), y + r * np.sin(a), 0.001)
        gl.glEnd()

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        if FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        self.frame = frame

        # input update
        if self.keys[pyglet.window.key.LEFT] or self.keys[pyglet.window.key.A]:
            self.yaw += self.turn_speed * dt
        if self.keys[pyglet.window.key.RIGHT] or self.keys[pyglet.window.key.D]:
            self.yaw -= self.turn_speed * dt
        if self.keys[pyglet.window.key.UP] or self.keys[pyglet.window.key.W]:
            self.offset[0] += np.cos(np.deg2rad(self.yaw)) * self.speed * dt
            self.offset[1] += np.sin(np.deg2rad(self.yaw)) * self.speed * dt
        if self.keys[pyglet.window.key.DOWN] or self.keys[pyglet.window.key.S]:
            self.offset[0] -= np.cos(np.deg2rad(self.yaw)) * self.speed * dt
            self.offset[1] -= np.sin(np.deg2rad(self.yaw)) * self.speed * dt

        # keep car near marker center
        limit = MARKER_LENGTH_M * 0.4
        self.offset[0] = float(np.clip(self.offset[0], -limit, limit))
        self.offset[1] = float(np.clip(self.offset[1], -limit, limit))
        self._collect_coins(limit)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

        self.rvec, self.tvec = None, None
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            chosen = None
            if MARKER_ID is None:
                chosen = 0
            else:
                for i, mid in enumerate(ids.flatten()):
                    if int(mid) == int(MARKER_ID):
                        chosen = i
                        break
            if chosen is not None:
                c = corners[chosen]
                img_corners = c[0].astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(self.objp, img_corners, self.camera_matrix, self.dist_coeffs)
                if ok:
                    self.rvec, self.tvec = rvec, tvec
                    mid = int(ids.flatten()[chosen])
                    cv2.putText(frame, f"Marker {mid} | Score {self.score}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Marker ID not found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No marker", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def on_draw(self):
        self.clear()

        # draw webcam background
        if self.frame is None:
            return

        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGB,
            w, h, 0,
            gl.GL_RGB, gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(rgb.ctypes.data)
        )

        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, w, 0, h, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_TEXTURE_2D)

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 1); gl.glVertex2f(0, 0)
        gl.glTexCoord2f(1, 1); gl.glVertex2f(w, 0)
        gl.glTexCoord2f(1, 0); gl.glVertex2f(w, h)
        gl.glTexCoord2f(0, 0); gl.glVertex2f(0, h)
        gl.glEnd()

        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)

        if self.rvec is None:
            return

        # projection from camera intrinsics
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        near, far = 0.01, 10.0
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 2 * fx / w
        proj[1, 1] = 2 * fy / h
        proj[0, 2] = 1 - 2 * cx / w
        proj[1, 2] = 2 * cy / h - 1
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = -1

        view = rvec_tvec_to_view_matrix(self.rvec, self.tvec)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadMatrixf((gl.GLfloat * 16)(*proj.T.flatten()))
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadMatrixf((gl.GLfloat * 16)(*view.T.flatten()))

        # draw coins in marker space
        for c in self.coins:
            self._draw_coin(c[0], c[1], MARKER_LENGTH_M * 0.05)

        # car transform within marker coordinates
        gl.glPushMatrix()
        gl.glTranslatef(self.offset[0], self.offset[1], 0.0)
        gl.glRotatef(self.yaw, 0, 0, 1)

        # draw filled triangles
        gl.glBegin(gl.GL_TRIANGLES)
        for a, b, c in self.faces:
            if self.colors is not None:
                gl.glColor3f(*self.colors[a])
            else:
                gl.glColor3f(0.9, 0.1, 0.1)
            if self.normals is not None:
                gl.glNormal3f(*self.normals[a])
            gl.glVertex3f(*self.vertices[a])
            if self.colors is not None:
                gl.glColor3f(*self.colors[b])
            gl.glVertex3f(*self.vertices[b])
            if self.colors is not None:
                gl.glColor3f(*self.colors[c])
            gl.glVertex3f(*self.vertices[c])
        gl.glEnd()
        gl.glPopMatrix()

    def on_close(self):
        self.cap.release()
        super().on_close()


def main():
    model_path = pick_first_model()
    vertices, faces, normals, colors = load_mesh(model_path)
    print(f"Loaded model: {model_path.name}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read webcam frame")

    h, w = frame.shape[:2]
    camera_matrix = build_default_camera_matrix(w, h)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    win = ARCarWindow(w, h, vertices, faces, normals, colors, camera_matrix, dist_coeffs)
    pyglet.app.run()


if __name__ == "__main__":
    main()
