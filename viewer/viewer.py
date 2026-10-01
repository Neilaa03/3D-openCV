import cv2
import numpy as np
import trimesh
from pathlib import Path

# Minimal AR 3D viewer using OpenCV + ArUco + trimesh

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_LENGTH_M = 0.06  # size of the black square side in meters

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

SUPPORTED_EXTS = {".obj", ".ply", ".glb", ".gltf"}


def collect_models():
    models = []
    if SRC_DIR.exists():
        for p in sorted(SRC_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                models.append(p)
    if not models:
        # fallback to repo root samples
        root = BASE_DIR.parent
        for name in ("avion.obj", "harry.obj", "harry.ply"):
            p = root / name
            if p.exists():
                models.append(p)
    return models


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

    # Center and scale to marker size
    center = (v.min(axis=0) + v.max(axis=0)) / 2.0
    v = v - center
    v[:, 2] -= v[:, 2].min()

    ext = v.max(axis=0) - v.min(axis=0)
    scale = (MARKER_LENGTH_M * 0.8) / max(ext[0], ext[1], 1e-6)
    v *= scale

    # Move onto marker center and lift slightly
    v[:, 0] += MARKER_LENGTH_M / 2.0
    v[:, 1] += MARKER_LENGTH_M / 2.0
    v[:, 2] += MARKER_LENGTH_M * 0.02

    return v, f


def draw_mesh_filled(img, vertices, faces, rvec, tvec, camera_matrix, dist_coeffs):
    if faces.size == 0:
        return
    pts2d, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
    pts2d = pts2d.reshape(-1, 2).astype(int)
    h, w = img.shape[:2]

    # build per-face average depth for a simple painter's algorithm
    zv = vertices[:, 2]
    face_depth = (zv[faces[:, 0]] + zv[faces[:, 1]] + zv[faces[:, 2]]) / 3.0
    order = np.argsort(face_depth)  # far -> near (after sort ascending)

    step = max(1, len(faces) // 2000)
    base_color = (0, 200, 0)
    edge_color = (20, 255, 20)

    for i in range(0, len(order), step):
        fi = order[i]
        a, b, c = faces[fi]
        tri = np.array([pts2d[a], pts2d[b], pts2d[c]], dtype=np.int32)

        # skip if triangle fully outside
        if np.all(tri[:, 0] < 0) or np.all(tri[:, 0] >= w) or np.all(tri[:, 1] < 0) or np.all(tri[:, 1] >= h):
            continue

        # simple face shading based on 2D area sign
        area = (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1]) - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0])
        if area < 0:
            color = (0, 140, 0)
        else:
            color = base_color

        cv2.fillConvexPoly(img, tri, color, cv2.LINE_AA)
        cv2.polylines(img, [tri], True, edge_color, 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read webcam frame")

    h, w = frame.shape[:2]
    camera_matrix = build_default_camera_matrix(w, h)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = getattr(cv2.aruco, "ArucoDetector", None)
    aruco_detector = detector(dictionary, parameters) if detector else None

    model_paths = collect_models()
    if not model_paths:
        raise RuntimeError("No models found. Put .obj/.ply/.glb in viewer/src/")

    model_index = 0
    vertices, faces = load_mesh(model_paths[model_index])

    objp = np.array([
        [0, 0, 0],
        [MARKER_LENGTH_M, 0, 0],
        [MARKER_LENGTH_M, MARKER_LENGTH_M, 0],
        [0, MARKER_LENGTH_M, 0],
    ], dtype=np.float32)

    window_name = "Minimal AR Viewer"

    def on_mouse(event, x, y, flags, param):
        nonlocal model_index, vertices, faces
        if event == cv2.EVENT_LBUTTONDOWN:
            model_index = (model_index + 1) % len(model_paths)
            vertices, faces = load_mesh(model_paths[model_index])

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if aruco_detector:
            corners, ids, _ = aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for c, mid in zip(corners, ids.flatten()):
                img_corners = c[0].astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(objp, img_corners, camera_matrix, dist_coeffs)
                if ok:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_M * 0.5)
                    draw_mesh_filled(frame, vertices, faces, rvec, tvec, camera_matrix, dist_coeffs)
                    break

        cv2.putText(frame, f"Model: {model_paths[model_index].name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Click to change model | q to quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("n"):
            model_index = (model_index + 1) % len(model_paths)
            vertices, faces = load_mesh(model_paths[model_index])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
