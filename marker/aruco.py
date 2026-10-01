import cv2
import numpy as np

# ===================== CONFIG =====================
MARKER_ID = 25
DICT = cv2.aruco.DICT_4X4_50

# ⚠️ mets la taille RÉELLE du carré noir (sans bord blanc) en mètres:
# ex: 6 cm => 0.06
MARKER_LENGTH_M = 0.06

CAM_INDEX_CANDIDATES = [0, 1, 2]
# ==================================================

def build_default_camera_matrix(width, height, fov_degrees=60.0):
    f = (width / 2.0) / np.tan(np.deg2rad(fov_degrees / 2.0))
    return np.array([[f, 0, width / 2.0],
                     [0, f, height / 2.0],
                     [0, 0, 1]], dtype=np.float32)

def open_camera():
    for idx in CAM_INDEX_CANDIDATES:
        # Linux: CAP_V4L2 aide souvent. Windows/Mac: ça ne gêne pas.
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(idx)
        if cap.isOpened():
            # test read
            ret, _ = cap.read()
            if ret:
                print(f"✅ Camera opened at index {idx}")
                return cap
        cap.release()
    return None

def draw_cube(frame, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # base
    cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
    # pillars
    for i in range(4):
        cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[i + 4]), (255, 0, 0), 2)
    # top
    cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)

    #fill the cube with colors
    cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), -1)
    cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), -1)
    

def cube_points(size):
    s = size
    return np.float32([
        [0, 0, 0],
        [s, 0, 0],
        [s, s, 0],
        [0, s, 0],
        [0, 0, -s],
        [s, 0, -s],
        [s, s, -s],
        [0, s, -s],
    ])

def main():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco introuvable. Installe opencv-contrib-python.")

    print("OpenCV:", cv2.__version__)

    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    cap = open_camera()
    if cap is None:
        raise RuntimeError(" Impossible d’ouvrir la webcam (index 0/1/2). Ferme Zoom/Teams/Discord, etc.")

    # read one frame to init intrinsics
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(" Webcam ouverte mais lecture impossible (permissions ou caméra occupée).")

    h, w = frame.shape[:2]
    camera_matrix = build_default_camera_matrix(w, h)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    # 3D model for marker corners (same order as detected corners)
    marker_3d = np.array([
        [0, 0, 0],
        [MARKER_LENGTH_M, 0, 0],
        [MARKER_LENGTH_M, MARKER_LENGTH_M, 0],
        [0, MARKER_LENGTH_M, 0],
    ], dtype=np.float32)

    cube_3d = cube_points(MARKER_LENGTH_M)

    print("Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for corner, mid in zip(corners, ids.flatten()):
                if int(mid) != MARKER_ID:
                    continue

                # corner shape: (1,4,2) -> (4,2)
                marker_2d = corner[0].astype(np.float32)

                ok, rvec, tvec = cv2.solvePnP(
                    marker_3d, marker_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    continue

                # draw axes (works on most versions; if missing, comment this line)
                if hasattr(cv2, "drawFrameAxes"):
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_M * 0.7)

                imgpts, _ = cv2.projectPoints(cube_3d, rvec, tvec, camera_matrix, dist_coeffs)
                draw_cube(frame, imgpts)

        cv2.imshow("Aruco Cube", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()