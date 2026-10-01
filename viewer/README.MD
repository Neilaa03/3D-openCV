# AR 3D Object Viewer with ArUco and OpenCV

## Project idea

This project is a Python application that uses the camera and **OpenCV + ArUco markers** to simulate an **augmented reality 3D object viewer**.

The app detects an **ArUco marker** in the webcam stream, then displays a **3D object** on top of that marker.  
The user can choose which object to display from a small list of 3D models.

The goal is similar to a simple AR figurine viewer: when the marker is visible, the selected object appears attached to it.

---

## Main features

- Detect an ArUco marker in real time
- Estimate the marker position and orientation
- Overlay the 3D objects on the detected marker one
- Display next object by user click
- Use OpenCV camera stream for simulation

---

