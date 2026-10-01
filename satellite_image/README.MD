# Execution du projet (C++ / CMake)

Ce projet affiche l'image satellite comme texture OpenGL, puis superpose une image drone que l'on peut deplacer a la souris.
Il calcule aussi automatiquement une homographie (SIFT + RANSAC) pour afficher une version rectifiee et alignee sur la carte.

## 1. Prerequis

- CMake (>= 3.12)
- Un compilateur C++17 (g++ / clang++)
- OpenCV (C++) 
- OpenGL
- GLUT (ex: freeglut)

## 2. Compilation

Depuis la racine du projet :

```bash
cmake -S . -B build
cmake --build build
```

## 3. Execution

```bash
./build/satellite_viewer
```

## 4. Controles

- Menu fixe a droite : clic gauche sur le bouton pour ouvrir la liste, puis clic gauche pour selectionner
- Glisser l'image drone : clic gauche + deplacement de la souris
- Basculer vue rectifiee/raw : `R`
- Defiler la liste : molette de la souris
- Quitter : `Q` ou `ESC`

## 5. Structure du projet

```
.
├── CMakeLists.txt
├── README.MD
├── READMe.MD
├── dataset/
│   ├── map.jpg
│   └── frames/
│       ├── ... images drone ...
└── src/
    └── main.cpp
```
