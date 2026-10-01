# Projet : Alignement d’images drone sur image satellite (OpenGL + OpenCV)

## Objectif
Superposer automatiquement des images drone sur une image satellite en calculant une transformation géométrique robuste entre les deux.

Pipeline cible :
1. Détecter des points d’intérêt (SIFT) sur drone et satellite.
2. Faire le matching des descripteurs.
3. Estimer une homographie robuste (RANSAC).
4. Rectifier l’image drone (`warpPerspective`).
5. Positionner/affiner avec les coordonnées GPS.
6. Afficher la superposition finale dans OpenGL.

---

## État actuel du projet (avancement réel)

### Fait
- Chargement de `dataset/map.jpg` (satellite).
- Chargement des images drone depuis `dataset/frames`.
- Affichage OpenGL de l’image satellite comme texture.
- Interface de sélection des images drone (menu déroulant).
- Superposition manuelle d’une image drone (drag souris).
- Extraction SIFT sur l’image satellite.
- Extraction SIFT sur chaque image drone sélectionnée.
- Matching des descripteurs (BFMatcher + ratio test de Lowe 0.75).
- Estimation d’homographie robuste par RANSAC (`findHomography`).
- Rectification perspective (`warpPerspective`) en coordonnées de la satellite.
- Superposition automatique de la vue drone rectifiée sur la carte.
- Affichage en console du nombre de keypoints et de matches.
- Affichage en console des inliers RANSAC.

### Partiellement fait
- Positionnement par GPS : parsing des coordonnées depuis les noms de fichiers + calibration affine GPS->pixel (quand assez d’images valides), mais pas encore utilisée comme contrainte forte d’optimisation.

### Pas encore implémenté
- Assemblage final multi-images avec fusion/alpha cohérents.
- Métriques qualité (inliers, erreur de reprojection).

---

## Où vous en êtes
Vous êtes à la fin de l’étape **matching local de features** (SIFT + BF + ratio test), avec une visualisation OpenGL déjà opérationnelle.

La prochaine vraie étape de vision est :
- passer de « bons matches » à « transformation fiable » via `findHomography(..., cv::RANSAC)`.

---

## Plan pour compléter le projet

## Étape A — Homographie robuste par image drone
Pour chaque image drone :
- Construire `pts_drone` et `pts_sat` à partir des bons matches.
- Estimer `H` avec RANSAC.
- Conserver aussi le masque d’inliers RANSAC.

Critères minimaux :
- au moins 4 correspondances.
- ratio inliers suffisant (ex: > 25%).

---

## Étape B — Rectification perspective
- Appliquer `cv::warpPerspective(drone, drone_rect, H, sat_size)`.
- Générer un masque binaire de validité (pixels projetés).
- Préparer une couche RGBA pour superposition propre.

---

## Étape C — Initialisation GPS (coarse alignment)
- Parser longitude/latitude depuis les noms de fichiers.
- Définir une transformation simple GPS -> pixel satellite (affine locale).
- Utiliser cette position comme initialisation (ou contrainte) avant affine fine.

---

## Étape D — Affichage final OpenGL
- Afficher satellite en fond.
- Afficher images drone rectifiées (ordre, alpha configurable).
- Ajouter un mode debug :
  - keypoints,
  - matches,
  - inliers,
  - contour projeté de l’image drone.

---

## Étape E — Validation
- Mesurer :
  - nombre de matches,
  - nombre d’inliers,
  - erreur de reprojection.
- Rejeter automatiquement les images mal alignées.

---

## Conseils d’implémentation (OpenCV)
Fonctions utiles :
- `cv::SIFT::create`
- `cv::BFMatcher::knnMatch`
- `cv::findHomography(..., cv::RANSAC, reprojThreshold, mask)`
- `cv::warpPerspective`
- `cv::perspectiveTransform`

Paramètres de départ recommandés :
- Lowe ratio : `0.75`
- RANSAC reprojection threshold : `3.0` à `5.0` pixels
- Inliers minimum : `20` (à ajuster selon dataset)

---

## Structure du projet
```text
.
├── CMakeLists.txt
├── README.MD         # Guide compilation/exécution
├── READMe.MD         # Suivi technique + roadmap vision
├── dataset/
│   ├── map.jpg
│   └── frames/
│       └── @index@lon@lat@.jpg
└── src/
    └── main.cpp
```

---

## Prochaine action concrète
Implémenter dans `src/main.cpp` la génération d’homographie RANSAC + `warpPerspective` pour l’image drone sélectionnée, puis afficher la version rectifiée à la place de la superposition manuelle brute.
