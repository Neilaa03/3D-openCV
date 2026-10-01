#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>
#include <cstdint>
#include <map>
#include <iostream>
#include <algorithm>
#include <GL/glut.h>

struct Point{
    double x, y, z;
    float r, g, b;
};

std::vector<Point> nuage;
double Xmin, Ymin, Zmin, Xmax, Ymax, Zmax;
Point barycentre;
double scaleAuto = 1.0;
int damierN = 20;
double damierCell = 1.0;
int damierNx = 1;
int damierNy = 1;
int terrainN = 180;
float rotX = 20.0f;
float rotY = -30.0f;
float zoom = 1.0f;
int lastX = 0, lastY = 0;
bool dragging = false;
bool useHeatmap = true;

enum RenderMode {
    MODE_POINTS = 0,
    MODE_CARREE = 1,
    MODE_TRIANGULATION = 2,
    MODE_CONTOUR = 3
};

RenderMode renderMode = MODE_TRIANGULATION;
int winW = 640, winH = 480;
const int sidebarW = 240;

struct Cell {
    double sumZ = 0.0;
    double sumR = 0.0;
    double sumG = 0.0;
    double sumB = 0.0;
    int count = 0;
    double z = 0.0;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    bool valid = false;
};

void HeatmapColor(double z, float& r, float& g, float& b)
{
    double t = 0.0;
    if (Zmax > Zmin) t = (z - Zmin) / (Zmax - Zmin);
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;

    // Gradient: bleu -> cyan -> vert -> jaune -> rouge
    if (t < 0.25) {
        double u = t / 0.25;
        r = 0.0f;
        g = (float)u;
        b = 1.0f;
    } else if (t < 0.5) {
        double u = (t - 0.25) / 0.25;
        r = 0.0f;
        g = 1.0f;
        b = (float)(1.0 - u);
    } else if (t < 0.75) {
        double u = (t - 0.5) / 0.25;
        r = (float)u;
        g = 1.0f;
        b = 0.0f;
    } else {
        double u = (t - 0.75) / 0.25;
        r = 1.0f;
        g = (float)(1.0 - u);
        b = 0.0f;
    }
}

bool ChargerNuage(const char* filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    double a,b,c,d,e,f,g,h;
    struct Accum {
        double sumZ;
        double sumR;
        double sumG;
        double sumB;
        int count;
    };
    std::map<std::pair<double,double>, Accum> acc;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        std::istringstream iss(line);
        if (!(iss >> a >> b >> c >> d >> e >> f >> g >> h)) continue;

        // Regroupement des doublons (même x,y) et moyenne du z
        auto key = std::make_pair(a, b);
        auto& v = acc[key];
        v.sumZ += c;
        v.sumR += d / 255.0;
        v.sumG += e / 255.0;
        v.sumB += f / 255.0;
        v.count += 1;
    }
    nuage.clear();
    nuage.reserve(acc.size());
    for (const auto& it : acc) {
        const auto& key = it.first;
        const auto& v = it.second;
        Point p;
        p.x = key.first;
        p.y = key.second;
        if (v.count > 0) {
            double inv = 1.0 / v.count;
            p.z = v.sumZ * inv;
            p.r = static_cast<float>(v.sumR * inv);
            p.g = static_cast<float>(v.sumG * inv);
            p.b = static_cast<float>(v.sumB * inv);
        } else {
            p.z = 0.0;
            p.r = p.g = p.b = 0.0f;
        }
        nuage.push_back(p);
    }
    return true;
}

void CalculerMinMax()
{
    Xmin = Ymin = Zmin = std::numeric_limits<double>::infinity();
    Xmax = Ymax = Zmax = -std::numeric_limits<double>::infinity();

    double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
    for (const auto& p : nuage)
    {
        if (p.x < Xmin) Xmin = p.x;
        if (p.y < Ymin) Ymin = p.y;
        if (p.z < Zmin) Zmin = p.z;

        if (p.x > Xmax) Xmax = p.x;
        if (p.y > Ymax) Ymax = p.y;
        if (p.z > Zmax) Zmax = p.z;

        sumX += p.x;
        sumY += p.y;
        sumZ += p.z;
    }

    if (!nuage.empty()) {
        barycentre.x = sumX / nuage.size();
        barycentre.y = sumY / nuage.size();
        barycentre.z = sumZ / nuage.size();
        double dx = Xmax - Xmin;
        double dy = Ymax - Ymin;
        double dz = Zmax - Zmin;
        double dmax = std::max(dx, std::max(dy, dz));
        if (dmax > 0.0) {
            double target = 4; // taille voulue dans la vue
            scaleAuto = dmax / target;
        } else {
            scaleAuto = 1.0;
        }
    } else {
        barycentre.x = barycentre.y = barycentre.z = 0.0;
        scaleAuto = 1.0;
    }
}

void Reshape(int w, int h)
{
    if (h == 0) h = 1;
    winW = w;
    winH = h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)w / (double)h, 0.1, 100.0);
}

void DrawText2D(int x, int y, const char* s)
{
    glRasterPos2i(x, y);
    while (*s) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *s++);
    }
}

void DessinerSidebar()
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, winH, 0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    glColor3f(0.10f, 0.10f, 0.12f);
    glBegin(GL_QUADS);
    glVertex2i(0, 0); glVertex2i(sidebarW, 0); glVertex2i(sidebarW, winH); glVertex2i(0, winH);
    glEnd();

    const int x0 = 16;
    const int y0 = 20;
    const int h = 34;
    const int w = sidebarW - 32;
    const int gap = 10;

    glColor3f(1.0f, 1.0f, 1.0f);
    DrawText2D(16, 26, "Toolbox");

    auto DrawBtn = [&](int idx, const char* label, bool active) {
        int y = y0 + 20 + idx * (h + gap);
        if (active) glColor3f(0.20f, 0.45f, 0.80f);
        else glColor3f(0.22f, 0.22f, 0.24f);
        glBegin(GL_QUADS);
        glVertex2i(x0, y); glVertex2i(x0 + w, y); glVertex2i(x0 + w, y + h); glVertex2i(x0, y + h);
        glEnd();
        glColor3f(1.0f, 1.0f, 1.0f);
        DrawText2D(x0 + 10, y + 22, label);
    };

    DrawBtn(0, "1) Points", renderMode == MODE_POINTS);
    DrawBtn(1, "2) Carres / Quads", renderMode == MODE_CARREE);
    DrawBtn(2, "3) Triangulation Smooth", renderMode == MODE_TRIANGULATION);
    DrawBtn(3, "4) Courbes de niveau", renderMode == MODE_CONTOUR);
    DrawBtn(4, useHeatmap ? "H) Heatmap: ON" : "H) Heatmap: OFF", useHeatmap);
    DrawBtn(5, "Reset View", false);

    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void DessinerBoite()
{
    glColor3f(1,1,1);
    glBegin(GL_LINES);

    auto vx = [&](double x, double y, double z){
        //glVertex3f(x - barycentre.x, y - barycentre.y, z - barycentre.z);
        glVertex3f(x - barycentre.x,
           z - barycentre.z,  // Z devient vertical
           y - barycentre.y);
    };

    // face bas (Zmin)
    vx(Xmin, Ymin, Zmin); vx(Xmax, Ymin, Zmin);
    vx(Xmax, Ymin, Zmin); vx(Xmax, Ymax, Zmin);
    vx(Xmax, Ymax, Zmin); vx(Xmin, Ymax, Zmin);
    vx(Xmin, Ymax, Zmin); vx(Xmin, Ymin, Zmin);

    // face haut (Zmax)
    vx(Xmin, Ymin, Zmax); vx(Xmax, Ymin, Zmax);
    vx(Xmax, Ymin, Zmax); vx(Xmax, Ymax, Zmax);
    vx(Xmax, Ymax, Zmax); vx(Xmin, Ymax, Zmax);
    vx(Xmin, Ymax, Zmax); vx(Xmin, Ymin, Zmax);

    // arêtes verticales
    vx(Xmin, Ymin, Zmin); vx(Xmin, Ymin, Zmax);
    vx(Xmax, Ymin, Zmin); vx(Xmax, Ymin, Zmax);
    vx(Xmax, Ymax, Zmin); vx(Xmax, Ymax, Zmax);
    vx(Xmin, Ymax, Zmin); vx(Xmin, Ymax, Zmax);

    glEnd();
}


void DessinerPoints()
{
    const double size = (damierCell > 0.0) ? (damierCell * 0.25) : 1.0;
    glBegin(GL_TRIANGLES);
    for (const auto& p : nuage)
    {
        glColor3f(p.r, p.g, p.b);
        // Triangle plat dans le plan du sol (x-y), centré sur le point
        double x = p.x - barycentre.x;
        double y = p.y - barycentre.y;
        double z = p.z - barycentre.z;
        glVertex3f(x, z, y - size);
        glVertex3f(x - size * 0.8660254, z, y + size * 0.5);
        glVertex3f(x + size * 0.8660254, z, y + size * 0.5);
    }
    glEnd();
}

void DessinerDamierSol(int n = 20)
{
    double dx = Xmax - Xmin;
    double dy = Ymax - Ymin;
    if (dx <= 0.0 || dy <= 0.0 || n <= 0) {
        return;
    }

    damierN = n;
    damierCell = std::max(dx, dy) / damierN;
    if (damierCell <= 0.0) damierCell = 1.0;
    damierNx = (int)std::ceil(dx / damierCell);
    damierNy = (int)std::ceil(dy / damierCell);
    const double zBase = Zmin;

    std::vector<double> sumZ(damierNx * damierNy, 0.0);
    std::vector<double> sumR(damierNx * damierNy, 0.0);
    std::vector<double> sumG(damierNx * damierNy, 0.0);
    std::vector<double> sumB(damierNx * damierNy, 0.0);
    std::vector<int> count(damierNx * damierNy, 0);

    for (const auto& p : nuage)
    {
        int i = (int)std::floor((p.x - Xmin) / damierCell);
        int k = (int)std::floor((p.y - Ymin) / damierCell);
        if (i < 0) i = 0;
        if (k < 0) k = 0;
        if (i >= damierNx) i = damierNx - 1;
        if (k >= damierNy) k = damierNy - 1;
        int idx = k * damierNx + i;
        sumZ[idx] += p.z;
        sumR[idx] += p.r;
        sumG[idx] += p.g;
        sumB[idx] += p.b;
        count[idx] += 1;
    }

    glBegin(GL_QUADS);
    for (int i = 0; i < damierNx; ++i)
    {
        for (int k = 0; k < damierNy; ++k)
        {
            int idx = k * damierNx + i;
            if (count[idx] == 0) {
                continue; // ne pas dessiner les pavés sans points
            }
            double inv = 1.0 / count[idx];
            double z = sumZ[idx] * inv;
            float r = static_cast<float>(sumR[idx] * inv);
            float g = static_cast<float>(sumG[idx] * inv);
            float b = static_cast<float>(sumB[idx] * inv);
            glColor3f(r, g, b);

            double x0 = Xmin + i * damierCell;
            double x1 = Xmin + (i + 1) * damierCell;
            double y0 = Ymin + k * damierCell;
            double y1 = Ymin + (k + 1) * damierCell;

            glVertex3f(x0 - barycentre.x, z - barycentre.z, y0 - barycentre.y);
            glVertex3f(x1 - barycentre.x, z - barycentre.z, y0 - barycentre.y);
            glVertex3f(x1 - barycentre.x, z - barycentre.z, y1 - barycentre.y);
            glVertex3f(x0 - barycentre.x, z - barycentre.z, y1 - barycentre.y);
        }
    }
    glEnd();
}

void ConstruireGrille(int n, std::vector<Cell>& grid, int& nx, int& ny, double& cell)
{
    double dx = Xmax - Xmin;
    double dy = Ymax - Ymin;
    if (dx <= 0.0 || dy <= 0.0 || n <= 1) {
        grid.clear();
        nx = ny = 0;
        cell = 1.0;
        return;
    }

    // Etape 4: pas défini à partir de la plus petite longueur
    cell = std::min(dx, dy) / n;
    if (cell <= 0.0) cell = 1.0;
    nx = (int)std::ceil(dx / cell);
    ny = (int)std::ceil(dy / cell);
    if (nx < 2) nx = 2;
    if (ny < 2) ny = 2;

    grid.assign(nx * ny, Cell{});
    for (const auto& p : nuage) {
        int i = (int)std::floor((p.x - Xmin) / cell);
        int k = (int)std::floor((p.y - Ymin) / cell);
        if (i < 0) i = 0;
        if (k < 0) k = 0;
        if (i >= nx) i = nx - 1;
        if (k >= ny) k = ny - 1;
        Cell& c = grid[k * nx + i];
        c.sumZ += p.z;
        c.sumR += p.r;
        c.sumG += p.g;
        c.sumB += p.b;
        c.count += 1;
    }

    for (auto& c : grid) {
        if (c.count > 0) {
            double inv = 1.0 / c.count;
            c.z = c.sumZ * inv;
            c.r = (float)(c.sumR * inv);
            c.g = (float)(c.sumG * inv);
            c.b = (float)(c.sumB * inv);
            c.valid = true;
        }
    }

}

void DessinerSurfaceVolcan()
{
    std::vector<Cell> grid;
    int nx = 0, ny = 0;
    double cell = 1.0;
    ConstruireGrille(terrainN, grid, nx, ny, cell);
    if (grid.empty()) return;

    const double zBase = Zmin;
    auto V = [&](double x, double y, double z) {
        glVertex3f(x - barycentre.x, z - barycentre.z, y - barycentre.y);
    };
    auto At = [&](int i, int k) -> const Cell& {
        return grid[k * nx + i];
    };

    // Triangulation "smooth": sommets partagés entre carrés voisins
    const int vnx = nx + 1;
    const int vny = ny + 1;
    std::vector<double> vz(vnx * vny, 0.0);
    std::vector<float> vr(vnx * vny, 0.0f), vg(vnx * vny, 0.0f), vb(vnx * vny, 0.0f);
    std::vector<int> vcount(vnx * vny, 0);
    auto Vidx = [&](int i, int k) { return k * vnx + i; };

    for (int k = 0; k < ny; ++k) {
        for (int i = 0; i < nx; ++i) {
            const Cell& c = At(i, k);
            if (!c.valid) continue;
            const int ids[4] = {Vidx(i, k), Vidx(i + 1, k), Vidx(i, k + 1), Vidx(i + 1, k + 1)};
            for (int t = 0; t < 4; ++t) {
                int id = ids[t];
                vz[id] += c.z;
                vr[id] += c.r;
                vg[id] += c.g;
                vb[id] += c.b;
                vcount[id] += 1;
            }
        }
    }

    std::vector<char> vvalid(vnx * vny, 0);
    for (int k = 0; k < vny; ++k) {
        for (int i = 0; i < vnx; ++i) {
            int id = Vidx(i, k);
            if (vcount[id] > 0) {
                float inv = 1.0f / (float)vcount[id];
                vz[id] *= inv;
                if (useHeatmap) {
                    HeatmapColor(vz[id], vr[id], vg[id], vb[id]);
                } else {
                    vr[id] *= inv;
                    vg[id] *= inv;
                    vb[id] *= inv;
                }
                vvalid[id] = 1;
            }
        }
    }

    // Triangulation des carrés avec altitude/couleur aux sommets
    glBegin(GL_TRIANGLES);
    for (int k = 0; k < ny; ++k) {
        for (int i = 0; i < nx; ++i) {
            int v00 = Vidx(i, k);
            int v10 = Vidx(i + 1, k);
            int v11 = Vidx(i + 1, k + 1);
            int v01 = Vidx(i, k + 1);
            if (!(vvalid[v00] && vvalid[v10] && vvalid[v11] && vvalid[v01])) continue;

            double x0 = Xmin + i * cell;
            double x1 = Xmin + (i + 1) * cell;
            double y0 = Ymin + k * cell;
            double y1 = Ymin + (k + 1) * cell;

            glColor3f(vr[v00], vg[v00], vb[v00]); V(x0, y0, vz[v00]);
            glColor3f(vr[v10], vg[v10], vb[v10]); V(x1, y0, vz[v10]);
            glColor3f(vr[v11], vg[v11], vb[v11]); V(x1, y1, vz[v11]);

            glColor3f(vr[v00], vg[v00], vb[v00]); V(x0, y0, vz[v00]);
            glColor3f(vr[v11], vg[v11], vb[v11]); V(x1, y1, vz[v11]);
            glColor3f(vr[v01], vg[v01], vb[v01]); V(x0, y1, vz[v01]);
        }
    }
    glEnd();

    // Parois verticales sur le contour
    glColor3f(0.88f, 0.88f, 0.88f);
    glBegin(GL_QUADS);
    for (int i = 0; i < nx; ++i) {
        int v0 = Vidx(i, 0), v1 = Vidx(i + 1, 0);
        if (vvalid[v0] && vvalid[v1]) {
            double x0 = Xmin + i * cell;
            double x1 = Xmin + (i + 1) * cell;
            double y = Ymin;
            V(x0, y, vz[v0]); V(x1, y, vz[v1]); V(x1, y, zBase); V(x0, y, zBase);
        }
    }
    for (int i = 0; i < nx; ++i) {
        int v0 = Vidx(i, ny), v1 = Vidx(i + 1, ny);
        if (vvalid[v0] && vvalid[v1]) {
            double x0 = Xmin + i * cell;
            double x1 = Xmin + (i + 1) * cell;
            double y = Ymin + ny * cell;
            V(x0, y, vz[v0]); V(x1, y, vz[v1]); V(x1, y, zBase); V(x0, y, zBase);
        }
    }
    for (int k = 0; k < ny; ++k) {
        int v0 = Vidx(0, k), v1 = Vidx(0, k + 1);
        if (vvalid[v0] && vvalid[v1]) {
            double x = Xmin;
            double y0 = Ymin + k * cell;
            double y1 = Ymin + (k + 1) * cell;
            V(x, y0, vz[v0]); V(x, y1, vz[v1]); V(x, y1, zBase); V(x, y0, zBase);
        }
    }
    for (int k = 0; k < ny; ++k) {
        int v0 = Vidx(nx, k), v1 = Vidx(nx, k + 1);
        if (vvalid[v0] && vvalid[v1]) {
            double x = Xmin + nx * cell;
            double y0 = Ymin + k * cell;
            double y1 = Ymin + (k + 1) * cell;
            V(x, y0, vz[v0]); V(x, y1, vz[v1]); V(x, y1, zBase); V(x, y0, zBase);
        }
    }
    glEnd();
}

void DessinerCarresVolcan()
{
    std::vector<Cell> grid;
    int nx = 0, ny = 0;
    double cell = 1.0;
    ConstruireGrille(terrainN, grid, nx, ny, cell);
    if (grid.empty()) return;

    auto V = [&](double x, double y, double z) {
        glVertex3f(x - barycentre.x, z - barycentre.z, y - barycentre.y);
    };
    auto At = [&](int i, int k) -> const Cell& {
        return grid[k * nx + i];
    };

    glBegin(GL_QUADS);
    for (int k = 0; k < ny; ++k) {
        for (int i = 0; i < nx; ++i) {
            const Cell& c = At(i, k);
            if (!c.valid) continue;
            float r = c.r, g = c.g, b = c.b;
            if (useHeatmap) HeatmapColor(c.z, r, g, b);
            glColor3f(r, g, b);

            double x0 = Xmin + i * cell;
            double x1 = Xmin + (i + 1) * cell;
            double y0 = Ymin + k * cell;
            double y1 = Ymin + (k + 1) * cell;
            V(x0, y0, c.z);
            V(x1, y0, c.z);
            V(x1, y1, c.z);
            V(x0, y1, c.z);
        }
    }
    glEnd();
}

void DessinerCourbesNiveau()
{
    std::vector<Cell> grid;
    int nx = 0, ny = 0;
    double cell = 1.0;
    ConstruireGrille(terrainN, grid, nx, ny, cell);
    if (grid.empty()) return;

    auto At = [&](int i, int k) -> const Cell& {
        return grid[k * nx + i];
    };
    auto V = [&](double x, double y, double z) {
        glVertex3f(x - barycentre.x, z - barycentre.z, y - barycentre.y);
    };

    const int vnx = nx + 1;
    const int vny = ny + 1;
    std::vector<double> vz(vnx * vny, 0.0);
    std::vector<int> vcount(vnx * vny, 0);
    auto Vidx = [&](int i, int k) { return k * vnx + i; };

    for (int k = 0; k < ny; ++k) {
        for (int i = 0; i < nx; ++i) {
            const Cell& c = At(i, k);
            if (!c.valid) continue;
            const int ids[4] = {Vidx(i, k), Vidx(i + 1, k), Vidx(i, k + 1), Vidx(i + 1, k + 1)};
            for (int t = 0; t < 4; ++t) {
                vz[ids[t]] += c.z;
                vcount[ids[t]] += 1;
            }
        }
    }

    std::vector<char> vvalid(vnx * vny, 0);
    for (int k = 0; k < vny; ++k) {
        for (int i = 0; i < vnx; ++i) {
            int id = Vidx(i, k);
            if (vcount[id] > 0) {
                vz[id] /= (double)vcount[id];
                vvalid[id] = 1;
            }
        }
    }

    const int nLevels = 16;
    const double dz = (Zmax > Zmin) ? (Zmax - Zmin) / (nLevels + 1) : 1.0;
    if (dz <= 0.0) return;

    glLineWidth(3.0f);
    for (int l = 1; l <= nLevels; ++l) {
        double iso = Zmin + l * dz;
        float lr, lg, lb;
        HeatmapColor(iso, lr, lg, lb);
        glColor3f(lr, lg, lb);
        glBegin(GL_LINES);
        for (int k = 0; k < ny; ++k) {
            for (int i = 0; i < nx; ++i) {
                int v00 = Vidx(i, k);
                int v10 = Vidx(i + 1, k);
                int v11 = Vidx(i + 1, k + 1);
                int v01 = Vidx(i, k + 1);
                if (!(vvalid[v00] && vvalid[v10] && vvalid[v11] && vvalid[v01])) continue;

                double x0 = Xmin + i * cell;
                double x1 = Xmin + (i + 1) * cell;
                double y0 = Ymin + k * cell;
                double y1 = Ymin + (k + 1) * cell;

                struct IPoint { double x, y, z; };
                IPoint p[4] = {
                    {x0, y0, vz[v00]},
                    {x1, y0, vz[v10]},
                    {x1, y1, vz[v11]},
                    {x0, y1, vz[v01]}
                };
                int e0[4] = {0, 1, 2, 3};
                int e1[4] = {1, 2, 3, 0};
                IPoint inter[4];
                int nInter = 0;
                for (int e = 0; e < 4; ++e) {
                    const IPoint& a = p[e0[e]];
                    const IPoint& b = p[e1[e]];
                    double da = a.z - iso;
                    double db = b.z - iso;
                    if ((da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0)) {
                        double t = (iso - a.z) / (b.z - a.z);
                        inter[nInter++] = {a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), iso};
                    }
                }
                if (nInter == 2) {
                    V(inter[0].x, inter[0].y, inter[0].z);
                    V(inter[1].x, inter[1].y, inter[1].z);
                } else if (nInter == 4) {
                    V(inter[0].x, inter[0].y, inter[0].z);
                    V(inter[1].x, inter[1].y, inter[1].z);
                    V(inter[2].x, inter[2].y, inter[2].z);
                    V(inter[3].x, inter[3].y, inter[3].z);
                }
            }
        }
        glEnd();
    }
    glLineWidth(1.0f);
}

void Display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(4,3,3, 0,0,0, 0,1,0);

    glRotatef(rotX, 1, 0, 0);
    glRotatef(rotY, 0, 1, 0);
    glScaled(zoom/scaleAuto, zoom/scaleAuto, zoom/scaleAuto);

    DessinerDamierSol(35);
    DessinerBoite();
    if (renderMode == MODE_POINTS) {
        DessinerPoints();
    } else if (renderMode == MODE_CARREE) {
        DessinerCarresVolcan();
    } else if (renderMode == MODE_CONTOUR) {
        DessinerCarresVolcan();
        DessinerCourbesNiveau();
    } else {
        DessinerSurfaceVolcan();
    }
    DessinerSidebar();
    glutSwapBuffers();
}

void MouseButton(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && x < sidebarW) {
        const int y0 = 20;
        const int h = 34;
        const int gap = 10;
        const int firstY = y0 + 20;
        int idx = (y - firstY) / (h + gap);
        int yIn = (y - firstY) % (h + gap);
        if (idx >= 0 && idx <= 5 && yIn >= 0 && yIn < h) {
            if (idx == 0) renderMode = MODE_POINTS;
            if (idx == 1) renderMode = MODE_CARREE;
            if (idx == 2) renderMode = MODE_TRIANGULATION;
            if (idx == 3) renderMode = MODE_CONTOUR;
            if (idx == 4) useHeatmap = !useHeatmap;
            if (idx == 5) { zoom = 1.0f; rotX = 20.0f; rotY = -30.0f; }
            glutPostRedisplay();
            return;
        }
    }

    if (state == GLUT_DOWN) {
        if (button == 3) { // molette haut
            zoom *= 1.1f;
            if (zoom > 20.0f) zoom = 20.0f;
            glutPostRedisplay();
            return;
        }
        if (button == 4) { // molette bas
            zoom *= 0.9f;
            if (zoom < 0.05f) zoom = 0.05f;
            glutPostRedisplay();
            return;
        }
    }

    if (button == GLUT_LEFT_BUTTON)
    {
        if (state == GLUT_DOWN) {
            dragging = true;
            lastX = x;
            lastY = y;
        } else {
            dragging = false;
        }
    }
}

void Keyboard(unsigned char key, int, int)
{
    if (key == '+' || key == '=') {
        zoom *= 1.1f;
        if (zoom > 20.0f) zoom = 20.0f;
        glutPostRedisplay();
    } else if (key == '-' || key == '_') {
        zoom *= 0.9f;
        if (zoom < 0.05f) zoom = 0.05f;
        glutPostRedisplay();
    } else if (key == 'h' || key == 'H') {
        useHeatmap = !useHeatmap;
        glutPostRedisplay();
    } else if (key == '1') {
        renderMode = MODE_POINTS;
        glutPostRedisplay();
    } else if (key == '2') {
        renderMode = MODE_CARREE;
        glutPostRedisplay();
    } else if (key == '3') {
        renderMode = MODE_TRIANGULATION;
        glutPostRedisplay();
    } else if (key == '4') {
        renderMode = MODE_CONTOUR;
        glutPostRedisplay();
    }
}

void MenuCallback(int value)
{
    switch (value) {
        case 1: renderMode = MODE_POINTS; break;
        case 2: renderMode = MODE_CARREE; break;
        case 3: renderMode = MODE_TRIANGULATION; break;
        case 4: useHeatmap = true; break;
        case 5: useHeatmap = false; break;
        case 6: zoom = 1.0f; rotX = 20.0f; rotY = -30.0f; break;
        default: break;
    }
    glutPostRedisplay();
}

void MouseMotion(int x, int y)
{
    if (!dragging) return;
    int dx = x - lastX;
    int dy = y - lastY;
    rotY += dx * 0.5f;
    rotX += dy * 0.5f;
    lastX = x;
    lastY = y;
    glutPostRedisplay();
}


int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(640 , 480);
    glutCreateWindow("Damier - Point Cloud");
    glutReshapeFunc(Reshape);
    glutMouseFunc(MouseButton);
    glutMotionFunc(MouseMotion);
    glutKeyboardFunc(Keyboard);

    glEnable(GL_DEPTH_TEST);

    if (!ChargerNuage("volcan.xyz")) {
        return 1;
    }
    CalculerMinMax();
    std::cout << "Xmin=" << Xmin << " Xmax=" << Xmax << std::endl;
    std::cout << "Ymin=" << Ymin << " Ymax=" << Ymax << std::endl;
    std::cout << "Zmin=" << Zmin << " Zmax=" << Zmax << std::endl;

    glutDisplayFunc(Display);
    glutMainLoop();
    return 0;
}
