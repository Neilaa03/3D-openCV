#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>
#include <cstdint>
#include <iostream>

#include <GL/glut.h>

struct Point{
    double x, y, z;
};

std::vector<Point> nuage;
double Xmin, Ymin, Zmin, Xmax, Ymax, Zmax;
Point barycentre;
double scaleAuto = 1.0;
int damierN = 20;
double damierCell = 1.0;
int damierNx = 1;
int damierNy = 1;
float rotX = 20.0f;
float rotY = -30.0f;
int lastX = 0, lastY = 0;
bool dragging = false;

bool ChargerNuage(const char* filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    double a,b,c,d,e,f,g,h;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        std::istringstream iss(line);
        if (!(iss >> a >> b >> c >> d >> e >> f >> g >> h)) continue;

        Point p;
        p.x = b; // comme dans ton main.cpp
        p.y = c;
        p.z = d;
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
            double target = 3; // taille voulue dans la vue
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
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)w / (double)h, 0.1, 100.0);
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
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (const auto& p : nuage)
    {
        if (damierCell <= 0.0) damierCell = 1.0;
        int i = (int)std::floor((p.x - Xmin) / damierCell);
        int k = (int)std::floor((p.y - Ymin) / damierCell);
        if (i < 0) i = 0;
        if (k < 0) k = 0;
        if (i >= damierNx) i = damierNx - 1;
        if (k >= damierNy) k = damierNy - 1;
        auto hash01 = [](std::uint32_t v) -> float {
            v ^= v >> 16;
            v *= 0x7feb352d;
            v ^= v >> 15;
            v *= 0x846ca68b;
            v ^= v >> 16;
            return (v & 0x00FFFFFF) / 16777215.0f;
        };
        std::uint32_t seed = (std::uint32_t)(i * 73856093u) ^ (std::uint32_t)(k * 19349663u);
        float r = hash01(seed);
        float g = hash01(seed ^ 0x9e3779b9u);
        float b = hash01(seed ^ 0x7f4a7c15u);
        //glColor3f(r, g, b);
        glColor3f(0, 0, 255);
        glVertex3f(p.x - barycentre.x,
                   p.z - barycentre.z,
                   p.y - barycentre.y);
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
    double z = Zmin;

    glBegin(GL_QUADS);
    for (int i = 0; i < damierNx; ++i)
    {
        for (int k = 0; k < damierNy; ++k)
        {
            auto hash01 = [](std::uint32_t v) -> float {
                v ^= v >> 16;
                v *= 0x7feb352d;
                v ^= v >> 15;
                v *= 0x846ca68b;
                v ^= v >> 16;
                return (v & 0x00FFFFFF) / 16777215.0f;
            };
            std::uint32_t seed = (std::uint32_t)(i * 73856093u) ^ (std::uint32_t)(k * 19349663u);
            float r = hash01(seed);
            float g = hash01(seed ^ 0x9e3779b9u);
            float b = hash01(seed ^ 0x7f4a7c15u);
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

void Display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(4,3,3, 0,0,0, 0,1,0);

    glRotatef(rotX, 1, 0, 0);
    glRotatef(rotY, 0, 1, 0);
    glScaled(1.0/scaleAuto, 1.0/scaleAuto, 1.0/scaleAuto);

    DessinerDamierSol(20);
    DessinerBoite();
    DessinerPoints();
    glutSwapBuffers();
}

void MouseButton(int button, int state, int x, int y)
{
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

    glEnable(GL_DEPTH_TEST);

    if (!ChargerNuage("Facade.xyz")) {
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
