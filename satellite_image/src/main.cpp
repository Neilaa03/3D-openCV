#include <opencv2/opencv.hpp>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {
struct Texture {
    GLuint id = 0;
    int w = 0;
    int h = 0;
};

Texture g_satellite;
Texture g_drone;
Texture g_drone_rectified;
std::vector<std::string> g_drone_paths;
std::vector<cv::Mat> g_drone_images;
std::vector<std::vector<cv::KeyPoint>> g_drone_keypoints;
std::vector<cv::Mat> g_drone_descriptors;
std::vector<cv::KeyPoint> g_sat_keypoints;
cv::Mat g_sat_descriptors;
std::vector<std::vector<cv::DMatch>> g_drone_matches;
std::vector<cv::Mat> g_drone_homographies;
std::vector<int> g_drone_inliers;
std::vector<cv::Mat> g_drone_rectified_rgba;
std::vector<cv::Point2f> g_drone_projected_centers;
int g_current_drone = -1;

int g_window_w = 800;
int g_window_h = 600;
float g_drone_x = 0.0f;
float g_drone_y = 0.0f;
bool g_dragging = false;
float g_drag_offset_x = 0.0f;
float g_drag_offset_y = 0.0f;
float g_drone_scale = 0.06f;
float g_satellite_scale = 1.6f;
int g_panel_w = 260;
int g_panel_padding = 12;
int g_item_h = 22;
int g_panel_header_h = 24;
int g_dropdown_h = 26;
int g_list_scroll = 0;
bool g_dropdown_open = false;
bool g_use_initial_window = true;
bool g_show_rectified = true;

struct GpsPoint {
    double lon = 0.0;
    double lat = 0.0;
};

std::vector<std::optional<GpsPoint>> g_drone_gps;
cv::Mat g_gps_to_pixel_affine;

Texture CreateTexture(const cv::Mat& image) {
    Texture tex;
    if (image.empty()) {
        std::cerr << "Image vide, impossible de creer la texture." << std::endl;
        std::exit(1);
    }

    cv::Mat rgba;
    if (image.channels() == 4) {
        cv::cvtColor(image, rgba, cv::COLOR_BGRA2RGBA);
    } else {
        cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);
    }
    cv::flip(rgba, rgba, 0);

    glGenTextures(1, &tex.id);
    glBindTexture(GL_TEXTURE_2D, tex.id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        rgba.cols,
        rgba.rows,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        rgba.data
    );
    tex.w = rgba.cols;
    tex.h = rgba.rows;
    return tex;
}

void DestroyTexture(Texture& tex) {
    if (tex.id != 0) {
        glDeleteTextures(1, &tex.id);
        tex.id = 0;
        tex.w = 0;
        tex.h = 0;
    }
}

void ComputeSiftForSatellite(const cv::Mat& sat_image) {
    auto sift = cv::SIFT::create();
    sift->detectAndCompute(sat_image, cv::noArray(), g_sat_keypoints, g_sat_descriptors);
    std::cout << "SIFT satellite: " << g_sat_keypoints.size() << " points" << std::endl;
}

void ComputeSiftForDrone(const cv::Mat& drone_img, std::vector<cv::KeyPoint>& kps, cv::Mat& desc) {
    auto sift = cv::SIFT::create();
    sift->detectAndCompute(drone_img, cv::noArray(), kps, desc);
}

std::optional<GpsPoint> ParseGpsFromPath(const std::string& path) {
    const std::filesystem::path p(path);
    const std::string name = p.filename().string();
    std::vector<std::string> fields;
    std::stringstream ss(name);
    std::string item;
    while (std::getline(ss, item, '@')) {
        if (!item.empty()) {
            fields.push_back(item);
        }
    }
    if (fields.size() < 3) {
        return std::nullopt;
    }
    try {
        GpsPoint gps;
        gps.lon = std::stod(fields[1]);
        gps.lat = std::stod(fields[2]);
        return gps;
    } catch (...) {
        return std::nullopt;
    }
}

void ComputeGpsCalibration() {
    std::vector<cv::Point2f> gps_points;
    std::vector<cv::Point2f> pixel_points;
    for (int i = 0; i < static_cast<int>(g_drone_gps.size()); ++i) {
        if (!g_drone_gps[i].has_value()) {
            continue;
        }
        if (i >= static_cast<int>(g_drone_homographies.size()) || g_drone_homographies[i].empty()) {
            continue;
        }
        if (i >= static_cast<int>(g_drone_projected_centers.size())) {
            continue;
        }
        const auto& gps = g_drone_gps[i].value();
        gps_points.emplace_back(static_cast<float>(gps.lon), static_cast<float>(gps.lat));
        pixel_points.push_back(g_drone_projected_centers[i]);
    }
    if (gps_points.size() < 3) {
        g_gps_to_pixel_affine = cv::Mat();
        return;
    }
    cv::Mat inliers;
    g_gps_to_pixel_affine = cv::estimateAffine2D(gps_points, pixel_points, inliers, cv::RANSAC);
    if (!g_gps_to_pixel_affine.empty()) {
        std::cout << "Calibration GPS->pixel calculee avec " << gps_points.size() << " points." << std::endl;
    }
}

void ComputeHomographyAndRectification(int index) {
    if (index < 0 || index >= static_cast<int>(g_drone_images.size())) {
        return;
    }
    if (index >= static_cast<int>(g_drone_matches.size()) || g_drone_matches[index].size() < 4) {
        return;
    }
    const auto& matches = g_drone_matches[index];
    std::vector<cv::Point2f> pts_drone;
    std::vector<cv::Point2f> pts_sat;
    pts_drone.reserve(matches.size());
    pts_sat.reserve(matches.size());

    for (const auto& m : matches) {
        pts_drone.push_back(g_drone_keypoints[index][m.queryIdx].pt);
        pts_sat.push_back(g_sat_keypoints[m.trainIdx].pt);
    }

    cv::Mat inlier_mask;
    cv::Mat H = cv::findHomography(pts_drone, pts_sat, cv::RANSAC, 4.0, inlier_mask);
    if (index >= static_cast<int>(g_drone_homographies.size())) {
        g_drone_homographies.resize(g_drone_images.size());
        g_drone_inliers.resize(g_drone_images.size(), 0);
        g_drone_rectified_rgba.resize(g_drone_images.size());
        g_drone_projected_centers.resize(g_drone_images.size(), cv::Point2f(-1.0f, -1.0f));
    }
    g_drone_homographies[index] = H;
    if (H.empty()) {
        g_drone_inliers[index] = 0;
        g_drone_rectified_rgba[index] = cv::Mat();
        std::cout << "Homographie drone " << (index + 1) << ": echec" << std::endl;
        return;
    }

    int inliers = 0;
    for (int i = 0; i < inlier_mask.rows; ++i) {
        if (inlier_mask.at<uchar>(i, 0) != 0) {
            ++inliers;
        }
    }
    g_drone_inliers[index] = inliers;
    const float inlier_ratio = matches.empty() ? 0.0f : static_cast<float>(inliers) / static_cast<float>(matches.size());
    std::cout << "Homographie drone " << (index + 1) << ": " << inliers
              << "/" << matches.size() << " inliers (" << inlier_ratio << ")" << std::endl;

    cv::Mat warped_bgr;
    cv::warpPerspective(g_drone_images[index], warped_bgr, H, cv::Size(g_satellite.w, g_satellite.h));
    cv::Mat mask(g_drone_images[index].size(), CV_8UC1, cv::Scalar(255));
    cv::Mat warped_mask;
    cv::warpPerspective(mask, warped_mask, H, cv::Size(g_satellite.w, g_satellite.h));

    cv::Mat warped_rgba;
    cv::cvtColor(warped_bgr, warped_rgba, cv::COLOR_BGR2RGBA);
    for (int y = 0; y < warped_rgba.rows; ++y) {
        cv::Vec4b* p = warped_rgba.ptr<cv::Vec4b>(y);
        const uchar* m = warped_mask.ptr<uchar>(y);
        for (int x = 0; x < warped_rgba.cols; ++x) {
            if (m[x] == 0) {
                p[x][3] = 0;
            } else {
                p[x][3] = 180;
            }
        }
    }
    g_drone_rectified_rgba[index] = warped_rgba;

    std::vector<cv::Point2f> center(1, cv::Point2f(
        static_cast<float>(g_drone_images[index].cols) * 0.5f,
        static_cast<float>(g_drone_images[index].rows) * 0.5f));
    std::vector<cv::Point2f> center_proj;
    cv::perspectiveTransform(center, center_proj, H);
    if (!center_proj.empty()) {
        g_drone_projected_centers[index] = center_proj[0];
    }
}

void ComputeMatches(int index) {
    if (index < 0 || index >= static_cast<int>(g_drone_descriptors.size())) {
        return;
    }
    if (g_sat_descriptors.empty() || g_drone_descriptors[index].empty()) {
        return;
    }
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(g_drone_descriptors[index], g_sat_descriptors, knn, 2);

    std::vector<cv::DMatch> good;
    for (const auto& m : knn) {
        if (m.size() < 2) {
            continue;
        }
        if (m[0].distance < 0.75f * m[1].distance) {
            good.push_back(m[0]);
        }
    }

    if (index >= static_cast<int>(g_drone_matches.size())) {
        g_drone_matches.resize(g_drone_descriptors.size());
    }
    g_drone_matches[index] = std::move(good);

    std::cout << "Matches drone " << (index + 1) << ": "
              << g_drone_matches[index].size() << " (ratio test 0.75)" << std::endl;
    ComputeHomographyAndRectification(index);
    ComputeGpsCalibration();
}

void LoadDroneTexture(int index) {
    if (index < 0 || index >= static_cast<int>(g_drone_images.size())) {
        return;
    }
    DestroyTexture(g_drone);
    DestroyTexture(g_drone_rectified);
    g_drone = CreateTexture(g_drone_images[index]);
    g_current_drone = index;
    g_drone_x = 0.0f;
    g_drone_y = 0.0f;

    if (index >= 0 && index < static_cast<int>(g_drone_keypoints.size())) {
        if (g_drone_descriptors[index].empty()) {
            ComputeSiftForDrone(g_drone_images[index], g_drone_keypoints[index], g_drone_descriptors[index]);
        }
        std::cout << "SIFT drone " << (index + 1) << ": "
                  << g_drone_keypoints[index].size() << " points" << std::endl;
    }

    ComputeMatches(index);
    if (g_show_rectified && index < static_cast<int>(g_drone_rectified_rgba.size()) &&
        !g_drone_rectified_rgba[index].empty()) {
        g_drone_rectified = CreateTexture(g_drone_rectified_rgba[index]);
    }
}

void ClearDroneTexture() {
    DestroyTexture(g_drone);
    DestroyTexture(g_drone_rectified);
    g_current_drone = -1;
}

bool IsInsideDrone(float x, float y) {
    if (g_drone.id == 0) {
        return false;
    }
    return x >= g_drone_x && x <= g_drone_x + g_drone.w * g_drone_scale &&
           y >= g_drone_y && y <= g_drone_y + g_drone.h * g_drone_scale;
}

std::string DroneLabel(int index) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "Drone %02d", index + 1);
    return std::string(buffer);
}

int PanelContentWidth() {
    int w = g_window_w - g_panel_w;
    return (w > 1) ? w : 1;
}

void Display() {
    glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    int content_w = PanelContentWidth();

    // Draw satellite scaled to fit the content area while preserving aspect ratio.
    float sat_w = static_cast<float>(g_satellite.w);
    float sat_h = static_cast<float>(g_satellite.h);
    float scale = 1.0f;
    if (sat_w > 0.0f && sat_h > 0.0f) {
        float sx = static_cast<float>(content_w) / sat_w;
        float sy = static_cast<float>(g_window_h) / sat_h;
        scale = (sx < sy) ? sx : sy;
    }
    scale *= g_satellite_scale;
    float draw_w = sat_w * scale;
    float draw_h = sat_h * scale;
    float offset_x = (static_cast<float>(content_w) - draw_w) * 0.5f;
    float offset_y = (static_cast<float>(g_window_h) - draw_h) * 0.5f;

    glEnable(GL_TEXTURE_2D);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBindTexture(GL_TEXTURE_2D, g_satellite.id);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(offset_x, offset_y);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(offset_x + draw_w, offset_y);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(offset_x + draw_w, offset_y + draw_h);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(offset_x, offset_y + draw_h);
    glEnd();

    if (g_drone_rectified.id != 0) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindTexture(GL_TEXTURE_2D, g_drone_rectified.id);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(offset_x, offset_y);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(offset_x + draw_w, offset_y);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(offset_x + draw_w, offset_y + draw_h);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(offset_x, offset_y + draw_h);
        glEnd();
        glDisable(GL_BLEND);
    } else if (g_drone.id != 0) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glBindTexture(GL_TEXTURE_2D, g_drone.id);
        float x0 = g_drone_x;
        float y0 = g_drone_y;
        float x1 = x0 + static_cast<float>(g_drone.w) * g_drone_scale;
        float y1 = y0 + static_cast<float>(g_drone.h) * g_drone_scale;

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(x0, y0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(x1, y0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(x1, y1);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(x0, y1);
        glEnd();

        glDisable(GL_BLEND);
    }

    glDisable(GL_TEXTURE_2D);

    // Right panel
    glColor3f(0.12f, 0.12f, 0.12f);
    glBegin(GL_QUADS);
    glVertex2f(static_cast<float>(content_w), 0.0f);
    glVertex2f(static_cast<float>(g_window_w), 0.0f);
    glVertex2f(static_cast<float>(g_window_w), static_cast<float>(g_window_h));
    glVertex2f(static_cast<float>(content_w), static_cast<float>(g_window_h));
    glEnd();

    glColor3f(0.9f, 0.9f, 0.9f);
    glRasterPos2f(static_cast<float>(content_w + g_panel_padding),
                  static_cast<float>(g_panel_padding + g_panel_header_h));
    const char* title = "Images drone";
    for (const char* c = title; *c; ++c) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
    }

    int dropdown_x0 = content_w + g_panel_padding;
    int dropdown_x1 = g_window_w - g_panel_padding;
    int dropdown_y0 = g_panel_padding + g_panel_header_h + 6;
    int dropdown_y1 = dropdown_y0 + g_dropdown_h;

    glColor3f(0.18f, 0.18f, 0.18f);
    glBegin(GL_QUADS);
    glVertex2f(static_cast<float>(dropdown_x0), static_cast<float>(dropdown_y0));
    glVertex2f(static_cast<float>(dropdown_x1), static_cast<float>(dropdown_y0));
    glVertex2f(static_cast<float>(dropdown_x1), static_cast<float>(dropdown_y1));
    glVertex2f(static_cast<float>(dropdown_x0), static_cast<float>(dropdown_y1));
    glEnd();

    glColor3f(0.95f, 0.95f, 0.95f);
    std::string selected = (g_current_drone >= 0) ? DroneLabel(g_current_drone) : "Selectionner une image";
    glRasterPos2f(static_cast<float>(dropdown_x0 + 8), static_cast<float>(dropdown_y1 - 8));
    for (char c : selected) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
    }

    // Dropdown arrow
    glBegin(GL_TRIANGLES);
    glVertex2f(static_cast<float>(dropdown_x1 - 18), static_cast<float>(dropdown_y0 + 10));
    glVertex2f(static_cast<float>(dropdown_x1 - 6), static_cast<float>(dropdown_y0 + 10));
    glVertex2f(static_cast<float>(dropdown_x1 - 12), static_cast<float>(dropdown_y0 + 18));
    glEnd();

    int list_top = dropdown_y1 + 8;
    int list_bottom = g_window_h - g_panel_padding;
    int list_h = list_bottom - list_top;
    int max_visible = (list_h > 0) ? (list_h / g_item_h) : 0;

    if (g_dropdown_open && max_visible > 0) {
        glColor3f(0.16f, 0.16f, 0.16f);
        glBegin(GL_QUADS);
        glVertex2f(static_cast<float>(dropdown_x0), static_cast<float>(list_top));
        glVertex2f(static_cast<float>(dropdown_x1), static_cast<float>(list_top));
        glVertex2f(static_cast<float>(dropdown_x1), static_cast<float>(list_bottom));
        glVertex2f(static_cast<float>(dropdown_x0), static_cast<float>(list_bottom));
        glEnd();

        int total = static_cast<int>(g_drone_paths.size());
        int max_scroll = (total > max_visible) ? (total - max_visible) : 0;
        if (g_list_scroll < 0) g_list_scroll = 0;
        if (g_list_scroll > max_scroll) g_list_scroll = max_scroll;

        int y = list_top + g_item_h - 6;
        for (int i = 0; i < max_visible; ++i) {
            int idx = i + g_list_scroll;
            if (idx >= total) {
                break;
            }
            bool active = (idx == g_current_drone);
            if (active) {
                glColor3f(0.25f, 0.35f, 0.55f);
                glBegin(GL_QUADS);
                glVertex2f(static_cast<float>(dropdown_x0 + 4), static_cast<float>(y - 12));
                glVertex2f(static_cast<float>(dropdown_x1 - 4), static_cast<float>(y - 12));
                glVertex2f(static_cast<float>(dropdown_x1 - 4), static_cast<float>(y + 6));
                glVertex2f(static_cast<float>(dropdown_x0 + 4), static_cast<float>(y + 6));
                glEnd();
                glColor3f(1.0f, 1.0f, 1.0f);
            } else {
                glColor3f(0.85f, 0.85f, 0.85f);
            }

            std::string label = DroneLabel(idx);
            glRasterPos2f(static_cast<float>(dropdown_x0 + 8), static_cast<float>(y));
            for (char c : label) {
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
            }
            y += g_item_h;
        }
    }

    glutSwapBuffers();
}

void Reshape(int w, int h) {
    g_window_w = (w > 1) ? w : 1;
    g_window_h = (h > 1) ? h : 1;
    glViewport(0, 0, g_window_w, g_window_h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, static_cast<double>(g_window_w),
            static_cast<double>(g_window_h), 0.0,
            -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void Keyboard(unsigned char key, int /*x*/, int /*y*/) {
    if (key == 27 || key == 'q' || key == 'Q') {
        std::exit(0);
    }
    if (key == 'r' || key == 'R') {
        g_show_rectified = !g_show_rectified;
        if (g_current_drone >= 0) {
            DestroyTexture(g_drone_rectified);
            if (g_show_rectified &&
                g_current_drone < static_cast<int>(g_drone_rectified_rgba.size()) &&
                !g_drone_rectified_rgba[g_current_drone].empty()) {
                g_drone_rectified = CreateTexture(g_drone_rectified_rgba[g_current_drone]);
            }
        }
        std::cout << "Mode rectifie: " << (g_show_rectified ? "ON" : "OFF") << std::endl;
        glutPostRedisplay();
    }
}

void Mouse(int button, int state, int x, int y) {
    if (button == 3 && state == GLUT_DOWN) { // wheel up
        if (g_dropdown_open) {
            g_list_scroll -= 1;
            glutPostRedisplay();
        }
        return;
    }
    if (button == 4 && state == GLUT_DOWN) { // wheel down
        if (g_dropdown_open) {
            g_list_scroll += 1;
            glutPostRedisplay();
        }
        return;
    }
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            float fx = static_cast<float>(x);
            float fy = static_cast<float>(y);
            int content_w = PanelContentWidth();
            int dropdown_x0 = content_w + g_panel_padding;
            int dropdown_x1 = g_window_w - g_panel_padding;
            int dropdown_y0 = g_panel_padding + g_panel_header_h + 6;
            int dropdown_y1 = dropdown_y0 + g_dropdown_h;
            int list_top = dropdown_y1 + 8;
            int list_bottom = g_window_h - g_panel_padding;

            if (fx >= content_w) {
                bool in_dropdown = (fx >= dropdown_x0 && fx <= dropdown_x1 && fy >= dropdown_y0 && fy <= dropdown_y1);
                if (in_dropdown) {
                    g_dropdown_open = !g_dropdown_open;
                    glutPostRedisplay();
                    return;
                }

                if (g_dropdown_open && fy >= list_top && fy <= list_bottom) {
                    int list_h = list_bottom - list_top;
                    int max_visible = (list_h > 0) ? (list_h / g_item_h) : 0;
                    int index_in_view = (static_cast<int>(fy) - list_top) / g_item_h;
                    int idx = g_list_scroll + index_in_view;
                    if (index_in_view >= 0 && index_in_view < max_visible &&
                        idx >= 0 && idx < static_cast<int>(g_drone_paths.size())) {
                        LoadDroneTexture(idx);
                        glutPostRedisplay();
                    }
                    return;
                }

                if (g_dropdown_open) {
                    g_dropdown_open = false;
                    glutPostRedisplay();
                }
                return;
            }
            if (IsInsideDrone(fx, fy)) {
                g_dragging = true;
                g_drag_offset_x = fx - g_drone_x;
                g_drag_offset_y = fy - g_drone_y;
            }
        } else if (state == GLUT_UP) {
            g_dragging = false;
        }
    }
}

void Motion(int x, int y) {
    if (!g_dragging) {
        return;
    }
    g_drone_x = static_cast<float>(x) - g_drag_offset_x;
    g_drone_y = static_cast<float>(y) - g_drag_offset_y;
    glutPostRedisplay();
}

}  // namespace

int main(int argc, char** argv) {
    std::string image_path = "dataset/map.jpg";
    if (argc > 1) {
        image_path = argv[1];
    }

    cv::Mat sat_image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (sat_image.empty()) {
        std::cerr << "Impossible de charger l'image: " << image_path << std::endl;
        return 1;
    }

    ComputeSiftForSatellite(sat_image);

    if (g_use_initial_window) {
        int max_w = 1280;
        int max_h = 800;
        float sx = static_cast<float>(max_w) / static_cast<float>(sat_image.cols);
        float sy = static_cast<float>(max_h) / static_cast<float>(sat_image.rows);
        float s = (sx < sy) ? sx : sy;
        if (s > 1.0f) {
            s = 1.0f;
        }
        g_window_w = static_cast<int>(sat_image.cols * s);
        g_window_h = static_cast<int>(sat_image.rows * s);
    } else {
        g_window_w = sat_image.cols;
        g_window_h = sat_image.rows;
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(g_window_w, g_window_h);
    glutCreateWindow("Satellite Viewer");

    g_satellite = CreateTexture(sat_image);

    const std::string frames_dir = "dataset/frames";
    if (std::filesystem::exists(frames_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(frames_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            g_drone_paths.push_back(entry.path().string());
        }
        std::sort(g_drone_paths.begin(), g_drone_paths.end());
    }
    if (!g_drone_paths.empty()) {
        g_drone_images.reserve(g_drone_paths.size());
        g_drone_gps.reserve(g_drone_paths.size());
        std::cout << "Chargement des images drone..." << std::endl;
        for (const auto& path : g_drone_paths) {
            cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Impossible de charger l'image drone: " << path << std::endl;
                continue;
            }
            g_drone_images.push_back(img);
            g_drone_gps.push_back(ParseGpsFromPath(path));
        }
        std::cout << "Images drone chargees: " << g_drone_images.size() << std::endl;
        g_drone_keypoints.resize(g_drone_images.size());
        g_drone_descriptors.resize(g_drone_images.size());
        g_drone_matches.resize(g_drone_images.size());
        g_drone_homographies.resize(g_drone_images.size());
        g_drone_inliers.resize(g_drone_images.size(), 0);
        g_drone_rectified_rgba.resize(g_drone_images.size());
        g_drone_projected_centers.resize(g_drone_images.size(), cv::Point2f(-1.0f, -1.0f));
    } else {
        std::cerr << "Aucune image drone trouvee dans dataset/frames." << std::endl;
    }

    glutDisplayFunc(Display);
    glutReshapeFunc(Reshape);
    glutKeyboardFunc(Keyboard);
    glutMouseFunc(Mouse);
    glutMotionFunc(Motion);
    glutPostRedisplay();
    glutMainLoop();

    return 0;
}
