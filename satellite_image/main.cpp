#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <filesystem>
#include <string>
#include <cstring>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>

#include <opencv2/opencv.hpp>

struct Texture {
    GLuint id = 0;
    int width = 0;
    int height = 0;
    bool valid = false;
};

struct Rect {
    float x, y, w, h;
};

static void draw_texture(const Texture& tex, const Rect& r, float rcol, float gcol, float bcol, float acol);

enum class ActiveField {
    None,
    BasePath,
    OverlayPath
};

static double g_mouse_x = 0.0;
static double g_mouse_y = 0.0;
static double g_last_mouse_x = 0.0;
static double g_last_mouse_y = 0.0;
static bool g_mouse_down = false;
static bool g_mouse_pressed = false;
static ActiveField g_active_field = ActiveField::None;

static std::string g_base_path;
static std::string g_overlay_path;

static bool point_in_rect(double x, double y, const Rect& r) {
    return x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
}

static void set_ortho(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, h, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

static void destroy_texture(Texture& tex) {
    if (tex.valid && tex.id != 0) {
        glDeleteTextures(1, &tex.id);
    }
    tex = Texture{};
}

static Texture load_texture(const std::string& path) {
    Texture tex;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        return tex;
    }

    cv::Mat rgba;
    if (image.channels() == 4) {
        cv::cvtColor(image, rgba, cv::COLOR_BGRA2RGBA);
    } else if (image.channels() == 3) {
        cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, rgba, cv::COLOR_GRAY2RGBA);
    } else {
        return tex;
    }

    cv::flip(rgba, rgba, 0);

    GLuint id = 0;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgba.cols, rgba.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
    glBindTexture(GL_TEXTURE_2D, 0);

    tex.id = id;
    tex.width = rgba.cols;
    tex.height = rgba.rows;
    tex.valid = true;
    return tex;
}

static void draw_text(float x, float y, const char* text, float r, float g, float b) {
    if (!text || text[0] == '\0') return;

    int ri = static_cast<int>(r * 255.0f + 0.5f);
    int gi = static_cast<int>(g * 255.0f + 0.5f);
    int bi = static_cast<int>(b * 255.0f + 0.5f);

    std::string key = std::string(text) + "|" + std::to_string(ri) + "," + std::to_string(gi) + "," + std::to_string(bi);
    static std::unordered_map<std::string, Texture> cache;

    auto it = cache.find(key);
    if (it == cache.end()) {
        const int pad = 4;
        int baseline = 0;
        double scale = 0.6;
        int thickness = 1;
        int font = cv::FONT_HERSHEY_SIMPLEX;
        cv::Size ts = cv::getTextSize(text, font, scale, thickness, &baseline);
        int w = ts.width + pad * 2;
        int h = ts.height + pad * 2 + baseline;

        cv::Mat bgra(h, w, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        cv::putText(bgra, text, cv::Point(pad, pad + ts.height), font, scale, cv::Scalar(bi, gi, ri, 255), thickness, cv::LINE_AA);

        cv::Mat rgba;
        cv::cvtColor(bgra, rgba, cv::COLOR_BGRA2RGBA);
        cv::flip(rgba, rgba, 0);

        Texture tex;
        GLuint id = 0;
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgba.cols, rgba.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
        glBindTexture(GL_TEXTURE_2D, 0);

        tex.id = id;
        tex.width = rgba.cols;
        tex.height = rgba.rows;
        tex.valid = true;

        it = cache.emplace(key, tex).first;
    }

    const Texture& tex = it->second;
    draw_texture(tex, {x, y, static_cast<float>(tex.width), static_cast<float>(tex.height)}, 1.0f, 1.0f, 1.0f, 1.0f);
}

static void draw_text_simple(float x, float y, const char* text, float r, float g, float b) {
    if (!text || text[0] == '\0') return;

    int ri = static_cast<int>(r * 255.0f + 0.5f);
    int gi = static_cast<int>(g * 255.0f + 0.5f);
    int bi = static_cast<int>(b * 255.0f + 0.5f);

    std::string key = std::string(text) + "|simple|" + std::to_string(ri) + "," + std::to_string(gi) + "," + std::to_string(bi);
    static std::unordered_map<std::string, Texture> cache;

    auto it = cache.find(key);
    if (it == cache.end()) {
        double scale = 0.6;
        int thickness = 1;
        int font = cv::FONT_HERSHEY_SIMPLEX;
        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, font, scale, thickness, &baseline);
        int w = ts.width + 2;
        int h = ts.height + 2;

        cv::Mat bgra(h, w, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        cv::putText(bgra, text, cv::Point(1, ts.height), font, scale, cv::Scalar(bi, gi, ri, 255), thickness, cv::LINE_AA);

        cv::Mat rgba;
        cv::cvtColor(bgra, rgba, cv::COLOR_BGRA2RGBA);
        cv::flip(rgba, rgba, 0);

        Texture tex;
        GLuint id = 0;
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgba.cols, rgba.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
        glBindTexture(GL_TEXTURE_2D, 0);

        tex.id = id;
        tex.width = rgba.cols;
        tex.height = rgba.rows;
        tex.valid = true;

        it = cache.emplace(key, tex).first;
    }

    const Texture& tex = it->second;
    draw_texture(tex, {x, y, static_cast<float>(tex.width), static_cast<float>(tex.height)}, 1.0f, 1.0f, 1.0f, 1.0f);
}

static void get_text_tex_size(const char* text, int& w, int& h) {
    const int pad = 4;
    int baseline = 0;
    double scale = 0.6;
    int thickness = 1;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::Size ts = cv::getTextSize(text ? text : "", font, scale, thickness, &baseline);
    w = ts.width + pad * 2;
    h = ts.height + pad * 2 + baseline;
}

static int get_text_line_height() {
    int w = 0, h = 0;
    get_text_tex_size("Ag", w, h);
    return h;
}

static void draw_rect(const Rect& r, float rcol, float gcol, float bcol, float acol) {
    glColor4f(rcol, gcol, bcol, acol);
    glBegin(GL_QUADS);
    glVertex2f(r.x, r.y);
    glVertex2f(r.x + r.w, r.y);
    glVertex2f(r.x + r.w, r.y + r.h);
    glVertex2f(r.x, r.y + r.h);
    glEnd();
}

static void draw_texture(const Texture& tex, const Rect& r, float rcol, float gcol, float bcol, float acol) {
    if (!tex.valid) return;
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindTexture(GL_TEXTURE_2D, tex.id);
    glColor4f(rcol, gcol, bcol, acol);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(r.x, r.y);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(r.x + r.w, r.y);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(r.x + r.w, r.y + r.h);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(r.x, r.y + r.h);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_BLEND);
    glDisable(GL_TEXTURE_2D);
}

static void draw_texture_rot(const Texture& tex, const Rect& r, float rcol, float gcol, float bcol, float acol, float angle_deg) {
    if (!tex.valid) return;
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindTexture(GL_TEXTURE_2D, tex.id);
    glColor4f(rcol, gcol, bcol, acol);

    float cx = r.x + r.w * 0.5f;
    float cy = r.y + r.h * 0.5f;

    glPushMatrix();
    glTranslatef(cx, cy, 0.0f);
    glRotatef(angle_deg, 0.0f, 0.0f, 1.0f);
    glTranslatef(-cx, -cy, 0.0f);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(r.x, r.y);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(r.x + r.w, r.y);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(r.x + r.w, r.y + r.h);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(r.x, r.y + r.h);
    glEnd();

    glPopMatrix();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_BLEND);
    glDisable(GL_TEXTURE_2D);
}

static bool draw_button(const Rect& r, const char* label) {
    bool hover = point_in_rect(g_mouse_x, g_mouse_y, r);
    bool pressed = hover && g_mouse_down;
    
    // Main button background
    float baseColor = hover ? 0.28f : 0.22f;
    draw_rect(r, baseColor, baseColor, baseColor, 1.0f);
    
    // Top and left highlight (beveled effect)
    draw_rect({r.x, r.y, r.w, 2.0f}, 0.45f, 0.45f, 0.45f, 1.0f);
    draw_rect({r.x, r.y, 2.0f, r.h}, 0.45f, 0.45f, 0.45f, 1.0f);
    
    // Bottom and right shadow (beveled effect)
    draw_rect({r.x, r.y + r.h - 2.0f, r.w, 2.0f}, 0.08f, 0.08f, 0.08f, 1.0f);
    draw_rect({r.x + r.w - 2.0f, r.y, 2.0f, r.h}, 0.08f, 0.08f, 0.08f, 1.0f);
    
    // Inner content area
    float inset = pressed ? 2.0f : 1.0f;
    draw_rect({r.x + inset, r.y + inset, r.w - inset * 2, r.h - inset * 2}, 0.15f, 0.15f, 0.15f, 1.0f);
    
    // Text (without background)
    double scale = 0.6;
    int thickness = 1;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    int baseline = 0;
    cv::Size ts = cv::getTextSize(label, font, scale, thickness, &baseline);
    float tx = r.x + (r.w - ts.width) * 0.5f;
    float ty = r.y + (r.h - ts.height) * 0.5f + 2.0f + (pressed ? 1.0f : 0.0f);
    float textColor = hover ? 1.0f : 0.9f;
    draw_text_simple(tx, ty, label, textColor, textColor, textColor);
    
    return hover && g_mouse_pressed;
}

static void draw_input(const Rect& r, const std::string& text, const char* label, bool active) {
    if (label && label[0] != '\0') {
        draw_text(r.x, r.y - 14, label, 0.8f, 0.8f, 0.8f);
    }
    float border = active ? 0.5f : 0.25f;
    draw_rect(r, border, border, border, 1.0f);
    draw_rect({r.x + 1, r.y + 1, r.w - 2, r.h - 2}, 0.08f, 0.08f, 0.08f, 1.0f);
    int tw = 0, th = 0;
    get_text_tex_size(text.c_str(), tw, th);
    float tx = r.x + 8.0f;
    float ty = r.y + (r.h - th) * 0.5f;
    draw_text(tx, ty, text.c_str(), 0.9f, 0.9f, 0.9f);
}

static void char_callback(GLFWwindow*, unsigned int codepoint) {
    if (g_active_field == ActiveField::None) return;
    if (codepoint < 32 || codepoint > 126) return;
    if (g_active_field == ActiveField::BasePath) {
        g_base_path.push_back(static_cast<char>(codepoint));
    } else if (g_active_field == ActiveField::OverlayPath) {
        g_overlay_path.push_back(static_cast<char>(codepoint));
    }
}

static void key_callback(GLFWwindow*, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    if (g_active_field == ActiveField::None) return;

    std::string* target = nullptr;
    if (g_active_field == ActiveField::BasePath) target = &g_base_path;
    if (g_active_field == ActiveField::OverlayPath) target = &g_overlay_path;
    if (!target) return;

    if (key == GLFW_KEY_BACKSPACE) {
        if (!target->empty()) target->pop_back();
    }
}

static bool file_exists(const std::string& path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

static bool try_set_default_base(std::string& path_out) {
    std::vector<std::string> candidates = {
        "../Village/map_village.jpg",
        "Drone/Village/map_village.jpg"
    };
    for (const auto& p : candidates) {
        if (file_exists(p)) {
            path_out = p;
            return true;
        }
    }
    return false;
}

static bool auto_locate_overlay(
    const cv::Mat& base_gray,
    const cv::Mat& overlay_gray,
    float& out_anchor_x,
    float& out_anchor_y,
    float& out_scale,
    float& out_rot_deg,
    std::string& out_status
) {
    // Resize base to 4096px — test showed this is optimal for AKAZE on aerial imagery
    const int BASE_MAX = 4096;
    double sb = std::min(1.0, (double)BASE_MAX / std::max(base_gray.cols, base_gray.rows));
    cv::Mat base_ds;
    cv::resize(base_gray, base_ds, cv::Size(), sb, sb, cv::INTER_AREA);

    // AKAZE: non-linear scale space, much more robust than ORB for aerial matching.
    // Low threshold = more keypoints detected across the wide map.
    auto akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.0003f);
    std::vector<cv::KeyPoint> kp_base;
    cv::Mat desc_base;
    akaze->detectAndCompute(base_ds, cv::noArray(), kp_base, desc_base);
    if (desc_base.empty()) {
        out_status = "Match failed: no features in base map";
        return false;
    }

    // Try overlay at multiple sizes — at one of these sizes the feature scale will
    // align with the base, giving a high inlier ratio
    const int overlay_sizes[] = {500, 768, 1000, 1500, 2000};

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    int   best_inliers = 0;
    float best_score   = 0.0f;
    cv::Mat  best_H;
    cv::Size best_base_size = base_ds.size();
    cv::Size best_overlay_size;

    for (int omax : overlay_sizes) {
        double so = std::min(1.0, (double)omax / std::max(overlay_gray.cols, overlay_gray.rows));
        cv::Mat overlay_ds;
        cv::resize(overlay_gray, overlay_ds, cv::Size(), so, so, cv::INTER_AREA);

        std::vector<cv::KeyPoint> kp_overlay;
        cv::Mat desc_overlay;
        akaze->detectAndCompute(overlay_ds, cv::noArray(), kp_overlay, desc_overlay);
        if (desc_overlay.empty()) continue;

        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(desc_overlay, desc_base, knn, 2);

        std::vector<cv::DMatch> good;
        for (auto& m : knn)
            if (m.size() == 2 && m[0].distance < 0.80f * m[1].distance)
                good.push_back(m[0]);

        if ((int)good.size() < 8) continue;

        std::vector<cv::Point2f> pts_o, pts_b;
        for (auto& m : good) {
            pts_o.push_back(kp_overlay[m.queryIdx].pt);
            pts_b.push_back(kp_base[m.trainIdx].pt);
        }

        std::vector<uchar> mask;
        cv::Mat H = cv::findHomography(pts_o, pts_b, cv::RANSAC, 8.0, mask, 10000, 0.9999);
        if (H.empty()) continue;

        int inliers = 0;
        for (auto v : mask) if (v) ++inliers;
        float ratio = (float)inliers / (float)good.size();

        if (inliers < 8 || ratio < 0.10f) continue;

        // Geometric validation: convex quad, scale in range, center in image
        std::vector<cv::Point2f> co = {
            {0.f, 0.f}, {(float)overlay_ds.cols, 0.f},
            {(float)overlay_ds.cols, (float)overlay_ds.rows}, {0.f, (float)overlay_ds.rows}
        };
        std::vector<cv::Point2f> cb;
        cv::perspectiveTransform(co, cb, H);

        bool convex = true;
        float sign = 0.0f;
        for (int i = 0; i < 4 && convex; ++i) {
            cv::Point2f e1 = cb[(i+1)%4] - cb[i];
            cv::Point2f e2 = cb[(i+2)%4] - cb[(i+1)%4];
            float c = e1.x*e2.y - e1.y*e2.x;
            if (i == 0) sign = c;
            else if (c * sign < 0) convex = false;
        }
        if (!convex) continue;

        float dxe = cb[1].x - cb[0].x, dye = cb[1].y - cb[0].y;
        float mw = std::sqrt(dxe*dxe + dye*dye);
        float sr = mw / (float)base_ds.cols;
        if (sr < 0.02f || sr > 0.95f) continue;

        float cx = (cb[0].x+cb[1].x+cb[2].x+cb[3].x)*0.25f;
        float cy = (cb[0].y+cb[1].y+cb[2].y+cb[3].y)*0.25f;
        if (cx < 0 || cx > base_ds.cols || cy < 0 || cy > base_ds.rows) continue;

        float score = ratio * (float)std::min(inliers, 50);
        if (score > best_score) {
            best_score        = score;
            best_inliers      = inliers;
            best_H            = H.clone();
            best_overlay_size = overlay_ds.size();
        }
    }

    if (best_H.empty()) {
        out_status = "Match failed: no valid homography found";
        return false;
    }

    // Decompose best result
    std::vector<cv::Point2f> co = {
        {0.f, 0.f}, {(float)best_overlay_size.width, 0.f},
        {(float)best_overlay_size.width, (float)best_overlay_size.height},
        {0.f, (float)best_overlay_size.height}
    };
    std::vector<cv::Point2f> cb;
    cv::perspectiveTransform(co, cb, best_H);

    float cx = (cb[0].x+cb[1].x+cb[2].x+cb[3].x)*0.25f;
    float cy = (cb[0].y+cb[1].y+cb[2].y+cb[3].y)*0.25f;
    out_anchor_x = cx / (float)best_base_size.width;
    out_anchor_y = cy / (float)best_base_size.height;

    auto edge_len = [](cv::Point2f a, cv::Point2f b){
        float dx=b.x-a.x, dy=b.y-a.y; return std::sqrt(dx*dx+dy*dy);
    };
    float mapped_w = (edge_len(cb[0],cb[1]) + edge_len(cb[3],cb[2])) * 0.5f;
    out_scale = mapped_w / (float)best_base_size.width;

    float dx = cb[1].x-cb[0].x, dy = cb[1].y-cb[0].y;
    out_rot_deg = (float)(std::atan2((double)dy,(double)dx)*180.0/CV_PI);

    char buf[80];
    std::snprintf(buf, sizeof(buf), "Located! %d inliers (%.0f%%)",
                  best_inliers, best_score / (float)std::min(best_inliers,50) * 100.0f);
    out_status = buf;
    return true;
}

static bool open_file_dialog(std::string& path_out) {
    // Try zenity
    const char* cmd = "zenity --file-selection --title='Select Overlay Image' "
                      "--file-filter='Image files|*.jpg *.jpeg *.png *.bmp *.gif *.tiff' "
                      "--file-filter='All files|*'";
    FILE* pipe = popen(cmd, "r");
    
    if (pipe) {
        char buffer[2048] = {0};
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            size_t len = std::strlen(buffer);
            while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r')) {
                buffer[len - 1] = '\0';
                len--;
            }
            if (len > 0) {
                path_out = buffer;
                std::fprintf(stdout, "Selected file: %s\n", path_out.c_str());
                fflush(stdout);
                pclose(pipe);
                return true;
            }
        }
        pclose(pipe);
    }
    
    // Try kdialog
    cmd = "kdialog --getopenfilename ~/ '*.jpg *.jpeg *.png *.bmp *.gif *.tiff|Image Files'";
    pipe = popen(cmd, "r");
    
    if (pipe) {
        char buffer[2048] = {0};
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            size_t len = std::strlen(buffer);
            while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r')) {
                buffer[len - 1] = '\0';
                len--;
            }
            if (len > 0) {
                path_out = buffer;
                std::fprintf(stdout, "Selected file (kdialog): %s\n", path_out.c_str());
                fflush(stdout);
                pclose(pipe);
                return true;
            }
        }
        pclose(pipe);
    }
    
    // Try xdg-open with a file manager
    cmd = "xdg-open ~/ 2>/dev/null &";
    system(cmd);
    
    // Fallback: use find to list image files and use fzf if available
    cmd = "find ~ -maxdepth 5 -type f \\( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.gif' -o -iname '*.tiff' \\) 2>/dev/null | fzf --preview 'file {}' 2>/dev/null";
    pipe = popen(cmd, "r");
    
    if (pipe) {
        char buffer[2048] = {0};
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            size_t len = std::strlen(buffer);
            while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r')) {
                buffer[len - 1] = '\0';
                len--;
            }
            if (len > 0) {
                path_out = buffer;
                std::fprintf(stdout, "Selected file (fzf): %s\n", path_out.c_str());
                fflush(stdout);
                pclose(pipe);
                return true;
            }
        }
        pclose(pipe);
    }
    
    std::fprintf(stdout, "No file dialog available - please install zenity, kdialog, or fzf\n");
    fflush(stdout);
    return false;
}

int main() {
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to init GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    GLFWwindow* window = glfwCreateWindow(1600, 1000, "Mapping App (OpenGL)", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetCharCallback(window, char_callback);
    glfwSetKeyCallback(window, key_callback);

    Texture base_tex;
    Texture overlay_tex;

    bool overlay_placed = false;
    float overlay_anchor_x = 0.5f;
    float overlay_anchor_y = 0.5f;
    float overlay_scale = 0.25f;
    float overlay_opacity = 0.6f;
    float overlay_rotation_deg = 0.0f;
    float base_scale = 1.0f;
    float base_pan_x = 0.0f;
    float base_pan_y = 0.0f;

    cv::Mat base_gray_cv;
    cv::Mat overlay_gray_cv;
    std::string auto_locate_status;
    bool locating = false;

    g_base_path = "";
    g_overlay_path = "";
    try_set_default_base(g_base_path);

    if (!g_base_path.empty()) {
        base_tex = load_texture(g_base_path);
        if (base_tex.valid)
            base_gray_cv = cv::imread(g_base_path, cv::IMREAD_GRAYSCALE);
    }

    g_last_mouse_x = 0.0;
    g_last_mouse_y = 0.0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        int win_w, win_h;
        glfwGetFramebufferSize(window, &win_w, &win_h);
        set_ortho(win_w, win_h);

        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        bool mouse_now = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        g_mouse_pressed = mouse_now && !g_mouse_down;
        g_mouse_down = mouse_now;
        g_mouse_x = mx;
        g_mouse_y = my;

        glClearColor(0.06f, 0.06f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        const float sidebar_w = 380.0f;
        const float margin = 16.0f;
        Rect sidebar = {static_cast<float>(win_w) - sidebar_w, 0.0f, sidebar_w, static_cast<float>(win_h)};
        draw_rect(sidebar, 0.12f, 0.12f, 0.13f, 1.0f);

        Rect canvas = {margin, margin, static_cast<float>(win_w) - sidebar_w - margin * 2, static_cast<float>(win_h) - margin * 2};
        draw_rect(canvas, 0.08f, 0.08f, 0.09f, 1.0f);

        // Enable scissor test to clip content to canvas area
        glEnable(GL_SCISSOR_TEST);
        int scissor_x = static_cast<int>(canvas.x);
        int scissor_y = static_cast<int>(win_h - canvas.y - canvas.h);
        int scissor_w = static_cast<int>(canvas.w);
        int scissor_h = static_cast<int>(canvas.h);
        glScissor(scissor_x, scissor_y, scissor_w, scissor_h);

        Rect base_rect = canvas;
        if (base_tex.valid) {
            float canvas_aspect = canvas.w / canvas.h;
            float tex_aspect = static_cast<float>(base_tex.width) / static_cast<float>(base_tex.height);
            if (tex_aspect > canvas_aspect) {
                base_rect.w = canvas.w;
                base_rect.h = canvas.w / tex_aspect;
                base_rect.x = canvas.x;
                base_rect.y = canvas.y + (canvas.h - base_rect.h) * 0.5f;
            } else {
                base_rect.h = canvas.h;
                base_rect.w = canvas.h * tex_aspect;
                base_rect.x = canvas.x + (canvas.w - base_rect.w) * 0.5f;
                base_rect.y = canvas.y;
            }
            
            // Apply zoom
            float prev_w = base_rect.w;
            float prev_h = base_rect.h;
            base_rect.w *= base_scale;
            base_rect.h *= base_scale;
            base_rect.x = canvas.x + (canvas.w - base_rect.w) * 0.5f + base_pan_x;
            base_rect.y = canvas.y + (canvas.h - base_rect.h) * 0.5f + base_pan_y;
            
            draw_texture(base_tex, base_rect, 1.0f, 1.0f, 1.0f, 1.0f);
        } else {
            draw_text(canvas.x + 10, canvas.y + 20, "No base image loaded", 0.8f, 0.8f, 0.8f);
        }

        if (overlay_tex.valid && overlay_placed && base_tex.valid) {
            float overlay_w = base_rect.w * overlay_scale;
            float overlay_h = overlay_w * (static_cast<float>(overlay_tex.height) / static_cast<float>(overlay_tex.width));
            float anchor_px = base_rect.x + overlay_anchor_x * base_rect.w;
            float anchor_py = base_rect.y + overlay_anchor_y * base_rect.h;

            Rect ov = {anchor_px - overlay_w * 0.5f, anchor_py - overlay_h * 0.5f, overlay_w, overlay_h};
            draw_texture_rot(overlay_tex, ov, 1.0f, 0.7f, 1.0f, overlay_opacity, overlay_rotation_deg);
        }

        bool cursor_on_canvas = !point_in_rect(g_mouse_x, g_mouse_y, sidebar);
        if (cursor_on_canvas && g_mouse_pressed && base_tex.valid && point_in_rect(g_mouse_x, g_mouse_y, base_rect)) {
            overlay_anchor_x = static_cast<float>((g_mouse_x - base_rect.x) / base_rect.w);
            overlay_anchor_y = static_cast<float>((g_mouse_y - base_rect.y) / base_rect.h);
        }

        // Handle map panning when zoomed in
        if (cursor_on_canvas && g_mouse_down && base_scale > 1.0f && base_tex.valid && point_in_rect(g_mouse_x, g_mouse_y, canvas)) {
            float delta_x = static_cast<float>(g_mouse_x - g_last_mouse_x);
            float delta_y = static_cast<float>(g_mouse_y - g_last_mouse_y);
            base_pan_x += delta_x;
            base_pan_y += delta_y;
        }

        g_last_mouse_x = g_mouse_x;
        g_last_mouse_y = g_mouse_y;

        // Disable scissor test for sidebar rendering
        glDisable(GL_SCISSOR_TEST);
        
        // Redraw sidebar background to ensure it's on top
        draw_rect(sidebar, 0.12f, 0.12f, 0.13f, 1.0f);

        float sx = sidebar.x + 24.0f;
        float sy = 40.0f;
        float input_w = sidebar.w - 48.0f;
        float line_h = static_cast<float>(get_text_line_height());
        float row_gap = 20.0f;
        float button_gap = 6.0f;
        float block_gap = 24.0f;
        float button_h = 36.0f;

        // Browse for overlay image
        Rect browse_btn = {sx, sy, input_w, button_h};
        if (draw_button(browse_btn, "Browse overlay image")) {
            std::fprintf(stdout, "Browse button clicked!\n");
            fflush(stdout);
            std::string selected_path;
            if (open_file_dialog(selected_path)) {
                std::fprintf(stdout, "Dialog returned path: %s\n", selected_path.c_str());
                fflush(stdout);
                if (file_exists(selected_path)) {
                    std::fprintf(stdout, "File exists, loading texture\n");
                    fflush(stdout);
                    g_overlay_path = selected_path;
                    destroy_texture(overlay_tex);
                    overlay_tex = load_texture(g_overlay_path);
                    if (overlay_tex.valid)
                        overlay_gray_cv = cv::imread(g_overlay_path, cv::IMREAD_GRAYSCALE);
                    auto_locate_status.clear();
                    std::fprintf(stdout, "Texture loaded: valid=%d\n", overlay_tex.valid);
                    fflush(stdout);
                } else {
                    std::fprintf(stdout, "File does not exist: %s\n", selected_path.c_str());
                    fflush(stdout);
                }
            } else {
                std::fprintf(stdout, "Dialog cancelled or failed\n");
                fflush(stdout);
            }
        }

        sy += button_h + row_gap;

        Rect place_btn = {sx, sy, input_w, button_h};
        if (draw_button(place_btn, "Place overlay")) {
            if (overlay_tex.valid) {
                overlay_placed = true;
            }
        }

        sy += button_h + button_gap;

        {
            Rect al_btn = {sx, sy, input_w, button_h};
            bool can_locate = base_tex.valid && overlay_tex.valid &&
                              !base_gray_cv.empty() && !overlay_gray_cv.empty();
            if (can_locate) {
                const char* btn_label = locating ? "Locating..." : "Auto Locate";
                if (!locating && draw_button(al_btn, btn_label)) {
                    locating = true;
                    auto_locate_status = "Locating...";
                } else if (locating) {
                    draw_rect(al_btn, 0.22f, 0.22f, 0.22f, 1.0f);
                    draw_text_simple(al_btn.x + 8, al_btn.y + 12, btn_label, 0.9f, 0.7f, 0.3f);
                }
            } else {
                draw_rect(al_btn, 0.15f, 0.15f, 0.16f, 1.0f);
                draw_rect({al_btn.x, al_btn.y, al_btn.w, 2.0f}, 0.2f, 0.2f, 0.2f, 1.0f);
                draw_text_simple(al_btn.x + 8, al_btn.y + 12, "Auto Locate", 0.4f, 0.4f, 0.4f);
            }
            sy += button_h + button_gap;
            if (!auto_locate_status.empty()) {
                bool ok = auto_locate_status.rfind("Located", 0) == 0;
                draw_text(sx, sy, auto_locate_status.c_str(),
                          ok ? 0.4f : 0.9f, ok ? 0.9f : 0.4f, 0.4f);
                sy += line_h + 4.0f;
            }
        }

        sy += row_gap;

        float split_gap = 8.0f;
        
        // Position adjustment controls
        if (overlay_placed && overlay_tex.valid) {
            // Display current position
            char pos_text[64];
            std::snprintf(pos_text, sizeof(pos_text), "Position: X=%.1f%% Y=%.1f%%", 
                         overlay_anchor_x * 100.0f, overlay_anchor_y * 100.0f);
            draw_text(sx, sy, pos_text, 0.75f, 0.75f, 0.75f);
            sy += line_h + 8.0f;
            
            // Up button
            Rect up_btn = {sx + input_w * 0.35f, sy, input_w * 0.3f, button_h - 8.0f};
            if (draw_button(up_btn, "up")) {
                overlay_anchor_y = std::max(0.0f, overlay_anchor_y - 0.02f);
            }
            sy += button_h - 2.0f + button_gap;
            
            // Left, Center, Right buttons
            float third_w = (input_w - split_gap * 2) / 3.0f;
            Rect left_btn = {sx, sy, third_w, button_h - 8.0f};
            Rect center_btn = {sx + third_w + split_gap, sy, third_w, button_h - 8.0f};
            Rect right_btn = {sx + third_w * 2 + split_gap * 2, sy, third_w, button_h - 8.0f};
            
            if (draw_button(left_btn, "left")) {
                overlay_anchor_x = std::max(0.0f, overlay_anchor_x - 0.02f);
            }
            if (draw_button(center_btn, "center")) {
                overlay_anchor_x = 0.5f;
                overlay_anchor_y = 0.5f;
            }
            if (draw_button(right_btn, "right")) {
                overlay_anchor_x = std::min(1.0f, overlay_anchor_x + 0.02f);
            }
            sy += button_h - 2.0f + button_gap;
            
            // Down button
            Rect down_btn = {sx + input_w * 0.35f, sy, input_w * 0.3f, button_h - 8.0f};
            if (draw_button(down_btn, "down")) {
                overlay_anchor_y = std::min(1.0f, overlay_anchor_y + 0.02f);
            }
            sy += button_h;
        }

        sy += row_gap;
        // Overlay size controls
        float half_w = (input_w - split_gap) * 0.5f;
        Rect overlay_scale_down = {sx, sy, half_w, button_h};
        Rect overlay_scale_up = {sx + half_w + split_gap, sy, half_w, button_h};
        if (draw_button(overlay_scale_down, "Overlay Size -")) {
            overlay_scale = std::max(0.02f, overlay_scale - 0.02f);
        }
        if (draw_button(overlay_scale_up, "Overlay Size +")) {
            overlay_scale = std::min(2.0f, overlay_scale + 0.02f);
        }

        sy += button_h + 8.0f;
        char overlay_scale_text[48];
        std::snprintf(overlay_scale_text, sizeof(overlay_scale_text), "Overlay size: %.0f%%", overlay_scale * 100.0f);
        draw_text(sx, sy, overlay_scale_text, 0.8f, 0.8f, 0.8f);

        sy += row_gap;
        // Overlay opacity controls
        Rect overlay_opacity_down = {sx, sy, half_w, button_h};
        Rect overlay_opacity_up = {sx + half_w + split_gap, sy, half_w, button_h};
        if (draw_button(overlay_opacity_down, "Opacity -")) {
            overlay_opacity = std::max(0.05f, overlay_opacity - 0.05f);
        }
        if (draw_button(overlay_opacity_up, "Opacity +")) {
            overlay_opacity = std::min(1.0f, overlay_opacity + 0.05f);
        }

        sy += button_h + 8.0f;
        char overlay_opacity_text[48];
        std::snprintf(overlay_opacity_text, sizeof(overlay_opacity_text), "Overlay opacity: %.0f%%", overlay_opacity * 100.0f);
        draw_text(sx, sy, overlay_opacity_text, 0.8f, 0.8f, 0.8f);

        sy += row_gap;
        // Overlay rotation controls
        Rect overlay_rot_down = {sx, sy, half_w, button_h};
        Rect overlay_rot_up = {sx + half_w + split_gap, sy, half_w, button_h};
        if (draw_button(overlay_rot_down, "Rotate -")) {
            overlay_rotation_deg -= 2.0f;
        }
        if (draw_button(overlay_rot_up, "Rotate +")) {
            overlay_rotation_deg += 2.0f;
        }

        sy += button_h + 8.0f;
        Rect overlay_rot_reset = {sx, sy, input_w, button_h};
        if (draw_button(overlay_rot_reset, "Reset rotation")) {
            overlay_rotation_deg = 0.0f;
        }

        sy += button_h + 8.0f;
        char overlay_rotation_text[48];
        std::snprintf(overlay_rotation_text, sizeof(overlay_rotation_text), "Rotation: %.1f deg", overlay_rotation_deg);
        draw_text(sx, sy, overlay_rotation_text, 0.8f, 0.8f, 0.8f);

        sy += row_gap;
        Rect scale_down = {sx, sy, half_w, button_h};
        Rect scale_up = {sx + half_w + split_gap, sy, half_w, button_h};
        if (draw_button(scale_down, "Zoom -")) {
            base_scale = std::max(0.5f, base_scale - 0.1f);
        }
        if (draw_button(scale_up, "Zoom +")) {
            base_scale = std::min(3.0f, base_scale + 0.1f);
        }

        sy += button_h + 8.0f;
        char scale_text[32];
        std::snprintf(scale_text, sizeof(scale_text), "Zoom: %.0f%%", base_scale * 100.0f);
        draw_text(sx, sy, scale_text, 0.8f, 0.8f, 0.8f);

        sy += 24.0f;
        draw_text(sx, sy, "Overlay preview", 0.85f, 0.85f, 0.85f);
        sy += line_h + 8.0f;
        Rect preview = {sx, sy, input_w, 200.0f};
        draw_rect(preview, 0.1f, 0.1f, 0.1f, 1.0f);
        if (overlay_tex.valid) {
            float prev_aspect = preview.w / preview.h;
            float tex_aspect = static_cast<float>(overlay_tex.width) / static_cast<float>(overlay_tex.height);
            Rect prev_rect = preview;
            if (tex_aspect > prev_aspect) {
                prev_rect.h = preview.w / tex_aspect;
                prev_rect.y = preview.y + (preview.h - prev_rect.h) * 0.5f;
            } else {
                prev_rect.w = preview.h * tex_aspect;
                prev_rect.x = preview.x + (preview.w - prev_rect.w) * 0.5f;
            }
            draw_texture(overlay_tex, prev_rect, 1.0f, 1.0f, 1.0f, 1.0f);
        }

        if (g_mouse_pressed && !point_in_rect(g_mouse_x, g_mouse_y, sidebar)) {
            g_active_field = ActiveField::None;
        }

        glfwSwapBuffers(window);

        // Run matching after swap so the "Locating..." frame is visible before the block
        if (locating) {
            float new_ax = overlay_anchor_x, new_ay = overlay_anchor_y;
            float new_sc = overlay_scale, new_rot = overlay_rotation_deg;
            if (auto_locate_overlay(base_gray_cv, overlay_gray_cv,
                                    new_ax, new_ay, new_sc, new_rot, auto_locate_status)) {
                overlay_anchor_x     = std::max(0.0f, std::min(1.0f, new_ax));
                overlay_anchor_y     = std::max(0.0f, std::min(1.0f, new_ay));
                overlay_scale        = std::max(0.02f, std::min(2.0f, new_sc));
                overlay_rotation_deg = new_rot;
                overlay_placed       = true;
            }
            locating = false;
        }
    }

    destroy_texture(base_tex);
    destroy_texture(overlay_tex);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
