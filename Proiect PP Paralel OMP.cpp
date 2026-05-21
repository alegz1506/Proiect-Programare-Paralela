#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <omp.h>

using namespace std;

constexpr double G = 1.0;
constexpr double TOTAL_MASS = 1.0;
constexpr double INITIAL_RADIUS = 1.0;
constexpr double CORE_RADIUS = 0.25;
constexpr double SOFTENING = 0.05;
constexpr double SOFTENING2 = SOFTENING * SOFTENING;
constexpr double DT = 0.001;
constexpr double DAMPING = 0.9999;

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
};

static inline double rand01() {
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

void init(vector<Body>& bodies) {
    srand(42);
    const int n = static_cast<int>(bodies.size());
    if (n == 0) return;

    double mass_sum = 0.0;

    for (int i = 0; i < n; ++i) {
        const double theta = 2.0 * M_PI * rand01();
        const double r = INITIAL_RADIUS * sqrt(rand01()) + 1e-4;

        bodies[i].x = r * cos(theta);
        bodies[i].y = r * sin(theta);
        bodies[i].z = (rand01() * 2.0 - 1.0) * 0.03;

        const double enclosed_mass = TOTAL_MASS * (r * r) / (r * r + CORE_RADIUS * CORE_RADIUS);
        const double v_circ = sqrt(G * enclosed_mass / sqrt(r * r + SOFTENING2));

        const double jitter = 0.01;
        bodies[i].vx = -sin(theta) * v_circ + (rand01() * 2.0 - 1.0) * jitter;
        bodies[i].vy =  cos(theta) * v_circ + (rand01() * 2.0 - 1.0) * jitter;
        bodies[i].vz = (rand01() * 2.0 - 1.0) * jitter * 0.2;

        bodies[i].ax = bodies[i].ay = bodies[i].az = 0.0;
        bodies[i].mass = 0.75 + 0.5 * rand01();
        mass_sum += bodies[i].mass;
    }

    for (auto& b : bodies) {
        b.mass = b.mass / mass_sum * TOTAL_MASS;
    }

    double cmx = 0.0, cmy = 0.0, cmz = 0.0;
    double cvx = 0.0, cvy = 0.0, cvz = 0.0;
    double total_mass = 0.0;

    for (const auto& b : bodies) {
        total_mass += b.mass;
        cmx += b.mass * b.x;
        cmy += b.mass * b.y;
        cmz += b.mass * b.z;
        cvx += b.mass * b.vx;
        cvy += b.mass * b.vy;
        cvz += b.mass * b.vz;
    }

    cmx /= total_mass; cmy /= total_mass; cmz /= total_mass;
    cvx /= total_mass; cvy /= total_mass; cvz /= total_mass;

    for (auto& b : bodies) {
        b.x -= cmx; b.y -= cmy; b.z -= cmz;
        b.vx -= cvx; b.vy -= cvy; b.vz -= cvz;
    }
}

void computeForces(vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double ax = 0.0;
        double ay = 0.0;
        double az = 0.0;

        const double xi = bodies[i].x;
        const double yi = bodies[i].y;
        const double zi = bodies[i].z;

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            const double dx = bodies[j].x - xi;
            const double dy = bodies[j].y - yi;
            const double dz = bodies[j].z - zi;

            const double dist2 = dx * dx + dy * dy + dz * dz + SOFTENING2;
            const double inv_dist = 1.0 / sqrt(dist2);
            const double inv_dist3 = inv_dist * inv_dist * inv_dist;
            const double s = G * bodies[j].mass * inv_dist3;

            ax += s * dx;
            ay += s * dy;
            az += s * dz;
        }

        bodies[i].ax = ax;
        bodies[i].ay = ay;
        bodies[i].az = az;
    }
}

void moveBodies(vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        bodies[i].vx = (bodies[i].vx + bodies[i].ax * DT) * DAMPING;
        bodies[i].vy = (bodies[i].vy + bodies[i].ay * DT) * DAMPING;
        bodies[i].vz = (bodies[i].vz + bodies[i].az * DT) * DAMPING;

        bodies[i].x += bodies[i].vx * DT;
        bodies[i].y += bodies[i].vy * DT;
        bodies[i].z += bodies[i].vz * DT;
    }
}

int main(int argc, char* argv[]) {
    int N = 10000;
    int STEPS = 1000;
    int vis_interval = 0;

    if (argc > 1) N = max(0, atoi(argv[1]));
    if (argc > 2) STEPS = max(0, atoi(argv[2]));
    if (argc > 3) vis_interval = max(0, atoi(argv[3]));

    vector<Body> bodies(static_cast<size_t>(N));
    init(bodies);

    omp_set_num_threads(16);
    const double t0 = omp_get_wtime();

    for (int step = 0; step < STEPS; ++step) {
        computeForces(bodies);
        moveBodies(bodies);

        if (vis_interval > 0 && step % vis_interval == 0) {
            const double elapsed = omp_get_wtime() - t0;

            cout << "STEP " << step << " " << elapsed;
            for (int i = 0; i < N; ++i) {
                cout << " " << bodies[i].x << " " << bodies[i].y << " " << bodies[i].z;
            }
            cout << "\n";
            cout.flush();
        }
    }

    return 0;
}
