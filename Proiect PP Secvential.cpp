#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

// constante
const double G = 6.674e-11;
const double SOFTENING = 1e-9;
const double DT = 0.01;

// structura corpurilor
struct Body {
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
};

// initializare valori random de pozitie si viteza, cu acceleratie 0
void init(vector<Body>& bodies) {
    srand(42);

    for (int i = 0; i < bodies.size(); i++) {
        bodies[i].x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        bodies[i].y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        bodies[i].z = (double)rand() / RAND_MAX * 2.0 - 1.0;

        bodies[i].vx = (double)rand() / RAND_MAX * 0.1;
        bodies[i].vy = (double)rand() / RAND_MAX * 0.1;
        bodies[i].vz = (double)rand() / RAND_MAX * 0.1;

        bodies[i].ax = 0.0;
        bodies[i].ay = 0.0;
        bodies[i].az = 0.0;

        bodies[i].mass = 1e24 + (double)rand() / RAND_MAX * 1e26;
    }
}

// calculare atractie gravitationala intre corpuri
void computeForces(vector<Body>& bodies) {
    int n = bodies.size();

    // le aduce la valoare initiala 0
    for (int i = 0; i < n; i++) {
        bodies[i].ax = 0.0;
        bodies[i].ay = 0.0;
        bodies[i].az = 0.0;
    }

    // face calcul pentru fiecare pereche de corpuri
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;

            double dist2 = dx * dx + dy * dy + dz * dz + SOFTENING;
            double dist = sqrt(dist2);
            double dist3 = dist2 * dist;

            // calcul acceleratie gravitationala pentru fiecare corp
            double forceOnI = G * bodies[j].mass / dist3;
            double forceOnJ = G * bodies[i].mass / dist3;

            bodies[i].ax += forceOnI * dx;
            bodies[i].ay += forceOnI * dy;
            bodies[i].az += forceOnI * dz;

            bodies[j].ax -= forceOnJ * dx;
            bodies[j].ay -= forceOnJ * dy;
            bodies[j].az -= forceOnJ * dz;
        }
    }
}

// actualizare viteza si pozitie
void move_Bodies(vector<Body>& bodies) {
    for (int i = 0; i < bodies.size(); i++) {
        bodies[i].vx = bodies[i].vx + bodies[i].ax * DT;
        bodies[i].vy = bodies[i].vy + bodies[i].ay * DT;
        bodies[i].vz = bodies[i].vz + bodies[i].az * DT;

        bodies[i].x = bodies[i].x + bodies[i].vx * DT;
        bodies[i].y = bodies[i].y + bodies[i].vy * DT;
        bodies[i].z = bodies[i].z + bodies[i].vz * DT;
    }
}

int main(int argc, char* argv[]) {
    long int N = 10000;
    long int STEPS = 1000;
    long int vis_interval = 0;

    if (argc > 1) N = atol(argv[1]);
    if (argc > 2) STEPS = atol(argv[2]);
    if (argc > 3) vis_interval = atol(argv[3]);

    vector<Body> bodies(N);

    init(bodies);

    cout << "Sequential N-body simulation" << endl;
    cout << "Bodies: " << N << " | Steps: " << STEPS << endl;
    cout << endl;
    cout.flush();

    auto t_start = chrono::high_resolution_clock::now();

    for (int step = 0; step < STEPS; step++) {
        computeForces(bodies);
        move_Bodies(bodies);

        if (vis_interval > 0 && step % vis_interval == 0) {
            auto t_now = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(t_now - t_start).count();
            cout << "STEP " << step << " " << elapsed;
            for (int i = 0; i < N; i++) {
                cout << " " << bodies[i].x << " " << bodies[i].y << " " << bodies[i].z;
            }
            cout << "\n";
            cout.flush();
        } else if (vis_interval == 0 && step % 100 == 0) {
            cout << "Step " << step << " | body[0] position: (";
            cout << bodies[0].x << ", ";
            cout << bodies[0].y << ", ";
            cout << bodies[0].z << ")" << endl;
        }
    }

    auto t_end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t_end - t_start).count();

    cout << endl;
    cout << "Done. Time: " << elapsed << " seconds" << endl;

    return 0;
}