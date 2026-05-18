#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>

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

    // fiecare thread se ocupa de un corp
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double ax = 0.0;
        double ay = 0.0;
        double az = 0.0;

        // calculeaza forta produsa de toate celelalte corpuri asupra lui i
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dz = bodies[j].z - bodies[i].z;

                double dist2 = dx * dx + dy * dy + dz * dz + SOFTENING;
                double dist = sqrt(dist2);
                double dist3 = dist2 * dist;

                // calcul acceleratie gravitationala pentru corpul i
                double forceOnI = G * bodies[j].mass / dist3;

                ax += forceOnI * dx;
                ay += forceOnI * dy;
                az += forceOnI * dz;
            }
        }

        bodies[i].ax = ax;
        bodies[i].ay = ay;
        bodies[i].az = az;
    }
}

// actualizare viteza si pozitie
void move_Bodies(vector<Body>& bodies) {
    int n = bodies.size();

    // fiecare corp este actualizat separat, deci se poate paraleliza simplu
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
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

    // setam cate thread-uri vrem sa folosim
    omp_set_num_threads(16);

    cout << "OpenMP N-body simulation" << endl;
    cout << "Bodies: " << N << " | Steps: " << STEPS << endl;
    cout << "Threads: " << omp_get_max_threads() << endl;
    cout << endl;
    cout.flush();

    double t_start = omp_get_wtime();

    for (int step = 0; step < STEPS; step++) {
        computeForces(bodies);
        move_Bodies(bodies);

        if (vis_interval > 0 && step % vis_interval == 0) {
            double elapsed = omp_get_wtime() - t_start;
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

    double elapsed = omp_get_wtime() - t_start;

    cout << endl;
    cout << "Done. Time: " << elapsed << " seconds" << endl;

    return 0;
}