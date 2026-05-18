#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// constante
const double G = 6.674e-11;
const double SOFTENING = 1e-9;
const double DT = 0.01;

const int THREADS_PER_BLOCK = 256;

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

// calculare atractie gravitationala intre corpuri pe GPU
__global__ void computeForcesKernel(Body* bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // verificam sa nu iesim din vector
    if (i < n) {
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

// actualizare viteza si pozitie pe GPU
__global__ void moveBodiesKernel(Body* bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // fiecare corp este actualizat separat
    if (i < n) {
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

    Body* d_bodies;

    // alocare memorie pe GPU
    cudaMalloc((void**)&d_bodies, N * sizeof(Body));

    // copiere date din CPU pe GPU
    cudaMemcpy(d_bodies, bodies.data(), N * sizeof(Body), cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cout << "CUDA N-body simulation" << endl;
    cout << "Bodies: " << N << " | Steps: " << STEPS << endl;
    cout << "Blocks: " << blocks << " | Threads per block: " << THREADS_PER_BLOCK << endl;
    cout << endl;
    cout.flush();

    cudaEvent_t cuda_start, cuda_end;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_end);

    auto wall_start = chrono::high_resolution_clock::now();
    cudaEventRecord(cuda_start);

    for (int step = 0; step < STEPS; step++) {
        computeForcesKernel<<<blocks, THREADS_PER_BLOCK>>>(d_bodies, N);
        moveBodiesKernel<<<blocks, THREADS_PER_BLOCK>>>(d_bodies, N);

        if (vis_interval > 0 && step % vis_interval == 0) {
            // sincronizam si copiem toate corpurile pentru vizualizare
            cudaMemcpy(bodies.data(), d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);
            auto wall_now = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(wall_now - wall_start).count();
            cout << "STEP " << step << " " << elapsed;
            for (int i = 0; i < N; i++) {
                cout << " " << bodies[i].x << " " << bodies[i].y << " " << bodies[i].z;
            }
            cout << "\n";
            cout.flush();
        } else if (vis_interval == 0 && step % 100 == 0) {
            Body body0;

            // copiem doar primul corp ca sa il afisam
            cudaMemcpy(&body0, d_bodies, sizeof(Body), cudaMemcpyDeviceToHost);

            cout << "Step " << step << " | body[0] position: (";
            cout << body0.x << ", ";
            cout << body0.y << ", ";
            cout << body0.z << ")" << endl;
        }
    }

    cudaEventRecord(cuda_end);
    cudaEventSynchronize(cuda_end);

    float elapsed_ms = 0.0;
    cudaEventElapsedTime(&elapsed_ms, cuda_start, cuda_end);

    // copiere date inapoi din GPU pe CPU
    cudaMemcpy(bodies.data(), d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);

    cout << endl;
    cout << "Done. Time: " << elapsed_ms / 1000.0 << " seconds" << endl;

    // eliberare memorie GPU
    cudaFree(d_bodies);

    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_end);

    return 0;
}