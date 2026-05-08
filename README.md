# Simulare N-Corpuri (N-Body Simulation)

Acest depozit contine o implementare in C++ a unei simulari gravitationale de tip N-corpuri (N-body). Proiectul a fost dezvoltat pentru a explora si compara tehnicile de calcul de inalta performanta (HPC), oferind trei variante distincte de calcul.

## Versiuni incluse

Proiectul contine trei fisiere sursa, fiecare demonstrand o abordare diferita:

1. **Implementare Secventiala (CPU):** O abordare clasica, single-thread, care ruleaza pe un singur nucleu al procesorului. Ideala pentru a intelege algoritmul de baza.
2. **Implementare OpenMP (CPU Multi-threading):** O versiune paralelizata care utilizeaza directiva `#pragma omp` pentru a distribui calculul fortelor si actualizarea pozitiilor pe mai multe nuclee ale procesorului (configurat la 16 thread-uri).
3. **Implementare CUDA (GPU):** Cea mai performanta versiune, care muta calculele masiv paralelizabile pe placa video. Foloseste kernel-uri custom (`computeForcesKernel`, `moveBodiesKernel`) pentru a accelera exponential simularea.

## Detalii Tehnice

- **Numar de corpuri (N):** 10.000
- **Numar de pasi (STEPS):** 1.000
- **Fizica:** Foloseste legea atractiei universale ($G = 6.674 \times 10^{-11}$) si include un factor de `SOFTENING` ($10^{-9}$) pentru a evita diviziunea la zero (sau forte infinite) la coliziuni.
- Integrarea miscarii se face folosind o metoda numerica simpla pe baza unui pas de timp (`DT = 0.01`).

## Cum se compileaza si ruleaza
## Fisier: `Proiect PP Secvential.cpp`

### Compilare:

```bash
g++ "Proiect PP Secvential.cpp" -o nbody
```
### Rulare:

```bash
./nbody
```

## Fisier: `Proiect PP Paralel OMP.cpp`

### Compilare:

```bash
g++ "Proiect PP Paralel OMP.cpp" -fopenmp -o nbody-omp
```

### Rulare:

```bash
./nbody-omp
```

## Fisier: `Proiect PP Paralel CUDA.cu`

### Compilare:

```bash
nvcc "Proiect PP Paralel CUDA.cu" -o nbody-cuda
```
### Rulare:

```bash
./nbody-cuda
```