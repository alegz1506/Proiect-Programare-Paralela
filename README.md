# Proiect Programare Paralela

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