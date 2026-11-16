# matmul-kernel
study repo for my gpu kernels

Usage:
```
make                       # configure + build
make run                   # build + run naive at 1024
make run SIZE=2048 REPS=20
make list
make csv                   # writes results/bench.csv
make csv CSV=results/naive_1024.csv
make clean
```

If Ninja is ever unavailable:
```
make GENERATOR="Unix Makefiles"
```
