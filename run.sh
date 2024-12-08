set -eux

g++-11 parallel_nash.cpp -o parallel_nash -I eigen -std=c++14 -fopenmp

for thread in 1 2 4 8;
do
    ./parallel_nash -p $thread;
done