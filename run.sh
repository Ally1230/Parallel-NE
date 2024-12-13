set -eux

# g++-11 parallel_nash.cpp -o parallel_nash -I eigen -std=c++14 -fopenmp

# for thread in 8 4 2 1;
# do
#     ./parallel_nash -p $thread;
# done

mpic++ parallel_nash_mpi.cpp -o parallel_nash -I eigen -std=c++14 -fopenmp

for thread in 8 4 2 1;
do
    mpirun -np $thread ./parallel_nash
done
