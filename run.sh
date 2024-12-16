# set -eux

# mpic++ parallel_nash_mpi.cpp -o parallel_nash -I eigen -std=c++14 -fopenmp
# for thread in 128 64 32 16 8 4 2 1;
# do
#     mpirun -np $thread ./parallel_nash
# done

g++ parallel_nash.cpp -o parallel_nash -I eigen -std=c++14 -fopenmp
for thread in 128 64 32 16 8 4 2 1;
do
    ./parallel_nash -p $thread;
done
