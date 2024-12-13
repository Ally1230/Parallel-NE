set -eux

# mpic++ parallel_nash_nplayer.cpp -o parallel_nash_nplayer -I eigen -std=c++14 -fopenmp


# for thread in 8 4 2 1;
# do
#     mpirun -np $thread ./parallel_nash
# done

g++-11 parallel_nash_nplayer.cpp -o parallel_nash_nplayer -I eigen -std=c++14 -fopenmp

for thread in 8 4 2 1;
do
    ./parallel_nash_nplayer -p $thread;
done