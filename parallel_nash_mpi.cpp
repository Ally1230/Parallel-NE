#include <iostream>
#include <vector>
#include <set>
#include <Eigen/Dense>
#include <cmath>
#include "data_example.cpp"
#include <chrono>
#include <unistd.h>
#include <mpi.h> // Include MPI header

using namespace std;
using namespace Eigen;

static double eps = 1e-9;

bool isValidProbabilityVector(const VectorXd &v)
{
    if (fabs(v.sum() - 1.0) > eps)
        return false;
    for (int i = 0; i < v.size(); ++i)
    {
        if (v(i) < -eps)
            return false;
    }
    return true;
}

// Generate all non-empty subsets of a set
vector<vector<int>> generateSubsets(const vector<int> &set)
{
    int n = (int)set.size();
    int subset_count = 1 << n;

    vector<vector<int>> subsets(subset_count);
    for (int i = 1; i < subset_count; ++i)
    {
        vector<int> subset;
        for (int j = 0; j < n; ++j)
        {
            if (i & (1 << j))
                subset.push_back(set[j]);
        }
        subsets[i] = subset;
    }
    return subsets;
}

vector<pair<vector<int>, vector<int>>> generateSubsetsDouble(const vector<int> &setA, const vector<int> &setB)
{
    int n = (int)setA.size();
    int m = (int)setB.size();
    vector<pair<vector<int>, vector<int>>> subset_pairs;

    // First generate all subsets of setA
    int subset_countA = 1 << n;
    vector<vector<int>> subsets(subset_countA);
    for (int i = 1; i < subset_countA; ++i)
    {
        vector<int> subset;
        for (int j = 0; j < n; ++j)
        {
            if (i & (1 << j))
                subset.push_back(setA[j]);
        }
        subsets[i] = subset;
    }

    // Create pairs of subsets of setA and setB
    int subset_countB = 1 << m;
    for (int i = 0; i < subset_countB; ++i)
    {
        vector<int> subset;
        for (int j = 0; j < m; ++j)
        {
            if (i & (1 << j))
                subset.push_back(setB[j]);
        }

        for (int k = 0; k < subset_countA; ++k)
        {
            subset_pairs.emplace_back(make_pair(subsets[k], subset));
        }
    }

    return subset_pairs;
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Synchronize before starting the timer
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::steady_clock::now();

    // Read data
    int n = 2;
    int m = 18;
    MatrixXd A(m, n), B(m, n);
    // Assuming example_random generates two matrices A and B
    tie(A, B) = example_random(m, n);

    vector<int> strategies_p1(m), strategies_p2(n);
    for (int i = 0; i < m; ++i)
        strategies_p1[i] = i;
    for (int j = 0; j < n; ++j)
        strategies_p2[j] = j;

    // Master generates all support_pairs
    vector<pair<vector<int>, vector<int>>> support_pairs;
    if (rank == 0)
    {
        support_pairs = generateSubsetsDouble(strategies_p1, strategies_p2);
        cout << "Master generated support_pairs of size: " << support_pairs.size() << "\n";
    }

    // Broadcast the size of support_pairs
    int support_size = 0;
    if (rank == 0)
        support_size = support_pairs.size();
    MPI_Bcast(&support_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast the support_pairs data
    if (rank != 0)
        support_pairs.resize(support_size);

    // To broadcast the data, we need to serialize it
    // We'll first broadcast the size of each pair of subsets, then the elements
    // First, prepare counts
    vector<int> subset_sizes_a(support_size);
    vector<int> subset_sizes_b(support_size);
    vector<int> flat_subset_a;
    vector<int> flat_subset_b;

    if (rank == 0)
    {
        for (int i = 0; i < support_size; ++i)
        {
            subset_sizes_a[i] = support_pairs[i].first.size();
            subset_sizes_b[i] = support_pairs[i].second.size();
            flat_subset_a.insert(flat_subset_a.end(), support_pairs[i].first.begin(), support_pairs[i].first.end());
            flat_subset_b.insert(flat_subset_b.end(), support_pairs[i].second.begin(), support_pairs[i].second.end());
        }
    }

    // Broadcast subset_sizes_a and subset_sizes_b
    MPI_Bcast(subset_sizes_a.data(), support_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(subset_sizes_b.data(), support_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate total elements to receive
    int total_a = 0, total_b = 0;
    for (int i = 0; i < support_size; ++i)
    {
        total_a += subset_sizes_a[i];
        total_b += subset_sizes_b[i];
    }

    if (rank != 0)
    {
        flat_subset_a.resize(total_a);
        flat_subset_b.resize(total_b);
    }

    // Broadcast flat_subset_a and flat_subset_b
    MPI_Bcast(flat_subset_a.data(), total_a, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(flat_subset_b.data(), total_b, MPI_INT, 0, MPI_COMM_WORLD);

    // Reconstruct support_pairs
    if (rank != 0)
    {
        int index_a = 0, index_b = 0;
        for (int i = 0; i < support_size; ++i)
        {
            int size_a = subset_sizes_a[i];
            int size_b = subset_sizes_b[i];
            support_pairs[i].first.assign(flat_subset_a.begin() + index_a, flat_subset_a.begin() + index_a + size_a);
            support_pairs[i].second.assign(flat_subset_b.begin() + index_b, flat_subset_b.begin() + index_b + size_b);
            index_a += size_a;
            index_b += size_b;
        }
    }

    // Now, distribute the work among processes
    // Each process will handle support_pairs starting from its rank and stepping by size
    vector<pair<vector<int>, vector<int>>> local_support_pairs;
    for (int i = rank; i < support_size; i += size)
    {
        local_support_pairs.emplace_back(support_pairs[i]);
    }

    if (rank == 0)
    {
        cout << "Distributing " << support_size << " support_pairs among " << size << " processes.\n";
    }

    // Initialize local equilibria
    vector<pair<vector<double>, vector<double>>> local_equilibria;

    // Start processing local_support_pairs
    for (const auto &pair_support : local_support_pairs)
    {
        const vector<int> &support_p1 = pair_support.first;
        const vector<int> &support_p2 = pair_support.second;

        int k = (int)support_p1.size();
        int l = (int)support_p2.size();
        if (k == 0 || l == 0)
            continue;

        // Build submatrices
        MatrixXd A_p1(k, l), B_p2(k, l);
        for (int i = 0; i < k; ++i)
        {
            for (int j = 0; j < l; ++j)
            {
                A_p1(i, j) = A(support_p1[i], support_p2[j]);
                B_p2(i, j) = B(support_p1[i], support_p2[j]);
            }
        }

        // Solve for y, u1:
        // [ A_p1      | -1_k ] [y ]   [0]
        // [ ones(l)^T |  0   ] [u1] = [1]
        {
            MatrixXd C1(k + 1, l + 1);
            VectorXd d1(k + 1);
            // Fill top-left with A_p1, top-right with -ones(k)
            C1.block(0, 0, k, l) = A_p1;
            C1.block(0, l, k, 1) = -VectorXd::Ones(k);
            // Bottom row: ones(l)^T and 0
            C1.block(k, 0, 1, l) = VectorXd::Ones(l).transpose();
            C1(k, l) = 0.0;

            d1.head(k) = VectorXd::Zero(k);
            d1(k) = 1.0;

            FullPivLU<MatrixXd> lu1(C1);
            if (!lu1.isInvertible())
                continue;
            VectorXd sol1 = lu1.solve(d1);

            VectorXd y = sol1.head(l);
            double u1 = sol1(l);

            if (!isValidProbabilityVector(y))
                continue;

            // Solve for x, u2:
            // [ B_p2^T    | -1_l ] [x ]   [0]
            // [ ones(k)^T |  0   ] [u2] = [1]
            MatrixXd C2(l + 1, k + 1);
            VectorXd d2(l + 1);
            C2.block(0, 0, l, k) = B_p2.transpose();
            C2.block(0, k, l, 1) = -VectorXd::Ones(l);
            C2.block(l, 0, 1, k) = VectorXd::Ones(k).transpose();
            C2(l, k) = 0.0;

            d2.head(l) = VectorXd::Zero(l);
            d2(l) = 1.0;

            FullPivLU<MatrixXd> lu2(C2);
            if (!lu2.isInvertible())
                continue;
            VectorXd sol2 = lu2.solve(d2);

            VectorXd x = sol2.head(k);
            double u2 = sol2(k);

            if (!isValidProbabilityVector(x))
                continue;

            // Construct full distributions
            vector<double> x_full(m, 0.0), y_full(n, 0.0);
            for (int i = 0; i < k; ++i)
                x_full[support_p1[i]] = x(i);
            for (int j = 0; j < l; ++j)
                y_full[support_p2[j]] = y(j);

            // Compute actual payoffs
            VectorXd Xfull = Map<VectorXd>(x_full.data(), m);
            VectorXd Yfull = Map<VectorXd>(y_full.data(), n);
            double u1_actual = (Xfull.transpose() * A * Yfull)(0, 0);
            double u2_actual = (Xfull.transpose() * B * Yfull)(0, 0);

            // Check no outside strategy for P1 is better
            {
                set<int> s1(support_p1.begin(), support_p1.end());
                bool valid = true;
                for (int i = 0; i < m && valid; ++i)
                {
                    if (s1.find(i) == s1.end())
                    {
                        double payoff_i = 0.0;
                        for (int j = 0; j < n; ++j)
                        {
                            payoff_i += A(i, j) * y_full[j];
                        }
                        if (payoff_i > u1_actual + eps)
                            valid = false;
                    }
                }

                // Check no outside strategy for P2 is better
                if (valid)
                {
                    set<int> s2(support_p2.begin(), support_p2.end());
                    for (int j = 0; j < n && valid; ++j)
                    {
                        if (s2.find(j) == s2.end())
                        {
                            double payoff_j = 0.0;
                            for (int i = 0; i < m; ++i)
                            {
                                payoff_j += x_full[i] * B(i, j);
                            }
                            if (payoff_j > u2_actual + eps)
                                valid = false;
                        }
                    }
                    if (valid)
                    {
                        local_equilibria.emplace_back(make_pair(x_full, y_full));
                    }
                }
            }
        }
    }

    // Serialize local_equilibria to send to master
    // Each equilibrium consists of m + n doubles
    int local_eq_count = local_equilibria.size();
    vector<double> flat_local_eq;
    flat_local_eq.reserve(local_eq_count * (m + n));
    for (const auto &eq : local_equilibria)
    {
        flat_local_eq.insert(flat_local_eq.end(), eq.first.begin(), eq.first.end());
        flat_local_eq.insert(flat_local_eq.end(), eq.second.begin(), eq.second.end());
    }

    // Gather the counts from all processes
    vector<int> all_eq_counts(size, 0);
    MPI_Gather(&local_eq_count, 1, MPI_INT, all_eq_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Gather the equilibria data using MPI_Gatherv
    // First, gather the sizes of the data to receive
    int local_data_size = flat_local_eq.size();
    vector<int> all_data_sizes(size, 0);
    MPI_Gather(&local_data_size, 1, MPI_INT, all_data_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Prepare displacements and receive buffer on master
    vector<int> displs(size, 0);
    int total_data = 0;
    if (rank == 0)
    {
        for (int i = 0; i < size; ++i)
        {
            displs[i] = total_data;
            total_data += all_data_sizes[i];
        }
    }

    vector<double> all_flat_eq;
    if (rank == 0)
    {
        all_flat_eq.resize(total_data);
    }

    MPI_Gatherv(flat_local_eq.data(), local_data_size, MPI_DOUBLE,
                all_flat_eq.data(), all_data_sizes.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Master reconstructs the equilibria
    set<pair<vector<double>, vector<double>>> equilibria;
    if (rank == 0)
    {
        // Iterate over all processes
        for (int proc = 0; proc < size; ++proc)
        {
            int eq_count_proc = all_eq_counts[proc];
            int data_size_proc = all_data_sizes[proc];
            for (int i = 0; i < eq_count_proc; ++i)
            {
                int offset = displs[proc] + i * (m + n);
                vector<double> x_full(m);
                vector<double> y_full(n);
                for (int xi = 0; xi < m; ++xi)
                {
                    x_full[xi] = all_flat_eq[offset + xi];
                }
                for (int yi = 0; yi < n; ++yi)
                {
                    y_full[yi] = all_flat_eq[offset + m + yi];
                }
                equilibria.emplace(make_pair(x_full, y_full));
            }
        }

        // End timer after gathering results
        auto end = std::chrono::steady_clock::now();

        // // Print result
        // cout << "Nash Equilibria found:\n";
        // int eq_count = 0;
        // for (const auto &eq : equilibria)
        // {
        //     cout << "Equilibrium " << ++eq_count << ":\n";
        //     cout << "Player 1 strategy: [ ";
        //     for (double prob : eq.first)
        //         cout << prob << " ";
        //     cout << "]\n";
        //     cout << "Player 2 strategy: [ ";
        //     for (double prob : eq.second)
        //         cout << prob << " ";
        //     cout << "]\n";
        // }

        // if (eq_count == 0)
            // cout << "No Nash Equilibria found.\n";

        // Calculate execution time in seconds
        const double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        std::cout << "Execution Time: " << duration << " seconds" << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}