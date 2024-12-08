#include <iostream>
#include <vector>
#include <set>
#include <Eigen/Dense>
#include <cmath>
#include "data_example.cpp"
#include <chrono>

using namespace std;
using namespace Eigen;

static double eps = 1e-9;

bool isValidProbabilityVector(const VectorXd &v) {
    if (fabs(v.sum() - 1.0) > eps) return false;
    for (int i = 0; i < v.size(); ++i) {
        if (v(i) < -eps) return false;
    }
    return true;
}

// Generate all non-empty subsets of a set
void generateSubsets(const vector<int> &set, vector<vector<int>> &subsets) {
    int n = (int)set.size();
    int subset_count = 1 << n;
    for (int i = 1; i < subset_count; ++i) {
        vector<int> subset;
        for (int j = 0; j < n; ++j) {
            if (i & (1 << j))
                subset.push_back(set[j]);
        }
        subsets.push_back(subset);
    }
}

int main() {
    // Read data
    int n = 10;
    int m = 10;
    auto [A, B] = example_random(m, n);

    const auto start = std::chrono::steady_clock::now();

    vector<int> strategies_p1(m), strategies_p2(n);
    for (int i = 0; i < m; ++i)
        strategies_p1[i] = i;
    for (int j = 0; j < n; ++j)
        strategies_p2[j] = j;

    // generate all non-empty subsets of strategies
    vector<vector<int>> supports_p1, supports_p2;
    generateSubsets(strategies_p1, supports_p1);
    generateSubsets(strategies_p2, supports_p2);

    set<pair<vector<double>, vector<double>>> equilibria;

    for (const auto &support_p1 : supports_p1) {
        for (const auto &support_p2 : supports_p2) {
            int k = (int)support_p1.size();
            int l = (int)support_p2.size();
            if (k == 0 || l == 0) continue;

            // Build submatrices
            MatrixXd A_p1(k, l), B_p2(k, l);
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < l; ++j) {
                    A_p1(i, j) = A(support_p1[i], support_p2[j]);
                    B_p2(i, j) = B(support_p1[i], support_p2[j]);
                }
            }

            // Solve for y, u1:
            // [ A_p1      | -1_k ] [y ]   [0]
            // [ ones(l)^T |  0   ] [u1] = [1]
            {
                MatrixXd C1(k+1, l+1);
                VectorXd d1(k+1);
                // Fill top-left with A_p1, top-right with -ones(k)
                C1.block(0,0,k,l) = A_p1;
                C1.block(0,l,k,1) = -VectorXd::Ones(k);
                // Bottom row: ones(l)^T and 0
                C1.block(k,0,1,l) = VectorXd::Ones(l).transpose();
                C1(k, l) = 0.0;

                d1.head(k) = VectorXd::Zero(k);
                d1(k) = 1.0;

                FullPivLU<MatrixXd> lu1(C1);
                if (!lu1.isInvertible()) continue;
                VectorXd sol1 = lu1.solve(d1);

                VectorXd y = sol1.head(l);
                double u1 = sol1(l);

                if (!isValidProbabilityVector(y)) continue;

                // Solve for x, u2:
                // [ B_p2^T    | -1_l ] [x ]   [0]
                // [ ones(k)^T |  0   ] [u2] = [1]
                MatrixXd C2(l+1, k+1);
                VectorXd d2(l+1);
                C2.block(0,0,l,k) = B_p2.transpose();
                C2.block(0,k,l,1) = -VectorXd::Ones(l);
                C2.block(l,0,1,k) = VectorXd::Ones(k).transpose();
                C2(l, k) = 0.0;

                d2.head(l) = VectorXd::Zero(l);
                d2(l) = 1.0;

                FullPivLU<MatrixXd> lu2(C2);
                if (!lu2.isInvertible()) continue;
                VectorXd sol2 = lu2.solve(d2);

                VectorXd x = sol2.head(k);
                double u2 = sol2(k);

                if (!isValidProbabilityVector(x)) continue;

                // Construct full distributions
                vector<double> x_full(m, 0.0), y_full(n, 0.0);
                for (int i = 0; i < k; ++i) x_full[support_p1[i]] = x(i);
                for (int j = 0; j < l; ++j) y_full[support_p2[j]] = y(j);

                // Compute actual payoffs
                VectorXd Xfull = Map<VectorXd>(x_full.data(), m);
                VectorXd Yfull = Map<VectorXd>(y_full.data(), n);
                double u1_actual = (Xfull.transpose() * A * Yfull)(0,0);
                double u2_actual = (Xfull.transpose() * B * Yfull)(0,0);

                // Check no outside strategy for P1 is better
                {
                    set<int> s1(support_p1.begin(), support_p1.end());
                    bool valid = true;
                    for (int i = 0; i < m && valid; ++i) {
                        if (s1.find(i) == s1.end()) {
                            double payoff_i = 0.0;
                            for (int j = 0; j < n; ++j) {
                                payoff_i += A(i,j)*y_full[j];
                            }
                            if (payoff_i > u1_actual + eps) valid = false;
                        }
                    }

                    // Check no outside strategy for P2 is better
                    if (valid) {
                        set<int> s2(support_p2.begin(), support_p2.end());
                        for (int j = 0; j < n && valid; ++j) {
                            if (s2.find(j) == s2.end()) {
                                double payoff_j = 0.0;
                                for (int i = 0; i < m; ++i) {
                                    payoff_j += x_full[i]*B(i,j);
                                }
                                if (payoff_j > u2_actual + eps) valid = false;
                            }
                        }
                        if (valid) {
                            equilibria.insert({x_full, y_full});
                        }
                    }
                }
            }
        }
    }
    // End timer
    const auto end = std::chrono::steady_clock::now();

    cout << "\nNash Equilibria found:\n";
    int eq_count = 0;
    for (const auto &eq : equilibria) {
        cout << "\nEquilibrium " << ++eq_count << ":\n";
        cout << "Player 1 strategy: [ ";
        for (double prob : eq.first) cout << prob << " ";
        cout << "]\n";
        cout << "Player 2 strategy: [ ";
        for (double prob : eq.second) cout << prob << " ";
        cout << "]\n";
    }

    if (eq_count == 0)
        cout << "No Nash Equilibria found.\n";


    // Calculate execution time in seconds
    const double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Execution Time: " << duration << " seconds" << std::endl;

    return 0;
}
