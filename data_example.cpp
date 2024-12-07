#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

pair<MatrixXd, MatrixXd> example_1010(){
    const int N = 10;
    MatrixXd payoff1(N, N);
    MatrixXd payoff2(N, N);

    // Populate the payoff matrices
    // Index 4 corresponds to strategy 5 in 1-based indexing.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int dist = std::abs(i - 4) + std::abs(j - 4);
            int val = 10 - dist;
            payoff1(i,j) = val;
            payoff2(i,j) = val;
        }
    }

    std::cout << "Player 1 Payoff Matrix:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << payoff1(i,j) << (j == N-1 ? '\n' : ' ');
        }
    }

    std::cout << "\nPlayer 2 Payoff Matrix:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << payoff2(i,j) << (j == N-1 ? '\n' : ' ');
        }
    }

    // The Nash equilibrium is at (S_5, T_5) which corresponds to payoff1[4][4], payoff2[4][4] = 10.
    return make_pair(payoff1, payoff2);
}

pair<MatrixXd, MatrixXd> example_55(){
    const int N = 5;
    MatrixXd payoff1(N, N);
    MatrixXd payoff2(N, N);

    // Populate the payoff matrices
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int dist = std::abs(i - 2) + std::abs(j - 2); // 2 is index for S_3 or T_3
            int val = 5 - dist;
            payoff1(i,j) = val;
            payoff2(i,j) = val;
        }
    }

    std::cout << "Player 1 Payoff Matrix:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << payoff1(i,j) << (j == N-1 ? '\n' : ' ');
        }
    }

    std::cout << "\nPlayer 2 Payoff Matrix:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << payoff2(i,j) << (j == N-1 ? '\n' : ' ');
        }
    }

    // Nash equilibrium is at (S_3, T_3), i.e., payoff1[2][2], payoff2[2][2].
    // At this point, payoffs are (5,5).
    return make_pair(payoff1, payoff2);
}

pair<MatrixXd, MatrixXd> example_2020(){

    const int N = 20;
    MatrixXd payoff1(N, N);
    MatrixXd payoff2(N, N);

    // Populate the payoff matrices
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            // Compute distance from the "center" strategy (index 9 corresponds to strategy 10)
            int dist = std::abs(i - 9) + std::abs(j - 9);
            int val = 10 - dist;
            payoff1(i,j) = val;
            payoff2(i,j) = val;
        }
    }

    // Print Player 1's payoff matrix
    std::cout << "Player 1 Payoff Matrix:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << payoff1(i,j) << (j == N-1 ? '\n' : ' ');
        }
    }

    std::cout << "\nPlayer 2 Payoff Matrix:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << payoff2(i,j) << (j == N-1 ? '\n' : ' ');
        }
    }

    return make_pair(payoff1, payoff2);
}
