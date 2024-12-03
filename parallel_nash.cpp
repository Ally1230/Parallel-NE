#include <iostream>
#include <vector>
#include <set>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Function to check if a vector has all elements non-negative and sums to 1
bool isValidProbabilityVector(const VectorXd &v) {
    const double epsilon = 1e-6;
    for(int i = 0; i < v.size(); ++i) {
        if(v(i) < -epsilon)
            return false;
    }
    return fabs(v.sum() - 1.0) < epsilon;
}

// Function to generate all subsets of a given set
void generateSubsets(const vector<int> &set, vector<vector<int>> &subsets) {
    int n = set.size();
    int subset_count = 1 << n;
    for(int i = 1; i < subset_count; ++i) { // Skip the empty set
        vector<int> subset;
        for(int j = 0; j < n; ++j) {
            if(i & (1 << j))
                subset.push_back(set[j]);
        }
        subsets.push_back(subset);
    }
}

int main() {
    int m, n;
    cout << "Enter the number of strategies for Player 1: ";
    cin >> m;
    cout << "Enter the number of strategies for Player 2: ";
    cin >> n;

    cout << "Enter the payoff matrix for Player 1 (" << m << "x" << n << "):\n";
    MatrixXd A(m, n);
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            cin >> A(i, j);

    cout << "Enter the payoff matrix for Player 2 (" << m << "x" << n << "):\n";
    MatrixXd B(m, n);
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
            cin >> B(i, j);

    vector<int> strategies_p1(m), strategies_p2(n);
    for(int i = 0; i < m; ++i) strategies_p1[i] = i;
    for(int j = 0; j < n; ++j) strategies_p2[j] = j;

    vector<vector<int>> supports_p1, supports_p2;
    generateSubsets(strategies_p1, supports_p1);
    generateSubsets(strategies_p2, supports_p2);

    set<pair<vector<double>, vector<double>>> equilibria;

    // Enumerate all possible supports
    for(const auto &support_p1 : supports_p1) {
        for(const auto &support_p2 : supports_p2) {
            int k = support_p1.size();
            int l = support_p2.size();

            // Skip if sizes are incompatible
            if(k == 0 || l == 0) continue;

            // Set up the payoff submatrices for the supports
            MatrixXd A_p1(k, l);
            MatrixXd B_p2(k, l);
            for(int i = 0; i < k; ++i)
                for(int j = 0; j < l; ++j) {
                    A_p1(i, j) = A(support_p1[i], support_p2[j]);
                    B_p2(i, j) = B(support_p1[i], support_p2[j]);
                }

            // Variables for Player 1's strategy and payoff
            VectorXd x(k);
            double u1;

            // Solve for Player 1's mixed strategy
            {
                MatrixXd C(l + 1, k + 1);
                C << A_p1.transpose(), -VectorXd::Ones(l),
                     VectorXd::Ones(k).transpose(), 0;
                VectorXd d = VectorXd::Zero(l + 1);
                d(l) = 1.0;
                // Check if the system is solvable
                FullPivLU<MatrixXd> lu(C);
                if(!lu.isInvertible()) continue;
                VectorXd sol = lu.solve(d);
                x = sol.head(k);
                u1 = sol(k);
            }

            // Variables for Player 2's strategy and payoff
            VectorXd y(l);
            double u2;

            // Solve for Player 2's mixed strategy
            {
                MatrixXd C(k + 1, l + 1);
                C << B_p2, -VectorXd::Ones(k),
                     VectorXd::Ones(l).transpose(), 0;
                VectorXd d = VectorXd::Zero(k + 1);
                d(k) = 1.0;
                // Check if the system is solvable
                FullPivLU<MatrixXd> lu(C);
                if(!lu.isInvertible()) continue;
                VectorXd sol = lu.solve(d);
                y = sol.head(l);
                u2 = sol(l);
            }

            // Check if strategies are valid probability distributions
            if(!isValidProbabilityVector(x) || !isValidProbabilityVector(y))
                continue;

            // Construct full mixed strategies
            vector<double> x_full(m, 0.0), y_full(n, 0.0);
            for(int i = 0; i < k; ++i)
                x_full[support_p1[i]] = x(i);
            for(int j = 0; j < l; ++j)
                y_full[support_p2[j]] = y(j);

            equilibria.insert({x_full, y_full});
        }
    }

    cout << "\nNash Equilibria found:\n";
    int eq_count = 0;
    for(const auto &eq : equilibria) {
        cout << "\nEquilibrium " << ++eq_count << ":\n";
        cout << "Player 1 strategy: [ ";
        for(double prob : eq.first)
            cout << prob << " ";
        cout << "]\n";
        cout << "Player 2 strategy: [ ";
        for(double prob : eq.second)
            cout << prob << " ";
        cout << "]\n";
    }

    if(eq_count == 0)
        cout << "No Nash Equilibria found.\n";

    return 0;
}
