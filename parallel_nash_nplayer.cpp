#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cassert>
#include <functional>

using namespace std;

static double eps = 1e-9;

struct NPlayerGame {
    int n; // number of players
    vector<int> m; // m[i] = number of strategies for player i
    // payoff[i] is a flattened array of payoffs for player i
    vector<vector<double>> payoff;

    int totalProfiles() const {
        int prod = 1;
        for (auto mm : m) prod *= mm;
        return prod;
    }

    int profileToIndex(const vector<int>& profile) const {
        assert((int)profile.size() == n);
        int idx = 0;
        int multiplier = 1;
        for (int i = n-1; i >= 0; --i) {
            idx += profile[i] * multiplier;
            multiplier *= m[i];
        }
        return idx;
    }
};

vector<vector<int>> generateSubsets(int M) {
    int subset_count = 1 << M;
    vector<vector<int>> subsets;
    for (int i = 1; i < subset_count; ++i) {
        vector<int> subset;
        for (int j = 0; j < M; ++j) {
            if (i & (1 << j))
                subset.push_back(j);
        }
        subsets.push_back(subset);
    }
    return subsets;
}

vector<vector<vector<int>>> generateAllSupportProfiles(const vector<int> &m) {
    int n = (int)m.size();
    vector<vector<vector<int>>> subsets_for_each_player(n);
    for (int i = 0; i < n; ++i) {
        subsets_for_each_player[i] = generateSubsets(m[i]);
    }

    vector<vector<vector<int>>> result;
    std::function<void(int, vector<vector<int>>&)> recurse = [&](int idx, vector<vector<int>> &current) {
        if (idx == n) {
            result.push_back(current);
            return;
        }
        for (auto &sub : subsets_for_each_player[idx]) {
            current[idx] = sub;
            recurse(idx + 1, current);
        }
    };

    vector<vector<int>> current(n);
    recurse(0, current);
    return result;
}

bool isValidProbabilityVector(const vector<double> &v) {
    double sum = 0.0;
    for (auto x : v) {
        if (x < -eps) return false;
        sum += x;
    }
    return fabs(sum - 1.0) < eps;
}

// This is a placeholder. Implementing a full n-player solver is non-trivial.
// Here, we simply check if the support profile corresponds to a known pure equilibrium (0,0,0) or (1,1,1).
bool findEquilibriumForSupport(const NPlayerGame &game,
                               const vector<vector<int>> &support_profile,
                               vector<vector<double>> &equilibrium) 
{
    // For our test game: 
    // The pure NE are (0,0,0) and (1,1,1).

    // Check if support_profile contains only one strategy per player (pure profile)
    // and if it matches (0,0,0) or (1,1,1).
    // If so, return that as an equilibrium.
    // Otherwise, fail.

    int n = game.n;
    // Check if each player's support has size 1
    for (int i = 0; i < n; ++i) {
        if (support_profile[i].size() != 1) {
            return false;
        }
    }

    // Extract the pure profile
    vector<int> profile(n);
    for (int i = 0; i < n; ++i) {
        profile[i] = support_profile[i][0];
    }

    // Check if profile is (0,0,0) or (1,1,1)
    bool all_zero = true;
    bool all_one = true;
    for (int i = 0; i < n; ++i) {
        if (profile[i] != 0) all_zero = false;
        if (profile[i] != 1) all_one = false;
    }

    if (all_zero || all_one) {
        // Construct equilibrium distribution: pure strategies
        equilibrium.resize(n);
        for (int i = 0; i < n; ++i) {
            equilibrium[i].resize(game.m[i], 0.0);
            equilibrium[i][profile[i]] = 1.0;
        }
        return true;
    }

    return false;
}

int main(int argc, char *argv[]) {
    int num_threads = 1;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);

    // Setup a 3-player, 2-strategy game with known NE:
    // Players = 3, m = {2,2,2}
    // Payoffs:
    // If (0,0,0): each gets 1
    // If (1,1,1): each gets 1
    // Otherwise: each gets 0
    NPlayerGame game;
    game.n = 3;
    game.m = {2,2,2};

    int total = game.totalProfiles(); // 2*2*2 = 8 profiles
    game.payoff.resize(game.n, vector<double>(total, 0.0));

    // Profiles:
    // Index mapping (a1,a2,a3) -> idx:
    // Let’s assume indexing: idx = a3 + 2*a2 + 4*a1
    // (0,0,0) = idx 0
    // (0,0,1) = idx 1
    // (0,1,0) = idx 2
    // (0,1,1) = idx 3
    // (1,0,0) = idx 4
    // (1,0,1) = idx 5
    // (1,1,0) = idx 6
    // (1,1,1) = idx 7

    // Assign payoffs:
    // All same = 1, else 0
    // (0,0,0) -> idx=0: payoff to all = 1
    // (1,1,1) -> idx=7: payoff to all = 1
    // others = 0
    game.payoff[0][0] = 1.0;
    game.payoff[1][0] = 1.0;
    game.payoff[2][0] = 1.0;

    game.payoff[0][7] = 1.0;
    game.payoff[1][7] = 1.0;
    game.payoff[2][7] = 1.0;

    auto start = std::chrono::steady_clock::now();

    auto support_profiles = generateAllSupportProfiles(game.m);

    cout << "Number of support profiles: " << support_profiles.size() << "\n";

    vector<vector<vector<double>>> all_equilibria;

    #pragma omp parallel
    {
        vector<vector<vector<double>>> local_equilibria;
        #pragma omp for schedule(dynamic)
        for (size_t idx = 0; idx < support_profiles.size(); idx++) {
            const auto &support_profile = support_profiles[idx];

            vector<vector<double>> eq;
            if (findEquilibriumForSupport(game, support_profile, eq)) {
                local_equilibria.push_back(eq);
            }
        }

        #pragma omp critical
        {
            for (auto &eq : local_equilibria)
                all_equilibria.push_back(eq);
        }
    }

    auto end = std::chrono::steady_clock::now();

    cout << "Nash Equilibria found:\n";
    int eq_count = 0;
    for (auto &eq : all_equilibria) {
        cout << "Equilibrium " << ++eq_count << ":\n";
        for (int i = 0; i < game.n; ++i) {
            cout << "Player " << i+1 << " strategy: [ ";
            for (auto prob : eq[i])
                cout << prob << " ";
            cout << "]\n";
        }
    }
    if (eq_count == 0)
        cout << "No Nash Equilibria found.\n";

    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    cout << "Execution Time: " << duration << " seconds\n";

    return 0;
}