# Algorithm: D-SVARM (Data Shapley via Stratified SVARM)

## Corrected Pseudocode

```
Algorithm: D-SVARM — Data Shapley Value Approximation without Requesting Marginals

Require:
  Training data D = {z_1, ..., z_n}
  Validation set D_val
  Budget T (total number of value function evaluations)

Ensure:
  Data Shapley estimates φ̂_1, ..., φ̂_n

──────────────────────────────────────────────
Phase 1: INITIALIZATION
──────────────────────────────────────────────

1.  Initialize φ̂⁺[i, ℓ] ← 0,  c⁺[i, ℓ] ← 0   for all i ∈ [n], ℓ ∈ {0, ..., n-1}
2.  Initialize φ̂⁻[i, ℓ] ← 0,  c⁻[i, ℓ] ← 0   for all i ∈ [n], ℓ ∈ {0, ..., n-1}

──────────────────────────────────────────────
Phase 2: WARMUP  (ensures every stratum has ≥1 sample)
──────────────────────────────────────────────

3.  for ℓ = 0, 1, ..., n-1 do
4.      A ← RandomSubset([n], ℓ + 1)        // sample a random subset of size ℓ+1
5.      v ← Score(Train({z_j : j ∈ A}), D_val)  // ONE value function evaluation
6.      for each i ∈ A do                    // i is IN the coalition
7.          φ̂⁺[i, ℓ] ← v                   // this is a sample for φ_{i,ℓ}^+
8.          c⁺[i, ℓ] ← 1
9.      end for
10.     for each i ∉ A do                    // i is NOT in the coalition
11.         φ̂⁻[i, ℓ+1] ← v                 // |A| = ℓ+1 and i∉A, so stratum is ℓ+1
12.         c⁻[i, ℓ+1] ← 1                 // (only when ℓ+1 ≤ n-1)
13.     end for
14. end for
15. t ← n                                   // warmup used n evaluations

──────────────────────────────────────────────
Phase 3: MAIN LOOP  (maximum sample reuse)
──────────────────────────────────────────────

16. while t < T do
17.     ℓ ← Uniform{0, 1, ..., n-1}         // sample a stratum index
18.     A ← RandomSubset([n], ℓ + 1)        // sample a coalition of size ℓ+1
19.     v ← Score(Train({z_j : j ∈ A}), D_val)  // ONE evaluation — THE key insight
20.
21.     // Update POSITIVE estimates for all players INSIDE A
22.     for each i ∈ A do
23.         φ̂⁺[i, ℓ] ← (c⁺[i, ℓ] · φ̂⁺[i, ℓ] + v) / (c⁺[i, ℓ] + 1)   // running mean
24.         c⁺[i, ℓ] ← c⁺[i, ℓ] + 1
25.     end for
26.
27.     // Update NEGATIVE estimates for all players OUTSIDE A
28.     if ℓ + 1 ≤ n - 1 then
29.         for each i ∉ A do
30.             φ̂⁻[i, ℓ+1] ← (c⁻[i, ℓ+1] · φ̂⁻[i, ℓ+1] + v) / (c⁻[i, ℓ+1] + 1)
31.             c⁻[i, ℓ+1] ← c⁻[i, ℓ+1] + 1
32.         end for
33.     end if
34.
35.     t ← t + 1
36. end while

──────────────────────────────────────────────
Phase 4: AGGREGATION
──────────────────────────────────────────────

37. for each i ∈ [n] do
38.     φ̂_i ← (1/n) · Σ_{ℓ=0}^{n-1} (φ̂⁺[i, ℓ] - φ̂⁻[i, ℓ])
39. end for
40. return φ̂_1, ..., φ̂_n
```


## Why This Works — Key Design Principles

### Principle 1: No Marginal Contributions
Line 19 computes V(A) alone — NOT V(A∪{i}) - V(A).
The Shapley decomposition φ_i = φ_i^+ - φ_i^- handles the "difference" 
at aggregation time (line 38), not at sampling time.

### Principle 2: Maximum Sample Reuse
Each single evaluation V(A) updates:
  - |A| = ℓ+1 players' POSITIVE estimates (those in A)
  - n - ℓ - 1 players' NEGATIVE estimates (those not in A)
  Total: ALL n players updated per evaluation.

### Principle 3: Stratification Reduces Variance
By partitioning coalitions by size ℓ, estimates within each stratum 
are more homogeneous (coalitions of similar size tend to produce 
similar performance). This yields variance bound:
  V[φ̂_i] = O(log n / (T - n·log n))
which beats the O(n/T) of traditional Monte Carlo Shapley.

### Principle 4: Unbiasedness
Because A is drawn uniformly from all subsets of size ℓ+1, and ℓ is 
uniform over {0,...,n-1}, the conditional distribution over coalitions 
containing i (resp. not containing i) matches the Shapley weighting 
w_S = 1/(n · C(n-1,|S|)). So E[φ̂_i] = φ_i exactly.


## Comparison with Your Original Version

| Aspect                    | Your Version              | Corrected D-SVARM           |
|---------------------------|---------------------------|------------------------------|
| Evals per iteration       | 2 (V(K) and V(K∪A))     | 1 (V(A) only)               |
| What is computed          | Δ = marginal of set A     | v = value of coalition A     |
| Players updated per iter  | Only |A| players           | ALL n players               |
| Background distribution   | X_tot mixed into training  | Not needed                   |
| Theoretical guarantee     | None (marginal-based)      | O(log n / (T - n·log n))    |
| Warmup                    | Missing                    | Ensures full coverage        |
| Decomposition             | Not used                   | φ_i = φ_i^+ - φ_i^-         |


## Adaptation for Edge Valuation (Your Collusion Graph)

When applying D-SVARM to edge valuation in the collusion graph:
  - "Players" = edges {e_1, ..., e_M}
  - "Coalition A" = a subset of edges
  - "Train(A)" = train GCN on graph with only edges in A
  - "Score(·, D_val)" = ROC-AUC on validation nodes
  - Replace n with M (number of edges) everywhere

The same algorithm applies verbatim — just swap data points for edges.