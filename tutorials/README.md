# POMCP Tutorial – Part I (Mathematical Details)

## 1. Complete POMDP Formulation

### 1.1 Core POMDP Tuple
A POMDP is formally defined as a 7-tuple:
\[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, T, \Omega, R, \gamma)
\]
where:

1. **State space**: \( \mathcal{S} \) (possibly continuous/infinite)
2. **Action space**: \( \mathcal{A} \) (finite or continuous)
3. **Observation space**: \( \mathcal{O} \) (finite or continuous)
4. **Transition function**: \( T(s' \mid s, a) = p(s' \mid s, a) \)
5. **Observation function**: \( \Omega(o \mid s', a) = p(o \mid s', a) \)
6. **Reward function**: \( R(s, a, s') \) or \( R(s, a) \)
7. **Discount factor**: \( \gamma \in [0, 1] \)

### 1.2 Belief State
The belief state \( b_t \) is a probability distribution over \( \mathcal{S} \):
\[
b_t(s) = \Pr(s_t = s \mid h_t)
\]
where the **history** is:
\[
h_t = (a_0, o_1, a_1, o_2, \dots, a_{t-1}, o_t)
\]

### 1.3 Belief-MDP Reformulation
The POMDP induces a continuous-state MDP:
- **State**: Belief \( b \in \Delta(\mathcal{S}) \) (simplex over states)
- **Transition**: \( \tau(b, a, o) \) where \( b' = \text{BayesUpdate}(b, a, o) \)
- **Reward**: \( \rho(b, a) = \mathbb{E}_{s \sim b}[R(s, a)] \)



## 2. Exact Bayesian Filtering Mathematics

### 2.1 Bayesian Filtering Equations

#### 2.1.1 Prediction Step (Chapman-Kolmogorov)
\[
b_{t+1}^-(s') = \sum_{s \in \mathcal{S}} T(s' \mid s, a_t) b_t(s)
\]
or for continuous states:
\[
b_{t+1}^-(s') = \int_{\mathcal{S}} T(s' \mid s, a_t) b_t(s) ds
\]

#### 2.1.2 Update Step (Bayes' Rule)
\[
b_{t+1}(s') = \frac{\Omega(o_{t+1} \mid s', a_t) b_{t+1}^-(s')}{\eta(o_{t+1} \mid b_t, a_t)}
\]
where the **normalization constant** (evidence/marginal likelihood) is:
\[
\eta(o_{t+1} \mid b_t, a_t) = \sum_{s' \in \mathcal{S}} \Omega(o_{t+1} \mid s', a_t) b_{t+1}^-(s')
\]
or for continuous:
\[
\eta(o_{t+1} \mid b_t, a_t) = \int_{\mathcal{S}} \Omega(o_{t+1} \mid s', a_t) b_{t+1}^-(s') ds'
\]

### 2.2 Recursive Belief Update
Combining both steps:
\[
\boxed{
b_{t+1}(s') = \frac{\Omega(o_{t+1} \mid s', a_t) \sum_{s \in \mathcal{S}} T(s' \mid s, a_t) b_t(s)}{\sum_{s' \in \mathcal{S}} \Omega(o_{t+1} \mid s', a_t) \sum_{s \in \mathcal{S}} T(s' \mid s, a_t) b_t(s)}
}
\]

### 2.3 Matrix Form (Discrete Case)
For discrete finite spaces:
- Let \( \mathbf{b}_t \) be belief vector: \( b_t[i] = \Pr(s_t = s_i) \)
- Let \( \mathbf{T}_a \) be transition matrix: \( T_a[i,j] = \Pr(s_{t+1} = s_j \mid s_t = s_i, a) \)
- Let \( \mathbf{O}_{a,o} \) be diagonal observation matrix: \( O_{a,o}[i,i] = \Pr(o \mid s_i, a) \)

Then:
\[
\mathbf{b}_{t+1} = \frac{\mathbf{O}_{a_t, o_{t+1}} \mathbf{T}_{a_t}^\top \mathbf{b}_t}{\mathbf{1}^\top \mathbf{O}_{a_t, o_{t+1}} \mathbf{T}_{a_t}^\top \mathbf{b}_t}
\]
where \( \mathbf{1} \) is vector of ones.



### 2.4 Bayesian Filtering Practices

Bayesian filtering solves the recursive estimation problem:
\[
b_{t+1}(s) \propto p(o_{t+1} \mid s, a_t) \int p(s \mid s', a_t) b_t(s') ds'
\]

Different methods make different trade-offs between **accuracy**, **computational cost**, and **representational power**.

#### 2.4.0. Particle Filter

See Section 3.



#### 2.4.1. **Kalman Filter (KF)** - The Gaussian Specialist

##### Assumptions:
- **Linear** dynamics: \( s_{t+1} = F s_t + G a_t + w_t \)
- **Linear** observations: \( o_t = H s_t + v_t \)
- **Gaussian** noise: \( w_t \sim \mathcal{N}(0, Q) \), \( v_t \sim \mathcal{N}(0, R) \)
- **Gaussian** belief: \( b_t(s) = \mathcal{N}(s \mid \mu_t, \Sigma_t) \)

##### Update Equations:
**Prediction**:
\[
\mu_{t+1}^- = F \mu_t + G a_t
\]
\[
\Sigma_{t+1}^- = F \Sigma_t F^\top + Q
\]

**Update** (Kalman Gain):
\[
K_{t+1} = \Sigma_{t+1}^- H^\top (H \Sigma_{t+1}^- H^\top + R)^{-1}
\]
\[
\mu_{t+1} = \mu_{t+1}^- + K_{t+1}(o_{t+1} - H \mu_{t+1}^-)
\]
\[
\Sigma_{t+1} = (I - K_{t+1} H) \Sigma_{t+1}^-
\]

##### Pros:
- **Optimal** for linear Gaussian systems
- **Exact** closed-form solution
- **Efficient**: \( O(n^3) \) for n-dimensional state

##### Cons:
- **Only works for linear Gaussian systems**
- Cannot represent multimodal beliefs

##### Variants:
- **Extended Kalman Filter (EKF)**: Linearizes nonlinear functions
- **Unscented Kalman Filter (UKF)**: Uses sigma points for nonlinear transforms
- **Ensemble Kalman Filter (EnKF)**: Monte Carlo version for high dimensions



#### 2.4.2. **Extended Kalman Filter (EKF)** - Nonlinear Approximation

##### Approach:
Linearize nonlinear dynamics/observations via Taylor expansion:
\[
s_{t+1} = f(s_t, a_t) + w_t, \quad o_t = h(s_t) + v_t
\]
\[
F_t = \frac{\partial f}{\partial s}\Big|_{s=\mu_t}, \quad H_t = \frac{\partial h}{\partial s}\Big|_{s=\mu_t^-}
\]

##### Update:
Same as KF but with time-varying \( F_t, H_t \)

##### Pros:
- Handles **mild nonlinearities**
- Widely used in robotics (SLAM, navigation)

##### Cons:
- **Approximation error** from linearization
- Can diverge for highly nonlinear systems
- Computes Jacobians (can be complex)



#### 2.4.3. **Unscented Kalman Filter (UKF)** - Sigma Point Method

### Core Idea:
Instead of linearizing, propagate **sigma points** through nonlinear functions:

**Sigma points** (2n+1 points for n-dim state):
\[
\mathcal{X}_0 = \mu, \quad \mathcal{X}_i = \mu \pm \sqrt{(n+\lambda)\Sigma}_i
\]
where \( \sqrt{\Sigma}_i \) is i-th column of matrix square root.

### Algorithm:
1. **Generate sigma points** from current belief
2. **Propagate each** through \( f \) and \( h \)
3. **Compute** predicted mean/covariance from transformed points

### Pros:
- **Better than EKF** for strong nonlinearities
- **No Jacobians** needed
- Same complexity as EKF (\( O(n^3) \))

### Cons:
- Still assumes **unimodal Gaussian** belief
- Can fail for highly non-Gaussian distributions



#### 2.4.4. **Grid-Based Filters** - Discrete State Solution

##### Approach:
Discretize state space into **finite grid**:
\[
\mathcal{S} = \{s_1, s_2, ..., s_N\}
\]
Represent belief as probability vector: \( b_t[i] = \Pr(s_t = s_i) \)

##### Update Equations:
\[
b_{t+1}[j] = \eta \cdot p(o_{t+1} \mid s_j, a_t) \sum_{i=1}^N p(s_j \mid s_i, a_t) b_t[i]
\]
where \( \eta \) normalizes.

##### Pros:
- **Exact** for discrete state spaces
- Can represent **multimodal** beliefs
- Simple implementation

##### Cons:
- **Curse of dimensionality**: \( N = m^d \) for d dimensions with m points each
- **Fixed resolution**: Cannot refine locally
- **Memory**: Stores \( N \times N \) transition matrix



#### 2.4.5. **Histogram Filter** - Grid Filter Variant

##### Approach:
Partition continuous space into **bins**, assume uniform distribution within each bin.

##### Update:
Similar to grid filter but with **integrated probabilities**:
\[
b_{t+1}(B_j) = \eta \cdot \int_{s \in B_j} p(o_{t+1} \mid s, a_t) \sum_i \int_{s' \in B_i} p(s \mid s', a_t) ds' b_t(B_i) ds
\]

##### Pros:
- Handles continuous states approximately
- Simpler than particle filters for low dimensions

##### Cons:
- **Resolution vs memory** trade-off
- **Blocky** approximations
- Still exponential in dimensions



#### 2.4.6. **Information Filter** - Dual Representation

##### Core Idea:
Use **canonical parameters** instead of moments:
\[
b(s) \propto \exp\left(-\frac{1}{2} s^\top \Omega s + \xi^\top s\right)
\]
where \( \Omega = \Sigma^{-1} \) (information matrix), \( \xi = \Sigma^{-1} \mu \) (information vector)

##### Update Equations:
**Prediction** (messy in information form):
\[
\Omega_{t+1}^- = (F \Omega_t^{-1} F^\top + Q)^{-1}
\]
\[
\xi_{t+1}^- = \Omega_{t+1}^- F \Omega_t^{-1} \xi_t
\]

**Update** (simple!):
\[
\Omega_{t+1} = \Omega_{t+1}^- + H^\top R^{-1} H
\]
\[
\xi_{t+1} = \xi_{t+1}^- + H^\top R^{-1} o_{t+1}
\]

##### Pros:
- **Update is additive** (simple for multiple sensors)
- Easy to represent **no information** (\(\Omega = 0\))
- Natural for **decentralized** filtering

##### Cons:
- **Prediction is complex** (requires matrix inversion)
- Equivalent to KF, just different representation



#### 2.4.7. **Rao-Blackwellized Particle Filter (RBPF)** - Hybrid Approach

##### Core Idea:
Decompose state: \( s = [s_1, s_2] \)
- \( s_1 \): Nonlinear part → track with particles
- \( s_2 \): Conditionally linear given \( s_1 \) → track with KF per particle

##### Example (SLAM):
- **Particle**: Robot pose hypothesis
- **Per-particle KF**: Map landmarks given that pose

##### Algorithm:
```
For each particle i:
    # Sample nonlinear part
    s1^{(i)} ~ p(s1 | s1^{(i)}, a)
    
    # Update linear part analytically
    s2^{(i)} updated using KF conditioned on s1^{(i)}
    
    # Compute weight
    w^{(i)} ∝ p(o | s1^{(i)}, s2^{(i)})
```

##### Pros:
- **Exploits** conditional linear structure
- More efficient than pure PF
- Can handle high-dimensional linear substructures

##### Cons:
- Needs **conditional linearity**
- More complex implementation



#### 2.4.8. **Point-Mass Filter** - Adaptive Grid

##### Approach:
Place grid points **adaptively** based on belief:
1. Start with coarse grid
2. Refine regions of high probability
3. Coarsen regions of low probability

##### Pros:
- **Adaptive resolution**
- More efficient than fixed grid

##### Cons:
- Complex grid management
- Still exponential in worst case



#### 2.4.9. **Moment Matching Approximations**

##### Approach:
Approximate belief by matching **moments**:
1. Propagate moments through nonlinearities
2. Approximate posterior with simple distribution (e.g., Gaussian)

##### Methods:
- **Assumed Density Filtering (ADF)**
- **Expectation Propagation (EP)**

##### Pros:
- Can handle some non-Gaussianities
- More general than KF

##### Cons:
- **Approximation error**
- Computationally intensive for high moments



#### 2.4.10. **Variational Bayesian Filters**

##### Approach:
Approximate posterior with tractable distribution \( q(s) \) by minimizing KL-divergence:
\[
q^* = \arg\min_{q \in \mathcal{Q}} \text{KL}(q(s) \| p(s \mid o_{1:t}))
\]

##### Pros:
- Can handle complex models
- Provides uncertainty estimates

##### Cons:
- **Non-convex optimization**
- Approximate
- Computationally heavy



#### Comparison Table

| Filter          | State Space | Belief Form        | Complexity | Optimality                    |
| --------------- | ----------- | ------------------ | ---------- | ----------------------------- |
| **Kalman**      | Continuous  | Gaussian           | O(n³)      | Optimal (linear Gaussian)     |
| **EKF**         | Continuous  | Gaussian           | O(n³)      | Approximate                   |
| **UKF**         | Continuous  | Gaussian           | O(n³)      | Approximate (better than EKF) |
| **Grid**        | Discrete    | Multimodal         | O(N²)      | Exact for discrete            |
| **Histogram**   | Continuous  | Piecewise constant | O(B²)      | Approximate                   |
| **Particle**    | Any         | Empirical          | O(N×T)     | Approximate (exact as N→∞)    |
| **RBPF**        | Mixed       | Hybrid             | O(N×n³)    | Approximate                   |
| **Information** | Continuous  | Gaussian           | O(n³)      | Equivalent to KF              |

---

## When to Use Which?

### **Use Kalman Filter when**:
- System is linear Gaussian
- You need exact optimal solution
- Computational efficiency is critical

### **Use EKF/UKF when**:
- Mild to moderate nonlinearities
- State dimension moderate (<100)
- Gaussian assumption holds

### **Use Particle Filter when**:
- Strong nonlinearities/non-Gaussian noise
- Multimodal beliefs possible
- Can afford computational cost
- Don't have analytical models (only generative)

### **Use Grid Filter when**:
- Low-dimensional state space (1-3D)
- Discrete or finely discretized
- Need exact inference
- Memory is not limiting

### **Use RBPF when**:
- Mixed linear/nonlinear structure
- High-dimensional but with conditional linearity (e.g., SLAM)

### **Use Information Filter when**:
- Multiple sensors fusing information
- Decentralized filtering needed
- Initial uncertainty is large/unbounded

---

## In POMDP Context

For **POMDP planning**, the choice depends on:

1. **State space complexity**:
   - Continuous/high-dim → PF or RBPF
   - Discrete/low-dim → Grid filter

2. **Real-time requirements**:
   - Fast updates needed → KF/EKF/UKF
   - Can afford computation → PF

3. **Belief multimodality**:
   - Unimodal → KF family
   - Multimodal → PF or Grid

4. **Model availability**:
   - Analytical models → KF/EKF/UKF
   - Only generative model → PF

**POMCP specifically** often uses Particle Filters because:
1. Works with **any generative model** (black-box simulator)
2. Handles **multimodal beliefs** (important for planning)
3. Scales better to **high dimensions** than grid methods
4. Naturally provides **samples** for Monte Carlo planning

However, **DESPOT** (another POMDP solver) sometimes uses **weighted particle sets** with **scenario trees**, which is conceptually similar but deterministically explores a fixed set of scenarios.

The key insight: **Particle filters provide the most general Bayesian filtering framework** for POMDPs, trading off optimality for generality and scalability.



## 3. Particle Filter Mathematics

### 3.1 Sequential Importance Sampling

#### 3.1.1 Weight Update Derivation
We want to approximate:
\[
b_t(s) = p(s \mid h_t) \propto p(o_t \mid s, a_{t-1}) p(s \mid h_{t-1})
\]

Using importance sampling with proposal distribution \( q(s \mid h_t) \):
\[
w_t^{(i)} \propto \frac{p(x_t^{(i)} \mid h_t)}{q(x_t^{(i)} \mid h_t)}
\]

For sequential filtering, use prior as proposal: \( q(s \mid h_t) = p(s \mid h_{t-1}) \)
\[
w_t^{(i)} \propto \frac{p(o_t \mid x_t^{(i)}, a_{t-1}) p(x_t^{(i)} \mid h_{t-1})}{p(x_t^{(i)} \mid h_{t-1})} = p(o_t \mid x_t^{(i)}, a_{t-1})
\]

#### 3.1.2 Algorithmic Weight Update
For particle \( i \):
\[
\tilde{w}_{t+1}^{(i)} = w_t^{(i)} \cdot \Omega(o_{t+1} \mid x_{t+1}^{(i)}, a_t)
\]
Normalized:
\[
w_{t+1}^{(i)} = \frac{\tilde{w}_{t+1}^{(i)}}{\sum_{j=1}^N \tilde{w}_{t+1}^{(j)}}
\]

Operationally, it weighting the particle by observation likelihood (e.g., the sensor proposed likelihood. Or assume sensor output is gaussian and solve the likelihood of the particle)

### 3.2 Effective Sample Size (ESS)

\[
\text{ESS}_t = \frac{1}{\sum_{i=1}^N (w_t^{(i)})^2} \in [1, N]
\]

Resampling occurs when:
\[
\text{ESS}_t < \alpha N, \quad \alpha \in (0, 1] \text{ (typically } \alpha = 0.5)
\]

Specifically, it resample N particles based on weight distribution, as a result, high-weight particles are replicated multiple times

and low-weight particles are eliminated. All weights reset to 1/N. It is for the purpose of preventing particle degeneration and focus on credible hypotheses. （Note: usually, after copying a particle when resampling, we may intentionally add small noise to the particle.state, this process a.k.a. **jittering**）



Imagine there is a robot, we don't know its coordinate at x-axis. Initially, we build 2 particles, (weight: 0.5, x = 1), and (weight: 0.5, x = 10). Then perform an **action** x += 1. They belief are (weight: 0.5, x = 2), (weight: 0.5, x = 11). Then perform an **observation**, may be from a sensor, it tell us robot at x = 9. Unfortunately, the sensor also not accurate, but its output is gaussian. So based on this, we reassigning weights to each particle, then normalize it. Then we get a new belief.

> The belief updation process in PF: belief[t] -> action -> observation -> particle filter or other kinds of beyesian filtering strategy -> belieft[t+1] 

If we can understand this process. We may understand why in the tree, there are two kinds of node.

==Kind-1 Node -> action -> kind-2 Node -> observation -> Kind-1 node==



### 3.3 Systematic Resampling Algorithm

Let \( u \sim \mathcal{U}(0, 1/N) \), then select particles with indices:
\[
i_j = \min\left\{ k : \sum_{i=1}^k w_t^{(i)} \geq \frac{j-1}{N} + u \right\}, \quad j = 1, \dots, N
\]



## 4. History Tree Mathematics

### 4.1 Belief at History Node
For history \( h = (a_0, o_1, \dots, a_{t-1}, o_t) \):
\[
b_h(s) = \Pr(s \mid h) = \frac{\Pr(o_t \mid s, a_{t-1}) \sum_{s'} T(s \mid s', a_{t-1}) b_{h'}(s')}{\eta(o_t \mid b_{h'}, a_{t-1})}
\]
where \( h' = (a_0, o_1, \dots, a_{t-2}, o_{t-1}) \).

### 4.2 Recursive Belief Computation
For \( h = h' \oplus (a, o) \):
\[
b_h(s) = \frac{\Omega(o \mid s, a) \sum_{s'} T(s \mid s', a) b_{h'}(s')}{\sum_{s} \Omega(o \mid s, a) \sum_{s'} T(s \mid s', a) b_{h'}(s')}
\]

### 4.3 Conditional Independencies
Given Markov property:
\[
\Pr(s_{t+1} \mid h_t, a_t) = \Pr(s_{t+1} \mid b_t, a_t)
\]
\[
\Pr(o_{t+1} \mid h_t, a_t) = \Pr(o_{t+1} \mid b_t, a_t)
\]



## 5. MCTS Mathematics for POMDPs

### 5.1 Value Function Representation

#### 5.1.1 Optimal Value Function (Belief Space)
The optimal value function satisfies Bellman equation:
\[
V^*(b) = \max_{a \in \mathcal{A}} \left[ \rho(b, a) + \gamma \sum_{o \in \mathcal{O}} \Pr(o \mid b, a) V^*(\tau(b, a, o)) \right]
\]
where:
\[
\rho(b, a) = \sum_{s \in \mathcal{S}} b(s) R(s, a)
\]
\[
\Pr(o \mid b, a) = \sum_{s' \in \mathcal{S}} \Omega(o \mid s', a) \sum_{s \in \mathcal{S}} T(s' \mid s, a) b(s)
\]

#### 5.1.2 Q-value Representation
For history-based representation:
\[
Q(h, a) = \mathbb{E} \left[ \sum_{k=0}^\infty \gamma^k r_{t+k} \mid h_t = h, a_t = a \right]
\]

### 5.2 UCB1 Action Selection

#### 5.2.1 Standard UCB1
For node \( h \), action \( a \):
\[
\text{UCB}(h, a) = \frac{Q(h, a)}{N(h, a)} + c \sqrt{\frac{\log N(h)}{N(h, a)}}
\]
where:
- \( Q(h, a) \): Cumulative return from action \( a \)
- \( N(h, a) \): Visit count of action \( a \) from \( h \)
- \( N(h) = \sum_a N(h, a) \): Total visits to \( h \)
- \( c \): Exploration constant (typically \( c = \sqrt{2} \))

#### 5.2.2 POMCP Modification
In POMCP, we use:
\[
\text{UCB}_{\text{POMCP}}(h, a) = \hat{Q}(h, a) + c \sqrt{\frac{\log N(h)}{N(h, a)}}
\]
where \( \hat{Q}(h, a) \) is estimated from simulations.

### 5.3 Progressive Widening

#### 5.3.1 Action Progressive Widening
Add new action when:
\[
N(h, a) > \tau_a N(h)^{\alpha_a}
\]
Parameters:
- \( \tau_a > 0 \): Threshold constant
- \( \alpha_a \in (0, 1] \): Growth rate (typically 0.5)

#### 5.3.2 Observation Progressive Widening
For continuous observations, cluster observations when:
\[
|\{o : N(h, a, o) > 0\}| > \tau_o N(h, a)^{\alpha_o}
\]

### 5.4 Value Backup Formulas

#### 5.4.1 Monte Carlo Return
For simulation trajectory \( \tau = (s_0, a_0, o_1, r_0, s_1, a_1, \dots) \):
\[
R(\tau) = \sum_{k=0}^{L-1} \gamma^k r_k
\]
where \( L \) is trajectory length (simulation horizon).

#### 5.4.2 Incremental Update
After simulation with return \( R \):
\[
Q(h, a) \leftarrow Q(h, a) + \frac{R - Q(h, a)}{N(h, a)}
\]
Equivalently:
\[
Q_{\text{new}}(h, a) = \frac{N(h, a) \cdot Q_{\text{old}}(h, a) + R}{N(h, a) + 1}
\]

#### 5.4.3 Discounted Backup
For depth \( d \), with immediate reward \( r \) and future value \( V \):
\[
Q(h, a) \leftarrow Q(h, a) + \frac{(r + \gamma V) - Q(h, a)}{N(h, a)}
\]

## 6. Generative Model Mathematics

### 6.1 Complete Generative Process
Given current state \( s \) and action \( a \):

1. **Sample next state**:
   \[
   s' \sim T(\cdot \mid s, a)
   \]

2. **Sample observation**:
   \[
   o \sim \Omega(\cdot \mid s', a)
   \]

3. **Compute reward**:
   \[
   r = R(s, a, s')
   \]

The joint probability:
\[
\Pr(s', o, r \mid s, a) = \delta(r - R(s, a, s')) \cdot \Omega(o \mid s', a) \cdot T(s' \mid s, a)
\]
where \( \delta(\cdot) \) is Dirac delta for deterministic rewards.

### 6.2 Expected Values

#### 6.2.1 Expected Reward
\[
\mathbb{E}[r \mid s, a] = \sum_{s'} T(s' \mid s, a) R(s, a, s')
\]

#### 6.2.2 Expected Observation Probability
\[
\Pr(o \mid s, a) = \sum_{s'} \Omega(o \mid s', a) T(s' \mid s, a)
\]



## 7. Convergence Analysis

### 7.1 Consistency of Particle Filter
As \( N \to \infty \), for bounded \( f \):
\[
\lim_{N \to \infty} \sum_{i=1}^N w_t^{(i)} f(x_t^{(i)}) \overset{a.s.}{=} \mathbb{E}_{s \sim b_t}[f(s)]
\]

### 7.2 Consistency of POMCP
Under assumptions:
1. Generative model matches true dynamics
2. Infinite computational budget (\( T \to \infty \))
3. Sufficient exploration

Then:
\[
\lim_{T \to \infty} Q_T(h, a) \overset{a.s.}{=} Q^*(h, a)
\]
where \( Q_T \) is estimate after \( T \) simulations.

### 7.3 Regret Bounds
For finite-horizon POMDP with horizon \( H \), with appropriate exploration constant \( c \):
\[
\mathbb{E}[\text{Regret}(T)] = O\left( \sqrt{|\mathcal{A}| H^3 T \log T} \right)
\]



## 8. Implementation Formulas

### 8.1 Tree Node Statistics
For node representing history \( h \):
- **Visit count**: \( N(h) \in \mathbb{N} \)
- **Action counts**: \( N(h, a) \in \mathbb{N} \)
- **Action values**: \( Q(h, a) \in \mathbb{R} \)
- **Children**: \( C(h, a, o) \) for each observation

### 8.2 Action Selection Pseudocode

```python
def select_action(node, exploration_constant=1.414):
    best_action = None
    best_value = -inf
    
    for a in node.actions:
        if node.N(a) == 0:
            return a  # Return unexplored action immediately
        
        # Upper Confidence Bound
        value = node.Q(a) / node.N(a) + exploration_constant * \
                sqrt(log(node.N()) / node.N(a))
        
        if value > best_value:
            best_value = value
            best_action = a
    
    return best_action
```

### 8.3 Simulation Return Calculation

```python
def calculate_return(rewards, discount=0.95):
    """
    rewards: list of rewards [r0, r1, ..., r_{L-1}]
    returns: discounted sum R = Σ_{k=0}^{L-1} γ^k r_k
    """
    total = 0
    current_discount = 1
    for r in rewards:
        total += current_discount * r
        current_discount *= discount
    return total
```

### 8.4 Incremental Mean Update
For new sample \( x_n \):
\[
\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}
\]
Variance can be updated similarly:
\[
\sigma_n^2 = \frac{(n-1)\sigma_{n-1}^2 + (x_n - \mu_{n-1})(x_n - \mu_n)}{n}
\]



## 9. Practical Considerations with Formulas

### 9.1 Numerical Stability

#### 9.1.1 Log-Domain Calculations
For small probabilities, work in log-domain:
\[
\log b_{t+1}(s') = \log \Omega(o_{t+1} \mid s', a_t) + \log \sum_s T(s' \mid s, a_t) b_t(s) - \log \eta
\]

Use log-sum-exp trick:
\[
\log \sum_i \exp(x_i) = m + \log \sum_i \exp(x_i - m)
\]
where \( m = \max_i x_i \).

#### 9.1.2 Particle Filter Resampling
To avoid weight degeneracy:
\[
w_t^{(i)} \leftarrow \exp(\log w_t^{(i)} - \log \sum_j \exp(\log w_t^{(j)}))
\]

### 9.2 Reward Scaling
For stability, scale rewards to \( [-1, 1] \):
\[
r_{\text{scaled}} = \frac{r - r_{\min}}{r_{\max} - r_{\min}} \times 2 - 1
\]

### 9.3 Discount Factor Effects
Effective horizon:
\[
H_{\text{eff}} = \frac{1}{1-\gamma}
\]
For \( \gamma = 0.95 \), \( H_{\text{eff}} = 20 \); for \( \gamma = 0.99 \), \( H_{\text{eff}} = 100 \).



## 10. Extended Example: Tiger Problem

### 10.1 Problem Specification
- States: \( \mathcal{S} = \{\text{left}, \text{right}\} \) (tiger location)
- Actions: \( \mathcal{A} = \{\text{listen}, \text{open-left}, \text{open-right}\} \)
- Observations: \( \mathcal{O} = \{\text{hear-left}, \text{hear-right}\} \)

### 10.2 Transition Probabilities
\[
T(\text{left} \mid s, \text{listen}) = \delta(s, \text{left})
\]
\[
T(\text{right} \mid s, \text{listen}) = \delta(s, \text{right})
\]
\[
T(s' \mid s, \text{open-*}) = \text{Uniform} \quad \text{(tiger resets)}
\]

### 10.3 Observation Probabilities
\[
\Omega(\text{hear-left} \mid \text{left}, \text{listen}) = 0.85
\]
\[
\Omega(\text{hear-right} \mid \text{left}, \text{listen}) = 0.15
\]
\[
\Omega(\text{hear-left} \mid \text{right}, \text{listen}) = 0.15
\]
\[
\Omega(\text{hear-right} \mid \text{right}, \text{listen}) = 0.85
\]

### 10.4 Reward Function
\[
R(s, \text{open-left}) = 
\begin{cases}
-100 & \text{if } s = \text{left} \\
+10 & \text{if } s = \text{right}
\end{cases}
\]
\[
R(s, \text{open-right}) = 
\begin{cases}
+10 & \text{if } s = \text{left} \\
-100 & \text{if } s = \text{right}
\end{cases}
\]
\[
R(s, \text{listen}) = -1 \quad \forall s
\]

### 10.5 Belief Update Example
Initial belief: \( b_0 = [0.5, 0.5] \)

After action `listen`, observation `hear-left`:
\[
b_1(\text{left}) = \frac{0.85 \times 0.5}{0.85 \times 0.5 + 0.15 \times 0.5} = 0.85
\]
\[
b_1(\text{right}) = \frac{0.15 \times 0.5}{0.85 \times 0.5 + 0.15 \times 0.5} = 0.15
\]

