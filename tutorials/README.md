# POMCP Tutorial – Part I (Mathematical Details)

## Extended Mathematical Formulation

---

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

### 3.2 Effective Sample Size (ESS)
\[
\text{ESS}_t = \frac{1}{\sum_{i=1}^N (w_t^{(i)})^2} \in [1, N]
\]

Resampling occurs when:
\[
\text{ESS}_t < \alpha N, \quad \alpha \in (0, 1] \text{ (typically } \alpha = 0.5)
\]

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

