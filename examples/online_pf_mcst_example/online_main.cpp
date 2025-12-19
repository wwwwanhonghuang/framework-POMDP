#include <iostream>
#include <chrono>

#include <Eigen/Dense>

#include <pomdp/belief.hpp>
#include <pomdp/history.hpp>
#include <pomdp/history/sequence_history.hpp>

#include <pomdp/belief/pf_belief.hpp>
#include <pomdp/updater/particle_filter_updater.hpp>

#include <pomdp/planning/planner_runner.hpp>

#include <pomdp/planning/mcst/mcst_planner.hpp>
#include <pomdp/planning/mcst/pomcp_planner.hpp>
#include <pomdp/planning/mcst/ucb_selection.hpp>
#include <pomdp/planning/mcst/progressive_widening.hpp>
#include <pomdp/planning/mcst/composite_selection.hpp>
#include <pomdp/particle_filter/proposal_kernel.hpp>

#include "proposal.hpp"
#include <pomdp/particle_filter/systematic_resampler.hpp>

#include "model.hpp"
#include "action_sampler.hpp"
#include "belief_sampler.hpp"
#include "rollout.hpp"



using namespace online_example;



// 真实系统（state 不可见）
//     ↓ 执行动作
// 观测 o_t
//     ↓
// Belief 更新（Particle Filter）
//     ↓
// 给定 (belief, history)
//     ↓
// Monte Carlo Tree Search 规划
//     ↓
// 选一个动作

// Layer 0：真实世界（本体层）
    // true_state_t
    //     ↓ 执行动作 a_t
    // 环境动力学
    //     ↓
    // (true_state_{t+1}, observation o_{t+1}, reward r_t)

// Layer 1：信念更新（认识论层，外部）
    // (belief_t, history_t)
    //     + action a_t
    //     + observation o_{t+1}  [这里的observation非算法内部observation，可能由传感器得到。对于planning节点，算法内部在规划时存在假想observation，需要进行区分。]
    //     ↓
    // belief_{t+1}

        // 这是 Bayes filtering
        // 不在 planner 内
        // 可替换为任何 belief 表示（PF / Gaussian / learned）


        // 在 POMDP 中，**标准的贝叶斯更新**是：
        // $$
        // \boxed{
        // b_{t+1}(s')
        // \;\propto\;
        // p(o_{t+1} \mid s', a_t)
        // \;\sum_{s}
        // p(s' \mid s, a_t)\, b_t(s)
        // }
        // $$
        // 解释每一项：

        // - $b_t(s)$：旧 belief（我之前对状态的相信）
        // - $p(s' \mid s, a_t)$：状态转移模型
        // - $p(o_{t+1} \mid s', a_t)$：观测模型
        // - “$\propto$”：最后要归一化

        // 这一步**完全在 planner 外部完成**。  可使用粒子滤波进行实现。是对该过程的蒙特卡洛近似。 “蒙特卡洛近似”就是：
        // 当一个期望 / 积分算不出来时，
        // 用大量随机样本的平均来近似它。

            // PF details
            // ## Step 1：预测（Prediction / Proposal）

            // 对每个粒子：
            // $$
            // x_{t+1}^i \sim p(s' \mid x_t^i, a_t)
            // $$
            // 在你的代码中对应：

            // ```
            // BootstrapProposal<State> proposal(model);
            // ```

            // 语义是：

            // > 用 **状态转移模型** 把粒子“往前推一步”

            // ------

            // ## Step 2：观测校正（Correction / Weighting）

            // 用**真实 observation** $o_{t+1}$ 更新权重：
            // $$
            // w_{t+1}^i
            // \;\propto\;
            // p(o_{t+1} \mid x_{t+1}^i, a_t)
            // $$
            // 代码语义（在 updater 内部）：

            // ```
            // weight *= observation_likelihood(x_next, a, obs)
            // ```

            // 👉 这是 **贝叶斯公式中的似然项**

            // ------

            // ## Step 3：归一化（Normalization）

            // $$
            // \sum_i w_{t+1}^i = 1
            // $$

            // 你在 `ParticleBelief::normalize()` 中已经提供了接口。

            // ------

            // ## Step 4：重采样（Resampling，可选）

            // 当有效样本数（ESS）过低：
            // $$
            // \text{ESS} = \frac{1}{\sum_i (w_i)^2}
            // $$
            // 就执行重采样，防止粒子退化：

// Layer 2：规划树的形态（形态学层）
    // 节点 = history h
    // 边 = action a → observation o
    // 每个 history 语义等价于一个 belief

        // 这是 POMDP 的天然决策空间

        // 与 MCST、算法、代码实现无关

// Layer 3：规划行为（行为学层，planner / MCST）
    // input belief_{t+1}, history_{t+1}

    // 1. 从 belief 采样一个 state s
    // 2. 从根 history 发起一次 simulate(s, h, depth=0)
    // 3. simulate 中：
    //    - selection（PW / UCB）
    //    - expansion（新 action / history）
    //    - simulation（generative model 产生 s', o, r） generative model 是一个在“决策相关层面”上近似真实世界动力学与感知过程的可采样模型，用以模拟状态、观测与回报的联合演化。
    //    - rollout（树外默认策略）
    //    - backup（更新 Q, N）

// Layer 4：动作决策（接口层）
    // planner 输出的唯一结果：
    // a* = argmax_a Q(h_root, a)


        // planner是在进行 给定 (belief, history) ↓ Monte Carlo Tree Search 规划 ↓ 选一个动作 的步骤。
        // planner 在“规划树”中进行拓展，
        // 这棵树只存在于 planner 内部，是一个“假想树”。

// “基于 history 的规划树（history–action–history）”

// 形式是：

// h₀
//  ├── a₁
//  │    └── h₁ = h₀ + (a₁, o₁)
//  ├── a₂
//  │    └── h₂ = h₀ + (a₂, o₂)

// 在 POMCP / MCST 中，树是 两类节点交替出现的：

// History Node (belief node)
//     ↓ choose action
// Action Node
//     ↓ sample transition
// History Node
//     ↓ choose action
// Action Node
//     ...

// UCB 只在“同一个 history node 下的 action nodes”之间选

int main() {
    // ------------------------------------------------------------
    // 3. Samplers
    // ------------------------------------------------------------
    ContinuousActionSampler action_sampler(/*a_max=*/1.0);
    PFBeliefSampler<State> belief_sampler;


    // ------------------------------------------------------------
    // 1. Model (ground truth + generative)
    // ------------------------------------------------------------
    ContinuousModel model(action_sampler, /*dt=*/0.1);

    // ------------------------------------------------------------
    // 2. Initial belief (particle filter)
    // ------------------------------------------------------------
    constexpr std::size_t NUM_PARTICLES = 500;


    pomdp::ParticleBelief<State> belief;
    belief.particles.reserve(NUM_PARTICLES);

    for (std::size_t i = 0; i < NUM_PARTICLES; ++i) {
        State x(4);
        x.setZero();

        pomdp::Particle<State> p;
        p.x = x;
        p.weight = 1.0 / NUM_PARTICLES;

        belief.particles.push_back(p);
    }

    BootstrapProposal<State> proposal(model);
    pomdp::SystematicResampler<State> resampler;

    // PF updater (uses model's kernels)
    pomdp::ParticleFilterUpdater<State> updater(
        model,
        proposal,
        resampler,
        /*ess_threshold=*/0.5
    );

 

    // ------------------------------------------------------------
    // 4. Rollout policy
    // ------------------------------------------------------------
    Rollout rollout([&]() {
        return action_sampler.sample_random_action();
    });
    // ------------------------------------------------------------
    // 5. Selection strategy (PW + UCB)
    // ------------------------------------------------------------
    auto pw = std::make_shared<pomdp::mcst::ProgressiveWidening>(
        action_sampler,
        /*k=*/1.0,
        /*alpha=*/0.5
    );

    auto ucb = std::make_shared<pomdp::mcst::UCBSelection>(
        /*c=*/1.4
    );

    pomdp::mcst::CompositeSelection selection;
    selection.add_strategy(pw);
    selection.add_strategy(ucb);

    // ------------------------------------------------------------
    // 6. Planner (MCST or POMCP)
    // ------------------------------------------------------------
    constexpr std::size_t HORIZON = 15;
    constexpr double DISCOUNT = 0.95;

    pomdp::mcst::MCSTPlanner<State> planner(
        belief_sampler,
        action_sampler,
        model,
        selection,
        rollout,
        HORIZON,
        DISCOUNT
    );

    // Online / anytime runner
    pomdp::PlannerRunner runner(planner);

    // ------------------------------------------------------------
    // 7. Execution loop (online)
    // ------------------------------------------------------------
    pomdp::SequenceHistory history;

    pomdp::Action action = action_sampler.sample_random_action();


    State true_state(4);
    true_state.setZero();

    constexpr std::size_t NUM_STEPS = 100;

    for (std::size_t t = 0; t < NUM_STEPS; ++t) {
        // --------------------------------------------------------
        // Observe environment
        // --------------------------------------------------------
        auto sim_result = model.step(true_state, action);
        true_state = sim_result.next_state;
        const pomdp::Observation& obs = sim_result.observation;

        // --------------------------------------------------------
        // Belief update (external to planner)
        // --------------------------------------------------------
        const pomdp::Action prev_action = action;

        auto new_belief = updater.update(
            belief,
            prev_action,
            obs,
            history
        );
        belief = *dynamic_cast<pomdp::ParticleBelief<State>*>(new_belief.release());

        // --------------------------------------------------------
        // Online planning (time-bounded)
        // --------------------------------------------------------
        runner.run_for_duration(
            belief,
            history,
            std::chrono::milliseconds(10)
        );

        // --------------------------------------------------------
        // Select best action anytime
        // --------------------------------------------------------
        action = planner.best_action();

        // --------------------------------------------------------
        // Logging
        // --------------------------------------------------------
        const auto& u = action_sampler.action_value(action);

        std::cout
            << "t=" << t
            << "  true_pos=("
            << true_state[0] << ", "
            << true_state[1] << ")"
            << " action=("
            << u[0] << ", "
            << u[1] << ") "
            << std::endl;

        // --------------------------------------------------------
        // Update history
        // --------------------------------------------------------
        
        history.append(action, obs);
    }

    return 0;
}
