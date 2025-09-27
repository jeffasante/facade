### **Summary of Experimental Phases**

The following table organizes the three specific stages of agent development for straightforward evaluation. The  "Act" functions as a specific research experiment which assesses a defined hypothesis to generate separate  performance metrics along with distinct behavior patterns. The process demonstrates how to identify weaknesses before proposing solutions and  then analyzing the final results.

| Metric / Aspect | Act I: The Naive Aggressor | Act II: The Clever but Flawed Strategist | Act III: The Efficient Strategist |
| :--- | :--- | :--- | :--- |
| **Experimental Goal** | Establish a baseline. Can the agent learn the basic reward signal (clicks)? | Teach the agent patience. Will rewarding 'skip' actions reduce aggression? | Force efficiency. Will a cost-per-action break the passive loop and create a balanced strategy? |
| **Key Intervention** | Baseline reward function (clicks are positive, dropouts are negative). | Added a positive reward (`+0.5`) for the 'skip' action. | Set 'skip' reward to `0.0` and introduced a significant time penalty (`-0.6`) for *every* action. |
| **Agent's Emergent Behavior** | Relentlessly shows ads, ignoring user fatigue and long-term consequences. | Discovers and exploits a loophole by spamming the 'skip' action for safe, steady rewards. | Makes decisive, tactical choices. Ends unprofitable sessions quickly. Balances risk and reward. |
| | | | |
| **Performance Metrics** | | | |
| &nbsp;&nbsp;&nbsp;**Average Reward** | -13.19 | -17.04 (Degraded) | **-14.29 (Recovered)** |
| &nbsp;&nbsp;&nbsp;**Total Dropouts** | 153 | **196 (Peak Failure)** | **150 (Best Performance)** |
| &nbsp;&nbsp;&nbsp;**Avg. Session Length** | ~8.6 steps | **~13.6 steps (Peak Inefficiency)** | **~8.0 steps (Most Efficient)** |
| | | | |
| **Key Finding / Learning** | A simple, greedy strategy is ineffective and leads to high user churn. | Agents can exploit loopholes in the reward function. A simple incentive can lead to unintended, negative emergent behavior. | Introducing a cost for time (efficiency) is critical for forcing a balanced policy. The agent learns to make strategic trade-offs only when inaction is also costly. |