# Paper Summarization

My notes on paper, [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290).  

## A brief summary on OpenAI RLHF papers

3 papers:
* \[1\] [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) from 2019
* \[2\] [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) from 2020
* \[3\] [Training language models to follow instructions with human feedback (instructGPT)](https://arxiv.org/abs/2203.02155) from 2022
### Reward model

The reward model $r_\theta: X \times Y \to R$  maps a prompt response pair $(x, y)$ to a scalar reward.

In [1], the reward model is formulated as a best-of-4 classification problem. Specifically, having collected a dataset $D$ of $(x, y_0, y_1, y_2, y_3, b)$ tuples, where $b \in \{0, 1, 2, 3\}$ is the best option human labelers select from four options $(y_0, y_1, y_2, y_3)$， We fit a reward model $r_\theta$ using the 4-class cross-entropy loss:

$\displaystyle loss(r_\theta) = -\log P(y_b|x) = -E_{(x, y_0, y_1, y_2, y_3, b)\sim D}[\log\frac{e^{r_\theta(x, y_b)}}{\sum_ie^{r_\theta(x, y_i)}}]$

In [2], given a dataset $D$ of $(x, y_w, y_l)$ tuple, where human labelers prefer $y_w$, the 2-class cross entropy loss:

$\displaystyle loss(r_\theta) = -\log P(y_w| x) = -E_{(x, y_w, y_l) \sim D}[\log\frac{e^{r_\theta(x, y_w)}}{e^{r_\theta(x, y_w)} + e^{r_\theta(x, y_l)}}]$

$= -E_{(x, y_w, y_l) \sim D}[\log\frac{1}{1 + e^{-(r_\theta(x,y_w) - r_\theta(x,y_l))}}] = -E_{(x, y_w, y_l) \sim D}[\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))$.   (A)

where $\sigma$ is the sigmoid function.

In \[3\], in order to speed up comparison collection, human labelers rank a group of $K=4$ or $K=9$ responses. This produces $K \choose 2$ comparisons for each prompt $x$. If each of the possible $K \choose 2$ comparisons is treated as a separate data point, each completion will potentially be used for $K-1$ separate updates. The model tends to overfit after a single epoch. In stead, we train on all $K \choose 2$ comparisons from each prompt as a single batch element. 
Specifically, 

$\displaystyle loss(r_\theta) = - \frac{1}{K \choose 2}E_{(x, y_w, y_l)\sim D}[\log(\sigma(r_\theta(x, y_w) - r_\theta(x, y_l)))]$

### Policy model

Here we denote policy model $\pi_\phi$, pretrained + supervised finetuned model $\pi_{ref}$, ppo value function $V(x,y)$, and reward model $r_\theta$.

In \[1\], $\pi_\phi$ and $V(x,y)$ share the underlying transformer initialized from $\pi_{ref}$ with different linear headings. So the objective function combines the policy loss and value function loss, roughly in the form of:
$loss = policy\\_loss + c \cdot value\\_loss$. where $c$ is a coefficient like 0.1.
Details please see the [PPO paper](https://arxiv.org/abs/1707.06347) and \[1\]. 

The RL trajectory starts at the generation of the first token in $y$. 

In order to compute policy loss and value loss, we need to know the per step reward $r_t$:

when t is the last step, 

$\displaystyle r_t = r_\theta(x, y) -\beta\log\frac{\pi_\phi(y|x)}{\pi_{ref}(y|x)}$

otherwise, 

$\displaystyle r_t = -\beta\log\frac{\pi_\phi(y|x)}{\pi_{ref}(y|x)}$.

i.e. the reward $r_\theta(x, y)$ is only for the entire response $y$. $\beta$ is a KL reward coefficient.

The KL term prevents $\pi_\phi$ from deviating away from $\pi_{ref}$, and also acts as an entropy bonus because when we take derivative wrt. $\phi$, $\pi_{ref}(y|x)$ is a constant.

In [2],  compared with [1], the ppo value function $V(x, y)$ is an independent model which does not share parameters with $\pi_\phi$, and is initialized from $r_\theta$. The paper says "this prevents updates to the value function from partially destroying the pretrained policy early in the training".
It looks like in the experiments carried by the authors, combined loss objective in [1] did not work as well as an independent model.

The InstructGPT in [3] continues to use an independent model for value function which is also initialized from the trained reward model. 

Different from previous works in [1] and [2], InstructGPT combines a pretrained loss term:

$\displaystyle objective(\phi) = E_{(x,y) \sim D_{\pi_\phi}}[r_\theta(x, y) -\beta\log\frac{\pi_\phi(y|x)}{\pi_{ref}(y|x)}] + \gamma E_{x \sim D_{pretrain}}[\log\pi_\phi(x)]$

where $\gamma$ is the pretraining loss coefficient.

Question, I don't see in the paper on how the pretrained loss term is combined with the first loss term. Maybe back extends the start of RL trajectory to the generation of first token in $x$, like $\log\pi_\phi(x)$ is a per step loss for every token of $x$? Or simply add the $\log\pi_\phi(x)$ of the entire $x$ onto the loss?

## Direct Preference Optimization

The optimal distribution $\pi^\*$ to maximize

$E_{s \sim \pi(s)}[f(s)]$  such that $D_{KL}(\pi||\pi_{ref}) < \epsilon$, 

where $D_{KL}(\pi||\pi_{ref})$ is the KL-divergence between $\pi$ and a reference distribution $\pi_{ref}$, is: 

$\displaystyle \pi^* = \frac{1}{Z}\pi_{ref} \cdot\exp(\frac{1}{\beta}f(s))$

where $\beta$ is the Lagrange multiplier, and $Z$ is a normalizing constant wrt. $s$.

In our case, $f(s)$ is the reward $r(x, y)$, $\pi_{ref} = \pi_{ref}(y|x)$, $\pi = \pi_\phi(y|x)$, and $Z = Z(x)$.
Rearrange the equation above to have:

$\displaystyle r(x, y) = \beta\log\frac{\pi_\phi(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)$

Replace $r_\theta(x, y)$ in the reward loss objective (A) above, we have a function containing only $\pi_\phi$ and $\pi_{ref}$:

$L_{DPO}(\pi_\phi; \pi_{ref}) = -E_{(x, y_w, y_l) \sim D}[\log(\sigma(\beta\log\frac{\pi_\phi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\phi(y_l|x)}{\pi_{ref}(y_l|x)}))]$

As a result, by doing gradient descent wrt. $\phi$ on $L_{DPO}(\pi_\phi; \pi_{ref})$, we can optimize $\pi_\phi$ without training a reward model.

The log probs are computed in the same way as PPO. For example, we concatenate $(x, y_w)$ and run it through the policy model to get $\log\pi_\phi(y_w|x)$. 

And that's it. NO RL training any more.

The PPO training requires to run 4 models, reward, policy, policy reference, and state value function model. DPO needs only 2, policy and policy reference. As each of the model is a gigantic transformer, DPO is much more resource efficient.
