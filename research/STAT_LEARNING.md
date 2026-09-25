# Statistical learning and CivicQ: what applies, and how

*A research note written alongside Round 15. It maps graduate-level ideas from statistical learning and mathematical statistics (Hastie, Tibshirani & Friedman's* Elements of Statistical Learning*; Wasserman's* All of Statistics*; van der Vaart's* Asymptotic Statistics*; the recent literature cited inline) onto CivicQ's open questions. Each section gives the mapping, the math that makes it concrete, and what it predicts or would change. Sections are ranked by how well they fit.*

The one-line summary: CivicQ's recent rounds are not missing machine learning models. The twin display is already an exact posterior, and 12 offices are too few to train anything. What they are missing is **decision theory, learning theory and inference theory**. These are the parts of statistics that say what to do with a posterior, how fast a log teaches you, and when a law found on a grid can be trusted.

---

## 1. A display is a classifier acting on a posterior (Bayes decision theory) — *Round 15*

**The mapping.** Label each arrival Y = 1{V > T}, "would be served late". A display that says "over 15" or nothing is a binary classifier h(X) of Y, and a ticket office sees only X (ticket ages, elapsed services, the hour). Rounds 12–14 compared classifiers by their downstream effect. Decision theory says the effect is not the classification error.

**What the posterior already is.** The twin display (Round 13b) samples a holder's presence as 1 − G(age) and replays the queue with the true schedule. Patience is i.i.d. and FIFO means later arrivals never matter, so its samples are draws from the exact posterior P(V ∈ · | X). A better predictor cannot beat it on calibration. E18a found it calibrated to within 0.006 in 11 of 12 offices. So Round 13b's "information limit" combined two losses: information the office does not have, and the *rule* used to act on the posterior (a quantile at T, tuned after the fact).

**The Bayes rule.** A display that does not see V still admits a share a(x) of arrivals at offered wait x. In Round 13's exact law this gives u(x) = a(x)(1 − G(x)), so every outcome is a functional of a. Let ψ(x) be the influence function, the change in failures per extra arrival turned away at offered wait x (von Mises calculus, as in semiparametric statistics). Then

  dF = E[ E[ψ_a(V) | X] · dπ(X) ]  ⇒  flag X iff E[ψ_a(V) | X] < 0,

a first-order condition at a fixed point in a. With full information and a = the cutoff, Round 13's theorem gives sign ψ = sign(x − T): the theorem is the special case. The twin's quantile rule is the special case where ψ is a step. A Bayes display with a step ψ reproduced the twin at q = 31/64 byte for byte.

**What it predicts.** ψ is smooth, not a step. Under the cutoff base, turning away a citizen who would just make the target costs only 0.01–0.31 failures in the E16 offices' busy hours, because their window goes to someone the cutoff would otherwise send home. Turning away one who would just miss it saves 0.34–0.89. The implied threshold is P(late | X) ≈ ψ_on/(ψ_on + |ψ_late|) ≈ 0.1–0.4. Round 13b found 0.2–0.3 by tuning.

**With returns.** For a mandatory service the loss is minutes plus K per wasted trip. Linearized around a steady state (day-map slope s, m_R minutes per extra returner):

  ψ_K(x) = Δlost(x) + (K + m_R)·Δlosses(x)/(1 − s).

The factor 1/(1 − s) is Round 9's critical slowing down, now entering a decision rule.

**Precision.** Each would-be-served citizen turned away who would have been on time buys a repeat visit and saves no late service. So Round 14's exchange rate should scale like ε_oracle/precision. That is a one-number explanation of why realistic displays pay about three times as much, and it is tested as H66.

**What Round 15 found (REPORT §5.19).**
- The one-step Bayes rule, with no tuning, was not significantly worse than the hindsight-tuned twin in 11 of 12 offices, and both recovered a median 23% of the oracle's gain.
- The twin was calibrated in 11 of 12 offices, and its AUC beat a head count's by at most 0.02. The limit is information, not the rule.
- Refitting ψ on the display's own log overshot and oscillated (this is §2's instability, observed).
- ψ_K linearized around the oracle turned away mostly on-time citizens, because around the oracle the swap makes that look free. A first-order rule is only as good as the point it is linearized at.
- Precision predicted the price in visits with Spearman 0.89, but with slope 0.45, not 1: turned-away on-time citizens also free windows.

---

## 2. Learning from data you generate yourself: performative prediction, selective labels, Berk–Nash equilibrium

**The mapping.** Two of CivicQ's loops are *performative*: the model changes the distribution it is later refitted on.
- *Round 11:* plan → ticket log → fitted patience → plan. A safely staffed office sees only short waits.
- *Any learned display:* the display turns people away, which shortens the waits it predicts. Anyone it turns away never gets a ticket, so their V is never recorded (the selective labels problem; Lakkaraju et al. 2017).

**The theory.**
- *Performative prediction* (Perdomo, Zrnic, Mendler-Dünner & Hardt 2020). Repeated risk minimization θ_{t+1} = argmin E_{D(θ_t)} ℓ converges to a *performatively stable* point when the sensitivity ε of D(θ) is below γ/β (strong convexity over smoothness). A stable point need not be optimal.
- *Berk–Nash equilibrium* (Esponda & Pouzo 2016) is the misspecified version. The agent's model minimizes KL divergence to the data its own action produces. Round 11's rule A, an exponential fitted to a well-staffed log, is exactly a Berk–Nash equilibrium: E14b predicted its stopping point (3, 6, 11 h above the oracle) and E14c found 3.1, 6.1 and 10.4. Naming it gives the literature's tools: when such equilibria exist, when they are unique, and why exploration moves them only partly.

**What it would change.**
1. An office that calibrates a head-count display from its own log fits on admitted citizens only. Randomizing the display off on a small share of days gives unbiased labels. Inverse-propensity weighting or *weighted conformal prediction* (Tibshirani, Barber, Candès & Ramdas 2019) then corrects the fit for the policy shift.
2. B* in Round 15 is repeated risk minimization on ψ. It did not settle: the number of citizens turned away jumped 1.7–5.7-fold at the first refit and swung back at the next. Damped updates (averaging tables, or a step size below 1) are the standard fix and the obvious next test.

**Candidate round.** "Calibrating a display from its own log": the stable point versus the optimum, and the price of randomized exploration in citizens per unit of bias removed.

---

## 3. How fast a ticket log teaches: current-status asymptotics — *a closed-form Round 10*

**The mapping.** Round 10 showed that the ticket log is current-status data: V is the inspection time and "called, nobody came" is the indicator 1{τ < V}. The NPMLE then converges at n^(−1/3) (Groeneboom & Wellner 1992). E13c measured the rate. The *constant* is known too:

  n^{1/3}(Ĝ(t) − G(t)) →d (4 G(t)(1 − G(t)) g(t) / h_V(t))^{1/3} · Z,  Z = argmin_s (W(s) + s²),

with h_V the density of inspection times (the office's own waits) at t and sd(Z) ≈ 0.513 (Chernoff's distribution).

**Check (post hoc, on E13's cached logs).** Plugging in each log's tickets per day and h_V(15), with no fitted constant:

| Days | Office lean: predicted / observed | 8 E citizen: predicted / observed |
|---|---|---|
| 10 | 0.062 / 0.068 | 0.066 / 0.089 |
| 30 | 0.043 / 0.045 | 0.045 / 0.049 |
| 100 | 0.029 / 0.031 | 0.030 / 0.031 |
| 300 | 0.020 / 0.020 | 0.021 / 0.021 |
| 1000 | 0.013 / 0.012 | 0.014 / 0.015 |

From 30 days on the formula is within 10%. It misses only the 8 E office's first 10 days, where 15 minutes sits at the edge of the observed waits.

**What it explains.** "The more waiting an office prevents, the less it learns" (H39–H40) is the h_V(T) in the denominator. Days needed to reach RMSE δ scale like G(1 − G)g / (h_V(T) · tickets per day · δ³).

**What it would change.**
- *Planning.* An office can compute the days it needs before collecting any data.
- *Exploration as optimal design.* Round 11 explored with 90% staffing days. The design question is which exploration maximizes h_V(T) per extra failure. It may be cheaper to lengthen waits only in the hours where V is near T.
- *Shape constraints.* A log-concave patience density lifts the rate to n^(−2/5) (Balabdaoui, Rufibach & Wellner 2009). Lognormal and exponential patience are both log-concave, so this is free information the NPMLE ignores.
- *Partial identification.* "Mean patience is not identified" (§5.14) can be made sharp. Manski-style bounds on the mean follow from G on the observed range plus any tail assumption.

---

## 4. Finding laws without being fooled by the grid: dimensional analysis, active subspaces, extrapolation splits

**The mapping.** H60 was registered from probes at c = 4 and 16 and broke at c = 1–2. The governing group n_T = cT/S was found only after the fact. This is covariate shift in the space of *model inputs*: a law fitted in one region was used outside it.

**Tools.**
- *Buckingham Π.* The stationary model's inputs (c, λ, S, T, patience mean m and CV) reduce to dimensionless groups c, ρ, T/S, m/S and CV. Every law should be stated in them from the start.
- *Active subspaces* (Constantine 2015). For a cheap, smooth model f (the exact law costs milliseconds), the eigenvectors of C = E[∇f ∇fᵀ] over log-inputs give the directions f actually varies along. If log ε depends on c and T/S only through log c + log(T/S) = log n_T, the top eigenvector is ∝ (1, 1, 0, …). The method would have found n_T before registration, with the spectral gap saying how good a one-variable law is (the 1.9× residual spread in E17e).
- *GP surrogates with sequential design.* A Gaussian process on the theory's output, sampled where its posterior variance is largest, finds the regions where a law fails.
- *Extrapolation splits.* Before registering a law, fit it on the interior of the grid and test it on the corners (smallest and largest c, extreme ρ). Random splits reward interpolation; region splits test what a law is for.

**Candidate check.** Run active subspaces on log ε from `display_returns.exchange_rate` and on the Round 6 crossover slack. Does it rediscover n_T = cT/S and (S/2T)·ln c without being told?

---

## 5. Inference hygiene for many-cell claims

- *Equivalence tests.* "No significant rise" (H59c, H63c) is absence of evidence. A two one-sided test (TOST; Schuirmann 1987) with a registered margin δ turns it into evidence of absence: "the rise is below δ".
- *Multiplicity.* Claims like "significant in 10 of 12" are family-wise statements. Benjamini–Hochberg control of the false discovery rate, or a registered Holm correction, states what they guarantee.
- *Partial pooling.* Per-hour late rates across 24 settings are natural candidates for hierarchical shrinkage (empirical Bayes, James–Stein). Hours with few arrivals borrow strength, and "significant misses" in small hours become rarer and more trustworthy.
- *Clustered calibration.* Citizens within a day are dependent. Calibration and AUC standard errors should resample days, as the rest of the protocol does.

---

## 6. Certified displays: conformal risk control

A display could carry a guarantee that holds whatever the model: "of the citizens told 'over 15' who would have stayed, at most δ would have been served on time". *Conformal risk control* (Angelopoulos, Bates, Fisch, Lei & Schuster 2022) picks the threshold on any score (head count, twin P(late)) to bound such a monotone loss in expectation, from a calibration log. With §1's precision law, a certified precision is a certified bound on the repeat-visit price ε. The catch is §2: calibration labels exist only for admitted citizens, so the calibration log must come from hidden-queue days or randomized ones, with weighted conformal for the shift.

---

## 7. Simulation optimization (lower priority)

SGS is greedy local search with common random numbers. The statistical versions are ranking-and-selection with indifference zones, stochastic kriging (Ankenman, Nelson & Staum 2010), and Bayesian optimization over rosters. They would put numbers on "SGS optimality is shown only for small instances" (§6). The gains are probably modest: SGS already matches exhaustive search where that is feasible.

---

## 8. What does not apply

- **Flexible predictors for V.** Gradient boosting or neural networks on (ticket ages, count, hour) cannot beat an exact posterior on calibration. They could only help where the model is wrong (misspecified patience), which is §2's problem, not a prediction problem.
- **Learning staffing rules end to end.** With 12–24 offices, a learned rule would memorize the design. Residual learning on top of Lag-SIPP is possible, but the ideas that generalize here have come from theory, not from fitting.
- **Reinforcement learning for displays.** The display is an admission-control problem (Naor 1969) with partial observation. Section 1's first-order condition is the tractable core of it. A full POMDP solution is not needed to get the structure, and the office could not verify it anyway.

---

## References (new here)
- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2022). Conformal risk control. arXiv:2208.02814.
- Ankenman, B., Nelson, B. L., & Staum, J. (2010). Stochastic kriging for simulation metamodeling. *Operations Research*, 58(2), 371–382.
- Balabdaoui, F., Rufibach, K., & Wellner, J. A. (2009). Limit distribution theory for maximum likelihood estimation of a log-concave density. *Annals of Statistics*, 37(3), 1299–1331.
- Constantine, P. G. (2015). *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter Studies*. SIAM.
- Esponda, I., & Pouzo, D. (2016). Berk–Nash equilibrium: A framework for modeling agents with misspecified models. *Econometrica*, 84(3), 1093–1130.
- Groeneboom, P., & Wellner, J. A. (1992). *Information Bounds and Nonparametric Maximum Likelihood Estimation*. Birkhäuser.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Lakkaraju, H., Kleinberg, J., Leskovec, J., Ludwig, J., & Mullainathan, S. (2017). The selective labels problem. *KDD*, 275–284.
- Perdomo, J., Zrnic, T., Mendler-Dünner, C., & Hardt, M. (2020). Performative prediction. *ICML*, 7599–7609.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *J. Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.
- Tibshirani, R. J., Barber, R. F., Candès, E., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
- van der Vaart, A. W. (1998). *Asymptotic Statistics*. Cambridge University Press.
