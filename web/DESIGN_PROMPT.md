# CivicQ Visualizer — design prompt

**Build an interactive visual explainer + sandbox for "CivicQ", a staffing simulator for a walk-in government service office (a permit/DMV-style counter).**

**Audience and goal:** The audience is me, the author. I'm technical but I want to *see* the queueing math instead of reading CSVs. The page should work as a visual workbench: I change the staffing and demand, and the waits, the queue, and the cost update immediately.

**Style:** Clean and quiet, somewhere between Linear and Distill.pub. Use green-tinted neutrals with a single accent: forest green in light mode, mint on deep green in dark mode, plus one warning color for "target missed". Set text in Inter or the system UI font, and put numbers in a monospace font with tabular figures. Keep motion subtle, around 150–200ms, and use no gradients, glass effects, or illustrations. **Light and dark mode** come from CSS variables, with a toggle in the header that follows the system setting by default. The layout must work down to phone width.

**The model (use these real values):**
- 8 hourly slots over an 8-hour day (480 min). Arrivals follow a non-homogeneous Poisson process with λ per hour = **[12, 15, 10, 8, 8, 12, 14, 10]**.
- Service is exponential with a mean of **8 min**. Lognormal (CV 0.5–1.5) and deterministic are also options.
- A single FIFO queue feeds *c* identical windows. Staffing per hour sᵢ ∈ {1…6}, and the default plan is **[2, 3, 3, 2, 2, 3, 3, 3]** (21 staff-hours).
- The office opens empty and the doors close at 480 min, but everyone already inside is served. The extra time is shown as *overtime*.
- Service target: P90 wait ≤ **15 min**, i.e. at most 10% of each hour's arrivals wait more than 15 min.
- Cost = w₁·(mean wait) + w₂·(staff-hours), with defaults w₁ = 1 and w₂ = 0.5.

**Sections (one scrolling page with a sticky left nav):**
1. **Live queue.** An animated strip view shows citizens as dots arriving, waiting in the queue, and moving to window boxes, with a clock scrubber and play/pause/speed controls (1×–60×). Beside it are live counters for queue length, busy windows, and current wait.
2. **Staffing plan editor.** Eight vertical steppers or draggable bars set windows per hour. Show the arrival rate λ per hour and the offered load λ/μ as a line over the bars. Hours where the load is at or above the number of windows are flagged as unstable. The total staff-hours and cost are pinned at the top.
3. **Wait-time results.** Show three charts: (a) a histogram of wait times with the mean and P90 marked and the 15-min target line, (b) a per-hour strip showing the % of arrivals waiting over 15 min, colored by pass/fail against 10%, and (c) utilization per hour. Draw 95% CI bands from N replications, with N adjustable from 10 to 300.
4. **The math, explained.** Short prose with scrubbable inline numbers (in the style of Bret Victor) covering:
   - the Erlang-C formula for P(wait), next to the simulated value in the same card;
   - why SIPP (treating each hour as steady state) is off: a queue-length-vs-time chart overlays the SIPP prediction on the simulation and shows congestion lagging demand;
   - offered load m(t) as a smooth curve over the hourly λ steps.
5. **Compare plans.** Choose two to four plans (SIPP, Lag-SIPP, OL-max, simulation-optimal, or custom). Show them side by side as small multiples with staff-hours, P90, and worst hour. Plans are compared using common random numbers, and a short note explains why that matters.
6. **Cost vs. service frontier.** A scatter plot of staff-hours against P90 for every feasible plan, with the Pareto frontier highlighted. Clicking a point loads that plan into the editor.
7. **Research findings.** Four compact cards with small charts:
   - SIPP overstaffs by +6.7% on average, up to 14.5%.
   - Lag correction recovers about half of that excess when service is 8 min or longer.
   - The cost of demand uncertainty grows with office size: +8% at 2 windows vs. +22% at 24.
   - Changing only the service-level definition moves the required staffing from 18 to 22 staff-hours.

**Global controls (right rail, collapsible):** mean service time, service distribution and CV, a demand multiplier (0.8–1.4×), demand CV, the wait threshold, α, the random seed, and the replication count. A "Reset to defaults" button and a shareable URL encode the full state.

**Interaction rules:**
- Every input updates the charts in under 100ms. Heavy runs show a thin progress bar and never a blocking spinner.
- Hovering a chart shows exact values in a tooltip.
- Keyboard: ←/→ changes the focused bar, and Space plays or pauses the animation.
- Number changes animate with a short tween.

**Deliver:** A high-fidelity design of the full page in both light and dark mode, plus the mobile layout for sections 2 and 3. Use realistic numbers from the model above, never lorem ipsum.

