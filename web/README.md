# CivicQ Visualizer

Interactive website for the CivicQ staffing simulator. The discrete-event simulator is ported from `cpp/src/simulation.cpp` to TypeScript and runs in a Web Worker, so every control updates in the browser without a backend. The design came from a Claude Design prompt, saved in [DESIGN_PROMPT.md](DESIGN_PROMPT.md).

```bash
cd web
npm install
npm run dev      # http://localhost:5173
npm test         # simulator + Erlang-C tests; cross-checks against cpp/build/queue_sim if it is built
npm run build    # static site in web/dist (relative paths, works on GitHub Pages)
```

## Layout

| Path | What it is |
|---|---|
| `src/sim/simulate.ts` | One simulated day (port of `QueueSimulator`), with common random numbers |
| `src/sim/stats.ts` | Replications, t and ratio confidence intervals, histogram and queue curve |
| `src/sim/analytic.ts` | Erlang-C, SIPP, Lag-SIPP, offered load m(t) (ports of `python/optimizer.py` and `research/staffing_methods.py`) |
| `src/sim/worker.ts` | Background runs: current plan, plan comparison, cost-vs-service frontier search |
| `src/sections/*` | One component per page section |
| `src/state.ts` | App state, mirrored into the URL so a view can be shared |

## Validation (`src/sim/sim.test.ts`)

- Erlang-C matches the known values in `python/test_validation.py`.
- Under constant load, the simulated mean wait falls inside the 95% CI of the exact M/M/c value.
- For the default plan, per-hour late shares agree with `research/staffing_methods.evaluate`.
- Mean and P90 wait agree with the C++ executable for four plans, when it is built.
