import { useEffect, useRef, useState } from 'react';
import type { SimConfig } from './sim/model';
import type { Aggregate } from './sim/stats';
import type { FrontierPoint, WorkerRequest, WorkerResponse } from './sim/worker';

type Kind = WorkerRequest['kind'];
type Listener = (m: WorkerResponse) => void;

let worker: Worker | null = null;
const listeners = new Set<Listener>();
let nextId = 1;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./sim/worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => listeners.forEach((l) => l(e.data));
  }
  return worker;
}

type Payload<K extends Kind> = Omit<Extract<WorkerRequest, { kind: K }>, 'kind' | 'id'>;

/** Sends a request whenever `key` changes; returns the latest result and progress. */
function useWorkerTask<K extends Kind, R>(
  kind: K,
  payload: Payload<K> | null,
  key: string,
  pick: (m: WorkerResponse) => R | undefined,
  debounceMs = 120,
) {
  const [result, setResult] = useState<R | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const idRef = useRef(0);
  const pickRef = useRef(pick);
  pickRef.current = pick;

  useEffect(() => {
    const l: Listener = (m) => {
      if (m.kind !== kind || m.id !== idRef.current) return;
      if ('progress' in m) setProgress(m.progress);
      else {
        const r = pickRef.current(m);
        if (r !== undefined) setResult(r);
        setProgress(null);
      }
    };
    listeners.add(l);
    return () => void listeners.delete(l);
  }, [kind]);

  useEffect(() => {
    if (!payload) return;
    const t = setTimeout(() => {
      idRef.current = nextId++;
      setProgress(0);
      getWorker().postMessage({ kind, id: idRef.current, ...payload } as WorkerRequest);
    }, debounceMs);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return { result, progress };
}

export function useRun(cfg: SimConfig, days: number, seed: number) {
  const payload = { cfg, days, seed };
  return useWorkerTask('run', payload, JSON.stringify(payload), (m) => ('result' in m ? m.result : undefined));
}

export function useCompare(cfg: SimConfig, plans: number[][], days: number, seed: number) {
  const payload = { cfg, plans, days, seed };
  return useWorkerTask<'compare', Aggregate[]>('compare', payload, JSON.stringify(payload), (m) =>
    'results' in m ? m.results : undefined,
  );
}

/** Runs only when `token` changes (a button press), using the settings at that moment. */
export function useFrontier(cfg: SimConfig, seed: number, token: number) {
  const payload = token > 0 ? { cfg: { ...cfg, plan: [] }, seed } : null;
  return useWorkerTask<'frontier', FrontierPoint[]>('frontier', payload, String(token), (m) =>
    'points' in m ? m.points : undefined,
    0,
  );
}
