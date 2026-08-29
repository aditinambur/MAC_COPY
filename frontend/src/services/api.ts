import { Run, RunConfig, RepairConfig, SystemInfo, MetricRow, CheckpointInfo } from '../types';

const API_BASE = '/api';

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const res = await fetch(`${API_BASE}/system/info`);
  if (!res.ok) throw new Error('Failed to fetch system info');
  return res.json();
}

export async function fetchRuns(): Promise<Run[]> {
  const res = await fetch(`${API_BASE}/runs`);
  if (!res.ok) throw new Error('Failed to fetch runs');
  return res.json();
}

export async function startTraining(config: RunConfig): Promise<{ run_id: string; log_file: string }> {
  const res = await fetch(`${API_BASE}/runs/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to start run' }));
    throw new Error(err.detail || 'Failed to start run');
  }
  return res.json();
}

export async function stopCurrentProcess(): Promise<{ run_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/runs/stop`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to stop process' }));
    throw new Error(err.detail || 'Failed to stop process');
  }
  return res.json();
}

export async function fetchRunLogs(runId: string, lines = 200): Promise<string> {
  const res = await fetch(`${API_BASE}/runs/${runId}/logs?lines=${lines}`);
  if (!res.ok) throw new Error('Failed to fetch logs');
  const data = await res.json();
  return data.logs || '';
}

export async function fetchRunMetrics(runId: string): Promise<MetricRow[]> {
  const res = await fetch(`${API_BASE}/runs/${runId}/metrics`);
  if (!res.ok) throw new Error('Failed to fetch metrics');
  const data = await res.json();
  return data.metrics || [];
}

export async function fetchRunCheckpoints(runId: string): Promise<CheckpointInfo[]> {
  const res = await fetch(`${API_BASE}/runs/${runId}/checkpoints`);
  if (!res.ok) throw new Error('Failed to fetch checkpoints');
  const data = await res.json();
  return data.checkpoints || [];
}

export async function triggerRenderGif(runId: string): Promise<{ status: string; gif_path: string }> {
  const res = await fetch(`${API_BASE}/runs/${runId}/render`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Rendering failed' }));
    throw new Error(err.detail || 'Rendering failed');
  }
  return res.json();
}

export function getRunGifUrl(runId: string): string {
  return `${API_BASE}/runs/${runId}/gif?t=${Date.now()}`;
}

export async function startCausalRepair(
  runId: string,
  config: RepairConfig
): Promise<{ run_id: string; repair_log: string }> {
  const res = await fetch(`${API_BASE}/runs/${runId}/repair`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to launch repair' }));
    throw new Error(err.detail || 'Failed to launch repair');
  }
  return res.json();
}

export async function fetchRepairLogs(runId: string, lines = 200): Promise<string> {
  const res = await fetch(`${API_BASE}/runs/${runId}/repair_logs?lines=${lines}`);
  if (!res.ok) throw new Error('Failed to fetch repair logs');
  const data = await res.json();
  return data.logs || '';
}

export async function toggleArchiveRun(runId: string): Promise<{ archived: boolean }> {
  const res = await fetch(`${API_BASE}/runs/${runId}/archive`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to toggle archive status');
  return res.json();
}

export async function deleteRun(runId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${runId}`, { method: 'DELETE' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to delete run' }));
    throw new Error(err.detail || 'Failed to delete run');
  }
}
