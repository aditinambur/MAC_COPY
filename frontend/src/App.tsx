import { useState, useEffect, useRef } from 'react';
import { 
  Home,
  Sliders, 
  Terminal,
  Play, 
  Square, 
  RefreshCw, 
  Eye, 
  AlertCircle,
  Database,
  ArrowUpRight,
  Settings,
  Plus,
  Activity,
  Trash2,
  Archive
} from 'lucide-react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';

interface Run {
  run_id: string;
  experiment_name: string;
  run_name: string;
  path: string;
  status: string;
  config: Record<string, any>;
  has_metrics: boolean;
  has_gif: boolean;
  archived: boolean;
}

interface MetricRow {
  total_num_steps: number;
  causal_influence_kl_mean?: number;
  causal_influence_value_sensitivity_mean?: number;
  [key: string]: any;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'config' | 'logs'>('dashboard');
  const [runs, setRuns] = useState<Run[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [logs, setLogs] = useState<string>('');
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [gifUrl, setGifUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isRendering, setIsRendering] = useState<boolean>(false);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [showArchived, setShowArchived] = useState<boolean>(false);
  
  // Config form state
  const [configForm, setConfigForm] = useState({
    experiment_name: 'check',
    seed: 1,
    num_env_steps: 100000,
    episode_length: 25,
    n_rollout_threads: 32,
    eval_interval: 5,
    disable_messages: false,
    eval_disable_messages: false,
    eval_noise_std: 0.25,
    use_eval: true
  });

  const logConsoleRef = useRef<HTMLDivElement>(null);

  // Poll for runs and active process
  useEffect(() => {
    fetchRuns();
    checkActiveRun();
    
    const interval = setInterval(() => {
      fetchRuns();
      checkActiveRun();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // Poll logs if there's an active run
  useEffect(() => {
    let logInterval: any;
    if (activeRunId) {
      fetchLogs(activeRunId);
      logInterval = setInterval(() => {
        fetchLogs(activeRunId);
      }, 2000);
    }
    return () => clearInterval(logInterval);
  }, [activeRunId]);

  // Auto-scroll log console
  useEffect(() => {
    if (logConsoleRef.current) {
      logConsoleRef.current.scrollTop = logConsoleRef.current.scrollHeight;
    }
  }, [logs]);

  // Fetch metrics when selectedRun changes
  useEffect(() => {
    if (selectedRun) {
      fetchMetrics(selectedRun.run_id);
      if (selectedRun.has_gif) {
        setGifUrl(`/api/runs/${selectedRun.run_id}/gif?t=${Date.now()}`);
      } else {
        setGifUrl(null);
      }
    }
  }, [selectedRun]);

  // Set initial selected run once runs list is loaded
  useEffect(() => {
    if (runs.length > 0 && !selectedRun) {
      setSelectedRun(runs[0]);
    }
  }, [runs]);

  const fetchRuns = async () => {
    try {
      const res = await fetch('/api/runs');
      if (res.ok) {
        const data = await res.json();
        setRuns(data);
      }
    } catch (err) {
      console.error("Error fetching runs:", err);
    }
  };

  const checkActiveRun = async () => {
    try {
      const res = await fetch('/api/runs/active');
      if (res.ok) {
        const data = await res.json();
        setActiveRunId(data.run_id);
      }
    } catch (err) {
      console.error("Error checking active run:", err);
    }
  };

  const fetchLogs = async (runId: string) => {
    try {
      const res = await fetch(`/api/runs/${runId}/logs`);
      if (res.ok) {
        const data = await res.json();
        setLogs(data.logs);
      }
    } catch (err) {
      console.error("Error fetching logs:", err);
    }
  };

  const fetchMetrics = async (runId: string) => {
    setIsLoading(true);
    try {
      const res = await fetch(`/api/runs/${runId}/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data.metrics);
      }
    } catch (err) {
      console.error("Error fetching metrics:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStartRun = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setErrorMsg(null);
    try {
      const res = await fetch('/api/runs/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configForm)
      });
      const data = await res.json();
      if (res.ok) {
        setActiveRunId(data.run_id);
        setActiveTab('logs');
        setLogs('[Spawning training subprocess...]\n');
      } else {
        setErrorMsg(data.detail || "Failed to start run.");
      }
    } catch (err) {
      setErrorMsg("Failed to connect to API server.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleStopRun = async () => {
    if (!window.confirm("Are you sure you want to stop the current training process early?")) return;
    try {
      const res = await fetch('/api/runs/stop', { method: 'POST' });
      if (res.ok) {
        setActiveRunId(null);
        fetchRuns();
      }
    } catch (err) {
      console.error("Error stopping run:", err);
    }
  };

  const handleGenerateGif = async () => {
    if (!selectedRun) return;
    setIsRendering(true);
    setRenderError(null);
    try {
      const res = await fetch(`/api/runs/${selectedRun.run_id}/render`, {
        method: 'POST'
      });
      const data = await res.json();
      if (res.ok) {
        setGifUrl(`/api/runs/${selectedRun.run_id}/gif?t=${Date.now()}`);
        fetchRuns();
      } else {
        setRenderError(data.detail || "Failed to generate GIF.");
      }
    } catch (err) {
      setRenderError("Failed to connect to rendering API.");
    } finally {
      setIsRendering(false);
    }
  };

  const handleToggleArchive = async (runId: string) => {
    try {
      const res = await fetch(`/api/runs/${runId}/archive`, { method: 'POST' });
      if (res.ok) {
        fetchRuns();
        if (selectedRun?.run_id === runId && !showArchived) {
          setSelectedRun(null);
        }
      }
    } catch (err) {
      console.error("Failed to toggle archive:", err);
    }
  };

  const handleDeleteRun = async (runId: string) => {
    if (!window.confirm("Are you sure you want to permanently delete this run and all its metrics/logs?")) return;
    try {
      const res = await fetch(`/api/runs/${runId}`, { method: 'DELETE' });
      if (res.ok) {
        fetchRuns();
        if (selectedRun?.run_id === runId) {
          setSelectedRun(null);
        }
      } else {
        const data = await res.json();
        alert(data.detail || "Failed to delete run.");
      }
    } catch (err) {
      console.error("Failed to delete run:", err);
    }
  };

  // Compute latest metric figures for cards
  const latestKL = metrics.length > 0 ? metrics[metrics.length - 1].causal_influence_kl_mean?.toFixed(4) : "0.0000";
  const latestVS = metrics.length > 0 ? metrics[metrics.length - 1].causal_influence_value_sensitivity_mean?.toFixed(4) : "0.0000";

  return (
    <div className="layout-container">
      {/* 1. Slim Left Sidebar (Royal Indigo theme) */}
      <aside className="sidebar">
        <div className="sidebar-top">
          <div className="sidebar-logo">M</div>
          <nav className="sidebar-nav">
            <button
              onClick={() => setActiveTab('dashboard')}
              title="Dashboard"
              className={`sidebar-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
            >
              <Home className="h-5 w-5" />
            </button>
            <button
              onClick={() => setActiveTab('config')}
              title="Run Configurator"
              className={`sidebar-btn ${activeTab === 'config' ? 'active' : ''}`}
            >
              <Sliders className="h-5 w-5" />
            </button>
            <button
              onClick={() => setActiveTab('logs')}
              title="Console Logs"
              className={`sidebar-btn ${activeTab === 'logs' ? 'active' : ''}`}
            >
              <Terminal className="h-5 w-5" />
            </button>
          </nav>
        </div>

        <div className="sidebar-footer">
          <button className="sidebar-btn">
            <Settings className="h-5 w-5" />
          </button>
          <div className="avatar">UA</div>
        </div>
      </aside>

      {/* 2. Content Area Wrapper */}
      <div className="content-wrapper">
        
        {/* Top Page Header */}
        <header className="page-header">
          <div className="page-header-title">
            <h1>
              {activeTab === 'dashboard' && "Analytics Dashboard"}
              {activeTab === 'config' && "Configuration Manager"}
              {activeTab === 'logs' && "Subprocess Console Logs"}
            </h1>
            <p>
              {activeTab === 'dashboard' && "Monitor emergent language communication & agent training curves."}
              {activeTab === 'config' && "Deploy customizable MAPPO emergent communication subprocesses."}
              {activeTab === 'logs' && "Stream unbuffered STDOUT/STDERR log files of active processes."}
            </p>
          </div>

          <div className="page-header-actions">
            {activeRunId && (
              <button onClick={handleStopRun} className="btn btn-secondary">
                <Square className="h-3.5 w-3.5 fill-current text-rose-500" />
                Terminate Run
              </button>
            )}
            <button onClick={() => setActiveTab('config')} className="btn btn-primary">
              <Plus className="h-3.5 w-3.5" />
              Add New Run
            </button>
          </div>
        </header>

        {/* 3. Dashboard Body Viewport */}
        <div className="dashboard-body">
          <div className="dashboard-container">
            
            {/* VIEW 1: DASHBOARD */}
            {activeTab === 'dashboard' && (
              <>
                {/* KPI metrics row */}
                <div className="kpi-grid">
                  {/* KPI 1: Active Run */}
                  <div className="card kpi-card">
                    <div className="kpi-info">
                      <span className="kpi-label">Active Run Status</span>
                      <div className="kpi-value-row">
                        <h3 className="kpi-value">{activeRunId ? activeRunId : "System Idle"}</h3>
                      </div>
                      <p className="card-subtitle">
                        {activeRunId ? "Executing python process" : "Ready to launch"}
                      </p>
                    </div>
                    <div className="kpi-icon-box indigo">
                      <Activity className="h-6 w-6" />
                    </div>
                  </div>

                  {/* KPI 2: KL Significance */}
                  <div className="card kpi-card">
                    <div className="kpi-info">
                      <span className="kpi-label">Causal KL Significance</span>
                      <div className="kpi-value-row">
                        <h3 className="kpi-value">{latestKL}</h3>
                        <span className="kpi-badge green">
                          <ArrowUpRight className="h-3.5 w-3.5" />
                          CIC
                        </span>
                      </div>
                      <p className="card-subtitle">Mean KL divergence on message ablation</p>
                    </div>
                    <div className="kpi-icon-box emerald">
                      <Database className="h-6 w-6" />
                    </div>
                  </div>

                  {/* KPI 3: Value Sensitivity */}
                  <div className="card kpi-card">
                    <div className="kpi-info">
                      <span className="kpi-label">Critic Value Sensitivity</span>
                      <div className="kpi-value-row">
                        <h3 className="kpi-value">{latestVS}</h3>
                        <span className="kpi-badge blue">
                          <ArrowUpRight className="h-3.5 w-3.5" />
                          Value
                        </span>
                      </div>
                      <p className="card-subtitle">Critic absolute ΔV sensitivity</p>
                    </div>
                    <div className="kpi-icon-box purple">
                      <Eye className="h-6 w-6" />
                    </div>
                  </div>
                </div>

                {/* Main Graph & Agent rollout area */}
                <div className="main-panel-grid">
                  {/* Graph card */}
                  <div className="card">
                    <div className="card-header-border">
                      <div>
                        <h2 className="card-title">Emergent Communication Causal Impact Trend</h2>
                        <p className="card-subtitle">
                          Selected Run: {selectedRun ? `${selectedRun.experiment_name} (${selectedRun.run_name})` : "None"}
                        </p>
                      </div>
                      {selectedRun && (
                        <button onClick={() => fetchMetrics(selectedRun.run_id)} className="btn btn-secondary">
                          <RefreshCw className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>

                    <div className="h-[320px] w-full mt-2" style={{ height: '320px' }}>
                      {isLoading ? (
                        <div className="h-full flex items-center justify-center text-slate-400 text-xs font-medium">Loading metrics...</div>
                      ) : metrics.length === 0 ? (
                        <div className="h-full flex items-center justify-center text-slate-400 text-xs italic">
                          No metric history found. Verify that evaluation steps have completed.
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={metrics} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                            <defs>
                              <linearGradient id="colorKl" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#4f46e5" stopOpacity={0.25}/>
                                <stop offset="95%" stopColor="#4f46e5" stopOpacity={0}/>
                              </linearGradient>
                              <linearGradient id="colorVs" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#a855f7" stopOpacity={0.25}/>
                                <stop offset="95%" stopColor="#a855f7" stopOpacity={0}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                            <XAxis dataKey="total_num_steps" stroke="#94a3b8" fontSize={11} tickLine={false} />
                            <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} />
                            <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 8 }} />
                            <Area type="monotone" dataKey="causal_influence_kl_mean" name="Mean Policy KL" stroke="#4f46e5" strokeWidth={3} fillOpacity={1} fill="url(#colorKl)" />
                            <Area type="monotone" dataKey="causal_influence_value_sensitivity_mean" name="Mean Critic Value" stroke="#a855f7" strokeWidth={3} fillOpacity={1} fill="url(#colorVs)" />
                          </AreaChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                  </div>

                  {/* Agent rollout visualizer */}
                  <div className="card">
                    <div className="card-header-border">
                      <div>
                        <h2 className="card-title">Visual Agent Rollout</h2>
                        <p className="card-subtitle">Rendered behavior from evaluation passes</p>
                      </div>
                    </div>
                    <div className="rollout-box">
                      {gifUrl ? (
                        <div className="rollout-img-wrapper">
                          <img src={gifUrl} alt="Evaluation rollout" className="rollout-img" />
                          <span className="rollout-label">render.gif</span>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center gap-3 text-slate-400 text-center px-4" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                          <AlertCircle className="h-8 w-8 text-slate-300" />
                          <span className="text-xs font-semibold">No simulation GIF generated</span>
                          <p className="text-[10px] max-w-[200px]" style={{ margin: '0 0 8px 0' }}>
                            Generate a simulation video of the trained policies on a single test episode.
                          </p>
                          <button
                            onClick={handleGenerateGif}
                            disabled={isRendering || !selectedRun}
                            className="btn btn-primary"
                          >
                            {isRendering ? "Rendering..." : "Generate Rollout GIF"}
                          </button>
                          {renderError && (
                            <p className="text-[10px] text-rose-500 font-semibold" style={{ color: '#ef4444', margin: '4px 0 0 0' }}>{renderError}</p>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Runs Library table */}
                <div className="card">
                  <div className="card-header-border">
                    <div>
                      <h2 className="card-title">Your Runs Library</h2>
                      <p className="card-subtitle">List of all active, finished, and stopped subprocess logs</p>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', fontWeight: '700', color: '#64748b', cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={showArchived}
                          onChange={(e) => setShowArchived(e.target.checked)}
                          style={{ accentColor: '#4f46e5', cursor: 'pointer' }}
                        />
                        Show Archived
                      </label>
                      <button onClick={fetchRuns} className="btn btn-secondary">
                        <RefreshCw className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>

                  <div className="runs-table-wrapper">
                    <table className="runs-table">
                      <thead>
                        <tr>
                          <th>Experiment/Name</th>
                          <th>Run Folder</th>
                          <th>Status</th>
                          <th>Data Assets</th>
                          <th>Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {runs.filter(run => showArchived ? true : !run.archived).length === 0 ? (
                          <tr>
                            <td colSpan={5} className="py-6 text-center text-slate-400 italic text-xs">No {showArchived ? "" : "unarchived "}runs recorded in workspace results.</td>
                          </tr>
                        ) : (
                          runs.filter(run => showArchived ? true : !run.archived).map((run) => (
                            <tr key={run.run_id}>
                              <td>
                                <span className="font-bold text-slate-800 block" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                  {run.experiment_name}
                                  {run.archived && (
                                    <span style={{ fontSize: '9px', backgroundColor: '#f1f5f9', color: '#64748b', padding: '1px 4px', borderRadius: '4px', fontWeight: 'bold' }}>
                                      Archived
                                    </span>
                                  )}
                                </span>
                                <span className="text-[11px] text-slate-400 font-medium">Seed: {run.config?.seed || 'N/A'}</span>
                              </td>
                              <td className="text-slate-500 font-mono text-xs">{run.run_name}</td>
                              <td>
                                <span className={`status-tag ${
                                  run.status === 'running' ? 'running' :
                                  run.status === 'completed' ? 'completed' :
                                  run.status === 'stopped' ? 'stopped' :
                                  'failed'
                                }`}>
                                  <span className={`status-indicator-dot ${
                                    run.status === 'running' ? 'running' :
                                    run.status === 'completed' ? 'completed' :
                                    run.status === 'stopped' ? 'stopped' :
                                    'failed'
                                  }`}></span>
                                  {run.status}
                                </span>
                              </td>
                              <td>
                                <div className="flex items-center gap-2">
                                  {run.has_metrics && (
                                    <span className="asset-badge">
                                      <Database className="h-3 w-3" />
                                      Metrics
                                    </span>
                                  )}
                                  {run.has_gif && (
                                    <span className="asset-badge accent">
                                      <Eye className="h-3 w-3" />
                                      Rollout GIF
                                    </span>
                                  )}
                                </div>
                              </td>
                              <td style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <button
                                  onClick={() => setSelectedRun(run)}
                                  className={`btn ${selectedRun?.run_id === run.run_id ? 'btn-outline-active' : 'btn-secondary'}`}
                                >
                                  View Details
                                </button>
                                <button
                                  onClick={() => handleToggleArchive(run.run_id)}
                                  title={run.archived ? "Unarchive Run" : "Archive Run"}
                                  className="btn btn-secondary"
                                  style={{ padding: '6px 10px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}
                                >
                                  <Archive className="h-3.5 w-3.5" style={{ color: run.archived ? '#4f46e5' : '#64748b', fill: run.archived ? '#e0e7ff' : 'transparent' }} />
                                </button>
                                <button
                                  onClick={() => handleDeleteRun(run.run_id)}
                                  disabled={run.status === 'running'}
                                  title="Delete Run"
                                  className="btn btn-secondary"
                                  style={{ padding: '6px 10px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', opacity: run.status === 'running' ? 0.5 : 1 }}
                                >
                                  <Trash2 className="h-3.5 w-3.5" style={{ color: '#ef4444' }} />
                                </button>
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}

            {/* VIEW 2: RUN CONFIGURATOR FORM */}
            {activeTab === 'config' && (
              <div className="card" style={{ maxWidth: '800px', margin: '0 auto' }}>
                <div className="card-header-border">
                  <div>
                    <h2 className="card-title">Configure Subprocess Settings</h2>
                    <p className="card-subtitle">Adjust hyperparameters. Starting a run executes MAPPO in a background docker process.</p>
                  </div>
                </div>

                {errorMsg && (
                  <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-600 text-sm flex items-center gap-2 font-medium">
                    <AlertCircle className="h-5 w-5 shrink-0" />
                    {errorMsg}
                  </div>
                )}

                <form onSubmit={handleStartRun} className="config-form">
                  <div className="form-row">
                    {/* Column 1 */}
                    <div className="flex flex-col gap-4">
                      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider border-b border-slate-100 pb-2">Experiment</h3>
                      
                      <div className="form-group">
                        <label>Experiment Name</label>
                        <input
                          type="text"
                          value={configForm.experiment_name}
                          onChange={(e) => setConfigForm({...configForm, experiment_name: e.target.value})}
                          className="saas-input"
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="form-group">
                          <label>Seed</label>
                          <input
                            type="number"
                            value={configForm.seed}
                            onChange={(e) => setConfigForm({...configForm, seed: parseInt(e.target.value) || 1})}
                            className="saas-input"
                          />
                        </div>
                        <div className="form-group">
                          <label>Env Steps</label>
                          <input
                            type="number"
                            value={configForm.num_env_steps}
                            onChange={(e) => setConfigForm({...configForm, num_env_steps: parseInt(e.target.value) || 10000})}
                            className="saas-input"
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="form-group">
                          <label>Episode Length</label>
                          <input
                            type="number"
                            value={configForm.episode_length}
                            onChange={(e) => setConfigForm({...configForm, episode_length: parseInt(e.target.value) || 25})}
                            className="saas-input"
                          />
                        </div>
                        <div className="form-group">
                          <label>Threads</label>
                          <input
                            type="number"
                            value={configForm.n_rollout_threads}
                            onChange={(e) => setConfigForm({...configForm, n_rollout_threads: parseInt(e.target.value) || 32})}
                            className="saas-input"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Column 2 */}
                    <div className="flex flex-col gap-4">
                      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider border-b border-slate-100 pb-2">Analysis Controls</h3>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div className="form-group">
                          <label>Eval Interval (episodes)</label>
                          <input
                            type="number"
                            value={configForm.eval_interval}
                            onChange={(e) => setConfigForm({...configForm, eval_interval: parseInt(e.target.value) || 5})}
                            className="saas-input"
                          />
                        </div>
                        <div className="form-group">
                          <label>Eval Noise Std</label>
                          <input
                            type="number"
                            step="0.05"
                            value={configForm.eval_noise_std}
                            onChange={(e) => setConfigForm({...configForm, eval_noise_std: parseFloat(e.target.value) || 0.0})}
                            className="saas-input"
                          />
                        </div>
                      </div>

                      <div className="form-checkbox-container">
                        <label className="form-checkbox-label">
                          <input
                            type="checkbox"
                            checked={configForm.disable_messages}
                            onChange={(e) => setConfigForm({...configForm, disable_messages: e.target.checked})}
                            className="form-checkbox"
                          />
                          <div className="form-checkbox-text">
                            <span>Disable Train Messages</span>
                            <p>Train without communication channel updates.</p>
                          </div>
                        </label>

                        <label className="form-checkbox-label">
                          <input
                            type="checkbox"
                            checked={configForm.eval_disable_messages}
                            onChange={(e) => setConfigForm({...configForm, eval_disable_messages: e.target.checked})}
                            className="form-checkbox"
                          />
                          <div className="form-checkbox-text">
                            <span>Disable Eval Messages</span>
                            <p>Ablates communications during evaluation.</p>
                          </div>
                        </label>

                        <label className="form-checkbox-label">
                          <input
                            type="checkbox"
                            checked={configForm.use_eval}
                            onChange={(e) => setConfigForm({...configForm, use_eval: e.target.checked})}
                            className="form-checkbox"
                          />
                          <div className="form-checkbox-text">
                            <span>Enable Evaluation</span>
                            <p>Collect causal influence metrics at epochs.</p>
                          </div>
                        </label>
                      </div>
                    </div>
                  </div>

                  <div className="border-t border-slate-100 pt-6 flex justify-end">
                    <button
                      type="submit"
                      disabled={activeRunId !== null || isSubmitting}
                      className="btn btn-primary"
                    >
                      <Play className="h-4 w-4 fill-current" />
                      {isSubmitting ? "Launching..." : "Launch Subprocess"}
                    </button>
                  </div>
                </form>
              </div>
            )}

            {/* VIEW 3: LIVE CONSOLE LOGS */}
            {activeTab === 'logs' && (
              <div className="card" style={{ maxWidth: '1000px', margin: '0 auto' }}>
                <div className="card-header-border">
                  <div>
                    <h2 className="card-title">Live Subprocess Console Output</h2>
                    <p className="card-subtitle">Showing terminal output for the active run: {activeRunId ? activeRunId : "None"}</p>
                  </div>
                  {activeRunId && (
                    <button onClick={handleStopRun} className="btn btn-secondary">
                      <Square className="h-3.5 w-3.5 fill-current text-rose-500" />
                      Stop Run
                    </button>
                  )}
                </div>

                <div ref={logConsoleRef} className="console-box">
                  {logs ? (
                    <pre className="whitespace-pre-wrap">{logs}</pre>
                  ) : (
                    <div className="h-full flex items-center justify-center text-slate-500 italic">
                      [Terminal console idle. No active logs running. Launch a new run from the configurator.]
                    </div>
                  )}
                </div>
              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  );
}
