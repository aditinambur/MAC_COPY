import React, { useState } from 'react';
import { 
  BarChart3, 
  Terminal, 
  Image, 
  Layers, 
  Wrench, 
  ArrowLeft,
  Radio
} from 'lucide-react';
import { Run, MetricRow, CheckpointInfo } from '../types';
import { MetricsChart } from '../components/metrics/MetricsChart';
import { TrajectoryCanvas } from '../components/visualizer/TrajectoryCanvas';
import { GifViewer } from '../components/visualizer/GifViewer';
import { TerminalLogViewer } from '../components/visualizer/TerminalLogViewer';
import { Badge } from '../components/common/Badge';

interface RunDetailPageProps {
  run: Run;
  metrics: MetricRow[];
  checkpoints: CheckpointInfo[];
  logs: string;
  onBack: () => void;
  onOpenRepair: (runId: string) => void;
  onRefreshRunData: () => void;
}

export const RunDetailPage: React.FC<RunDetailPageProps> = ({
  run,
  metrics,
  checkpoints,
  logs,
  onBack,
  onOpenRepair,
  onRefreshRunData,
}) => {
  const [activeTab, setActiveTab] = useState<'metrics' | 'trajectory' | 'gif' | 'logs' | 'checkpoints'>(
    run.status === 'running' && metrics.length === 0 ? 'logs' : 'metrics'
  );

  const numAgents = run.config?.num_agents || 2;
  const numLandmarks = run.config?.num_landmarks || 3;

  return (
    <div className="page-container">
      {/* Top Navigation & Header */}
      <div className="detail-header-bar">
        <button className="back-btn" onClick={onBack}>
          <ArrowLeft size={16} /> Back to Experiments
        </button>

        <div className="detail-header-actions">
          <button className="btn-primary-sm" onClick={() => onOpenRepair(run.run_id)}>
            <Wrench size={14} /> Open in Causal Repair Studio
          </button>
        </div>
      </div>

      {/* Main Run Banner */}
      <div className="run-banner">
        <div className="banner-left">
          <h2>{run.run_id}</h2>
          <span className="banner-path">{run.path}</span>
        </div>
        <div className="banner-right">
          <Badge variant={run.status}>{run.status.toUpperCase()}</Badge>
        </div>
      </div>

      {/* Detail Tab Navigation */}
      <div className="detail-tabs">
        <button
          className={`tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          <BarChart3 size={16} /> Causal Metrics & Charts
        </button>
        <button
          className={`tab-btn ${activeTab === 'trajectory' ? 'active' : ''}`}
          onClick={() => setActiveTab('trajectory')}
        >
          <Radio size={16} /> 2D Trajectory Simulation
        </button>
        <button
          className={`tab-btn ${activeTab === 'gif' ? 'active' : ''}`}
          onClick={() => setActiveTab('gif')}
        >
          <Image size={16} /> MPE Render GIF
        </button>
        <button
          className={`tab-btn ${activeTab === 'checkpoints' ? 'active' : ''}`}
          onClick={() => setActiveTab('checkpoints')}
        >
          <Layers size={16} /> Checkpoints ({checkpoints.length})
        </button>
        <button
          className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`}
          onClick={() => setActiveTab('logs')}
        >
          <Terminal size={16} /> Console Logs
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content-wrapper">
        {activeTab === 'metrics' && (
          <div className="tab-pane">
            <MetricsChart metrics={metrics} />
          </div>
        )}

        {activeTab === 'trajectory' && (
          <div className="tab-pane">
            <TrajectoryCanvas
              numAgents={numAgents}
              numLandmarks={numLandmarks}
              isPerturbed={false}
            />
          </div>
        )}

        {activeTab === 'gif' && (
          <div className="tab-pane">
            <GifViewer
              runId={run.run_id}
              hasGif={run.has_gif}
              onGifGenerated={onRefreshRunData}
            />
          </div>
        )}

        {activeTab === 'checkpoints' && (
          <div className="tab-pane">
            <div className="checkpoints-table-card">
              <h4>Saved Policy Checkpoints</h4>
              {checkpoints.length === 0 ? (
                <p className="p-4 text-gray">No checkpoints found in models/ directory.</p>
              ) : (
                <div className="checkpoints-grid">
                  {checkpoints.map((cp) => (
                    <div key={cp.name} className="checkpoint-card-item">
                      <div className="cp-header">
                        <strong>{cp.name}</strong>
                        {cp.is_best && <span className="best-tag">best</span>}
                      </div>
                      <span className="cp-step">{cp.step > 0 ? `Step ${cp.step.toLocaleString()}` : 'Best Eval'}</span>
                      <button
                        className="btn-outline-sm mt-3 w-full"
                        onClick={() => onOpenRepair(run.run_id)}
                      >
                        <Wrench size={13} /> Test Causal Repair
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'logs' && (
          <div className="tab-pane">
            <TerminalLogViewer logs={logs} title={`Training Logs - ${run.run_id}`} />
          </div>
        )}
      </div>
    </div>
  );
};
