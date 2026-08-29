import React from 'react';
import { Cpu, Zap, Activity, RefreshCw, Square } from 'lucide-react';
import { SystemInfo } from '../../types';
import { Badge } from '../common/Badge';

interface HeaderProps {
  systemInfo: SystemInfo | null;
  onRefresh: () => void;
  onStopActive: () => void;
  isStopping: boolean;
  onViewActiveRun?: (runId: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  systemInfo,
  onRefresh,
  onStopActive,
  isStopping,
  onViewActiveRun,
}) => {
  return (
    <header className="app-header">
      <div className="header-left">
        <div className="logo-badge">
          <Zap size={18} className="text-primary" />
          <span className="logo-text">MAC</span>
          <span className="logo-sub">Emergent Comm & Repair</span>
        </div>
      </div>

      <div className="header-right">
        {/* GPU Accelerator Status */}
        <div className="gpu-status-pill">
          <Cpu size={15} className={systemInfo?.cuda_available ? "text-emerald" : "text-gray"} />
          <span className="gpu-name">
            {systemInfo?.cuda_available
              ? `${systemInfo.device_name} (CUDA)`
              : 'CPU Mode'}
          </span>
        </div>

        {/* Active Task Indicator */}
        {systemInfo?.active_run_id ? (
          <div className="active-task-bar">
            <div
              className="active-task-clickable"
              onClick={() => onViewActiveRun && onViewActiveRun(systemInfo.active_run_id!)}
              title="Click to view live logs & metrics for this active run"
            >
              <Activity size={14} className="animate-spin text-amber inline mr-1" />
              <span className="active-task-text">
                {systemInfo.process_type === 'repair' ? 'Repairing' : 'Training'}:{' '}
                <strong>{systemInfo.active_run_id}</strong> (Click to view live)
              </span>
            </div>
            <button
              className="stop-btn"
              onClick={onStopActive}
              disabled={isStopping}
              title="Stop current execution"
            >
              <Square size={13} fill="currentColor" />
              <span>Stop</span>
            </button>
          </div>
        ) : (
          <Badge variant="idle">System Idle</Badge>
        )}

        <button className="icon-btn" onClick={onRefresh} title="Refresh System Status">
          <RefreshCw size={16} />
        </button>
      </div>
    </header>
  );
};
