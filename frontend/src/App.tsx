import React, { useState, useEffect } from 'react';
import { 
  fetchSystemInfo, 
  fetchRuns, 
  startTraining, 
  stopCurrentProcess, 
  fetchRunLogs, 
  fetchRunMetrics, 
  fetchRunCheckpoints, 
  startCausalRepair, 
  fetchRepairLogs, 
  toggleArchiveRun, 
  deleteRun 
} from './services/api';
import { Run, RunConfig, RepairConfig, SystemInfo, MetricRow, CheckpointInfo } from './types';
import { Header } from './components/layout/Header';
import { Sidebar, TabType } from './components/layout/Sidebar';
import { DashboardPage } from './pages/DashboardPage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { RunDetailPage } from './pages/RunDetailPage';
import { CausalRepairPage } from './pages/CausalRepairPage';
import { RunList } from './components/experiments/RunList';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<TabType>('dashboard');
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  // Run specific details
  const [selectedRunMetrics, setSelectedRunMetrics] = useState<MetricRow[]>([]);
  const [selectedRunCheckpoints, setSelectedRunCheckpoints] = useState<CheckpointInfo[]>([]);
  const [selectedRunLogs, setSelectedRunLogs] = useState<string>('');
  const [selectedRunRepairLogs, setSelectedRunRepairLogs] = useState<string>('');

  // Process actions state
  const [isLaunching, setIsLaunching] = useState<boolean>(false);
  const [isStopping, setIsStopping] = useState<boolean>(false);
  const [isRepairing, setIsRepairing] = useState<boolean>(false);

  // Poll system info & runs
  const loadRunsAndSystem = async () => {
    try {
      const [sys, allRuns] = await Promise.all([
        fetchSystemInfo().catch(() => null),
        fetchRuns().catch(() => []),
      ]);
      setSystemInfo(sys);
      setRuns(allRuns);

      setSelectedRunId((prev) => {
        if (!prev && allRuns.length > 0) {
          return allRuns[0].run_id;
        }
        if (prev && allRuns.some((r) => r.run_id === prev)) {
          return prev;
        }
        return allRuns.length > 0 ? allRuns[0].run_id : null;
      });
    } catch (err) {
      console.error('Failed to poll system status:', err);
    }
  };

  useEffect(() => {
    loadRunsAndSystem();
    const interval = setInterval(loadRunsAndSystem, 3000);
    return () => clearInterval(interval);
  }, []);

  // Poll active run logs/metrics
  useEffect(() => {
    if (!selectedRunId) return;

    const loadRunDetails = async () => {
      try {
        const [metrics, checkpoints, logs, repairLogs] = await Promise.all([
          fetchRunMetrics(selectedRunId).catch(() => []),
          fetchRunCheckpoints(selectedRunId).catch(() => []),
          fetchRunLogs(selectedRunId).catch(() => ''),
          fetchRepairLogs(selectedRunId).catch(() => ''),
        ]);
        setSelectedRunMetrics(metrics);
        setSelectedRunCheckpoints(checkpoints);
        setSelectedRunLogs(logs);
        setSelectedRunRepairLogs(repairLogs);
      } catch (err) {
        console.error('Failed to load run details:', err);
      }
    };

    loadRunDetails();
    const detailInterval = setInterval(loadRunDetails, 2500);
    return () => clearInterval(detailInterval);
  }, [selectedRunId]);

  // Handlers
  const handleLaunchExperiment = async (config: RunConfig) => {
    setIsLaunching(true);
    try {
      const res = await startTraining(config);
      setSelectedRunId(res.run_id);
      setCurrentTab('detail');
      await loadRunsAndSystem();
    } catch (err: any) {
      alert(`Launch error: ${err.message}`);
    } finally {
      setIsLaunching(false);
    }
  };

  const handleStopActive = async () => {
    setIsStopping(true);
    try {
      await stopCurrentProcess();
      await loadRunsAndSystem();
    } catch (err: any) {
      alert(`Stop error: ${err.message}`);
    } finally {
      setIsStopping(false);
    }
  };

  const handleStartRepair = async (config: RepairConfig) => {
    if (!selectedRunId) return;
    setIsRepairing(true);
    try {
      await startCausalRepair(selectedRunId, config);
      await loadRunsAndSystem();
    } catch (err: any) {
      alert(`Repair error: ${err.message}`);
    } finally {
      setIsRepairing(false);
    }
  };

  const handleToggleArchive = async (runId: string) => {
    try {
      await toggleArchiveRun(runId);
      await loadRunsAndSystem();
    } catch (err: any) {
      alert(`Archive error: ${err.message}`);
    }
  };

  const handleDeleteRun = async (runId: string) => {
    if (!confirm(`Are you sure you want to permanently delete run ${runId}?`)) return;
    try {
      await deleteRun(runId);
      if (selectedRunId === runId) {
        setSelectedRunId(null);
      }
      await loadRunsAndSystem();
    } catch (err: any) {
      alert(`Delete error: ${err.message}`);
    }
  };

  const selectedRun = runs.find((r) => r.run_id === selectedRunId) || null;

  return (
    <div className="app-layout">
      <Header
        systemInfo={systemInfo}
        onRefresh={loadRunsAndSystem}
        onStopActive={handleStopActive}
        isStopping={isStopping}
        onViewActiveRun={(activeId) => {
          setSelectedRunId(activeId);
          setCurrentTab('detail');
        }}
      />

      <div className="app-main-content">
        <Sidebar
          currentTab={currentTab}
          onSelectTab={setCurrentTab}
          selectedRunId={selectedRunId}
        />

        <main className="page-body">
          {currentTab === 'dashboard' && (
            <DashboardPage
              runs={runs}
              systemInfo={systemInfo}
              selectedRunId={selectedRunId}
              onSelectRun={(id) => {
                setSelectedRunId(id);
                setCurrentTab('detail');
              }}
              onInspectRepair={(id) => {
                setSelectedRunId(id);
                setCurrentTab('repair');
              }}
              onToggleArchive={handleToggleArchive}
              onDeleteRun={handleDeleteRun}
              onNavigateToSimulate={() => setCurrentTab('experiments')}
            />
          )}

          {currentTab === 'experiments' && (
            <ExperimentsPage
              runs={runs}
              selectedRunId={selectedRunId}
              onLaunch={handleLaunchExperiment}
              isLaunching={isLaunching}
              isBusy={systemInfo?.status === 'busy'}
              onSelectRun={(id) => {
                setSelectedRunId(id);
                setCurrentTab('detail');
              }}
              onInspectRepair={(id) => {
                setSelectedRunId(id);
                setCurrentTab('repair');
              }}
              onToggleArchive={handleToggleArchive}
              onDeleteRun={handleDeleteRun}
            />
          )}

          {currentTab === 'detail' && selectedRun && (
            <RunDetailPage
              run={selectedRun}
              metrics={selectedRunMetrics}
              checkpoints={selectedRunCheckpoints}
              logs={selectedRunLogs}
              onBack={() => setCurrentTab('experiments')}
              onOpenRepair={(id) => {
                setSelectedRunId(id);
                setCurrentTab('repair');
              }}
              onRefreshRunData={loadRunsAndSystem}
            />
          )}

          {currentTab === 'repair' && (
            <CausalRepairPage
              selectedRun={selectedRun}
              checkpoints={selectedRunCheckpoints}
              repairLogs={selectedRunRepairLogs}
              onStartRepair={handleStartRepair}
              isRepairing={isRepairing}
              onBack={() => setCurrentTab('detail')}
            />
          )}

          {currentTab === 'archive' && (
            <div className="page-container">
              <div className="page-header">
                <h2>Archived Experiments</h2>
                <p>Experiments hidden from the main dashboard.</p>
              </div>
              <RunList
                runs={runs}
                selectedRunId={selectedRunId}
                onSelectRun={(id) => {
                  setSelectedRunId(id);
                  setCurrentTab('detail');
                }}
                onInspectRepair={(id) => {
                  setSelectedRunId(id);
                  setCurrentTab('repair');
                }}
                onToggleArchive={handleToggleArchive}
                onDeleteRun={handleDeleteRun}
                showArchived={true}
              />
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
