import React from 'react';
import { 
  LayoutDashboard, 
  PlaySquare, 
  Wrench, 
  Archive,
  BarChart3
} from 'lucide-react';

export type TabType = 'dashboard' | 'experiments' | 'detail' | 'repair' | 'archive';

interface SidebarProps {
  currentTab: TabType;
  onSelectTab: (tab: TabType) => void;
  selectedRunId: string | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  selectedRunId,
}) => {
  const navItems = [
    { id: 'dashboard' as TabType, label: 'Dashboard', icon: LayoutDashboard },
    { id: 'experiments' as TabType, label: 'Simulate & Train', icon: PlaySquare },
    { id: 'detail' as TabType, label: 'Run Details', icon: BarChart3, disabled: !selectedRunId },
    { id: 'repair' as TabType, label: 'Causal Repair Studio', icon: Wrench, disabled: !selectedRunId },
    { id: 'archive' as TabType, label: 'Archived Runs', icon: Archive },
  ];

  return (
    <aside className="app-sidebar">
      <div className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          const isDisabled = item.disabled;

          return (
            <button
              key={item.id}
              className={`sidebar-link ${isActive ? 'active' : ''} ${
                isDisabled ? 'disabled' : ''
              }`}
              onClick={() => !isDisabled && onSelectTab(item.id)}
              disabled={isDisabled}
            >
              <Icon size={18} />
              <span>{item.label}</span>
              {item.id === 'detail' && selectedRunId && (
                <span className="sidebar-pill">{selectedRunId.split('_').pop()}</span>
              )}
            </button>
          );
        })}
      </div>

      <div className="sidebar-footer">
        <div className="version-info">
          <span>MAC Framework v2.1</span>
          <p>MPE simple_spread MARL</p>
        </div>
      </div>
    </aside>
  );
};
