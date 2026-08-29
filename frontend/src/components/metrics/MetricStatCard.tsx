import React from 'react';
import { LucideIcon } from 'lucide-react';

interface MetricStatCardProps {
  title: string;
  value: string | number;
  subtext?: string;
  icon: LucideIcon;
  color?: string;
}

export const MetricStatCard: React.FC<MetricStatCardProps> = ({
  title,
  value,
  subtext,
  icon: Icon,
  color = 'text-primary',
}) => {
  return (
    <div className="metric-stat-card">
      <div className="stat-card-header">
        <span className="stat-title">{title}</span>
        <div className={`stat-icon-wrap ${color}`}>
          <Icon size={18} />
        </div>
      </div>
      <div className="stat-value">{value}</div>
      {subtext && <div className="stat-subtext">{subtext}</div>}
    </div>
  );
};
