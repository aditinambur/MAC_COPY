import React, { useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { MetricRow } from '../../types';

interface MetricsChartProps {
  metrics: MetricRow[];
}

export const MetricsChart: React.FC<MetricsChartProps> = ({ metrics }) => {
  const [selectedMetric, setSelectedMetric] = useState<string>('all');

  if (!metrics || metrics.length === 0) {
    return (
      <div className="empty-metrics-box">
        <p>No causal influence metrics recorded yet. Metrics update at each eval interval.</p>
      </div>
    );
  }

  // Detect which metrics actually have values
  const hasKl = metrics.some((m) => m.causal_influence_kl_mean !== undefined && m.causal_influence_kl_mean !== null);
  const hasCommEffect = metrics.some((m) => m.comm_effect !== undefined && m.comm_effect !== null);
  const hasValue = metrics.some((m) => m.causal_influence_value_sensitivity_mean !== undefined && m.causal_influence_value_sensitivity_mean !== null);
  const hasReward = metrics.some((m) => (m.eval_reward !== undefined && m.eval_reward !== null) || (m.reward !== undefined && m.reward !== null));

  // Format data points for charts
  const data = metrics.map((m, idx) => {
    return {
      index: idx + 1,
      step: m.step || m.episode || idx * 5,
      kl_divergence: m.causal_influence_kl_mean !== undefined && m.causal_influence_kl_mean !== null ? Number(m.causal_influence_kl_mean.toFixed(4)) : null,
      value_sensitivity: m.causal_influence_value_sensitivity_mean !== undefined && m.causal_influence_value_sensitivity_mean !== null ? Number(m.causal_influence_value_sensitivity_mean.toFixed(4)) : null,
      comm_effect: m.comm_effect !== undefined && m.comm_effect !== null ? Number(m.comm_effect.toFixed(2)) : null,
      eval_reward: m.eval_reward !== undefined && m.eval_reward !== null ? Number(m.eval_reward.toFixed(2)) : (m.reward !== undefined && m.reward !== null ? Number(m.reward.toFixed(2)) : null),
    };
  });

  return (
    <div className="metrics-chart-container">
      <div className="chart-header">
        <h4 className="chart-title">Causal Influence of Communication (CIC) Over Training</h4>
        
        <div className="chart-controls">
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="metric-select"
          >
            <option value="all">All Causal Signals</option>
            {hasKl && <option value="kl">Policy Sensitivity: KL(P || Q)</option>}
            {hasCommEffect && <option value="comm_effect">Causal Effect: comm_effect</option>}
            {hasValue && <option value="value">Value Sensitivity: |V_real - V_zero|</option>}
            {hasReward && <option value="reward">Evaluation Reward</option>}
          </select>
        </div>
      </div>

      <div className="chart-body">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a3342" />
            <XAxis
              dataKey="step"
              stroke="#64748b"
              label={{ value: 'Evaluation Step / Epoch', position: 'insideBottom', offset: -10, fill: '#64748b' }}
            />
            <YAxis stroke="#64748b" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0f172a',
                borderColor: '#334155',
                borderRadius: '8px',
                color: '#f8fafc',
              }}
            />
            <Legend verticalAlign="top" height={36} />

            {(selectedMetric === 'all' ? hasKl : selectedMetric === 'kl') && (
              <Line
                type="monotone"
                dataKey="kl_divergence"
                name="Policy Sensitivity (KL)"
                stroke="#38bdf8"
                strokeWidth={2}
                dot={{ r: 3, fill: '#38bdf8' }}
                activeDot={{ r: 6 }}
              />
            )}

            {(selectedMetric === 'all' ? hasCommEffect : selectedMetric === 'comm_effect') && (
              <Line
                type="monotone"
                dataKey="comm_effect"
                name="Comm Effect (Reward Gain)"
                stroke="#10b981"
                strokeWidth={2}
                dot={{ r: 3, fill: '#10b981' }}
                activeDot={{ r: 6 }}
              />
            )}

            {(selectedMetric === 'all' ? hasValue : selectedMetric === 'value') && (
              <Line
                type="monotone"
                dataKey="value_sensitivity"
                name="Value Sensitivity"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={{ r: 3, fill: '#f59e0b' }}
                activeDot={{ r: 6 }}
              />
            )}

            {(selectedMetric === 'reward') && (
              <Line
                type="monotone"
                dataKey="eval_reward"
                name="Evaluation Reward"
                stroke="#a855f7"
                strokeWidth={2}
                dot={{ r: 3, fill: '#a855f7' }}
                activeDot={{ r: 6 }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
