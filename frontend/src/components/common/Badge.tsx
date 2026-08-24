import React from 'react';

interface BadgeProps {
  variant: 'running' | 'completed' | 'failed' | 'stopped' | 'idle' | 'busy' | 'accepted' | 'rejected' | 'unknown' | 'default';
  children: React.ReactNode;
}

export const Badge: React.FC<BadgeProps> = ({ variant, children }) => {
  const getBadgeClass = () => {
    switch (variant) {
      case 'running':
      case 'busy':
        return 'badge-running';
      case 'completed':
      case 'accepted':
        return 'badge-success';
      case 'failed':
      case 'rejected':
        return 'badge-danger';
      case 'stopped':
        return 'badge-warning';
      case 'idle':
        return 'badge-idle';
      default:
        return 'badge-default';
    }
  };

  return <span className={`badge ${getBadgeClass()}`}>{children}</span>;
};
