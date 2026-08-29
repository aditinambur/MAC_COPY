import React, { useRef, useEffect } from 'react';
import { Terminal, Copy, Check } from 'lucide-react';

interface TerminalLogViewerProps {
  logs: string;
  title?: string;
  autoScroll?: boolean;
}

export const TerminalLogViewer: React.FC<TerminalLogViewerProps> = ({
  logs,
  title = 'Console Output Stream',
  autoScroll = true,
}) => {
  const logContainerRef = useRef<HTMLPreElement | null>(null);
  const [copied, setCopied] = React.useState(false);

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const handleCopy = () => {
    navigator.clipboard.writeText(logs);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="terminal-viewer">
      <div className="terminal-header">
        <div className="terminal-title">
          <Terminal size={14} />
          <span>{title}</span>
        </div>
        <button className="copy-btn" onClick={handleCopy}>
          {copied ? <Check size={13} className="text-emerald" /> : <Copy size={13} />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre ref={logContainerRef} className="terminal-body">
        {logs || '[Waiting for output stream...]'}
      </pre>
    </div>
  );
};
