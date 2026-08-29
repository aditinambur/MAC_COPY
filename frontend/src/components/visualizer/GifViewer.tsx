import React, { useState } from 'react';
import { RefreshCw, Image, AlertCircle } from 'lucide-react';
import { getRunGifUrl, triggerRenderGif } from '../../services/api';

interface GifViewerProps {
  runId: string;
  hasGif: boolean;
  onGifGenerated?: () => void;
}

export const GifViewer: React.FC<GifViewerProps> = ({
  runId,
  hasGif,
  onGifGenerated,
}) => {
  const [isRendering, setIsRendering] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [imgKey, setImgKey] = useState<number>(Date.now());

  const handleGenerate = async () => {
    setIsRendering(true);
    setError(null);
    try {
      await triggerRenderGif(runId);
      setImgKey(Date.now());
      if (onGifGenerated) onGifGenerated();
    } catch (err: any) {
      setError(err.message || 'Failed to render GIF');
    } finally {
      setIsRendering(false);
    }
  };

  return (
    <div className="gif-viewer-container">
      <div className="gif-header">
        <h4>Environment Render Trajectory (render.gif)</h4>
        <button
          className="btn-outline-sm"
          onClick={handleGenerate}
          disabled={isRendering}
        >
          <RefreshCw size={13} className={isRendering ? 'animate-spin' : ''} />
          {isRendering ? 'Generating GIF...' : hasGif ? 'Re-render GIF' : 'Render GIF'}
        </button>
      </div>

      {error && (
        <div className="error-alert">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      <div className="gif-body">
        {hasGif ? (
          <img
            key={imgKey}
            src={getRunGifUrl(runId)}
            alt={`Render trajectory for ${runId}`}
            className="render-gif-img"
          />
        ) : (
          <div className="empty-gif-box">
            <Image size={36} className="text-gray" />
            <p>No render.gif generated yet for this run.</p>
            <button className="btn-primary-sm" onClick={handleGenerate} disabled={isRendering}>
              Generate GIF Visualizer
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
