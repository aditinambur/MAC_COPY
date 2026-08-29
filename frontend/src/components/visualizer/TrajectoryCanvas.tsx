import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, ShieldAlert, Radio } from 'lucide-react';

interface AgentState {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  color: string;
  token: number; // 0 to 4
}

interface LandmarkState {
  id: number;
  x: number;
  y: number;
  occupiedBy?: number;
}

interface TrajectoryCanvasProps {
  numAgents?: number;
  numLandmarks?: number;
  isPerturbed?: boolean;
  perturbationScope?: 'partner_full' | 'partner' | 'all';
}

export const TrajectoryCanvas: React.FC<TrajectoryCanvasProps> = ({
  numAgents = 2,
  numLandmarks = 3,
  isPerturbed = false,
  perturbationScope = 'partner_full',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [step, setStep] = useState<number>(0);
  const [showGhostPerception, setShowGhostPerception] = useState<boolean>(true);
  const [showCommChannel, setShowCommChannel] = useState<boolean>(true);

  // Generate simulated dynamic trajectory episode for interactive visual proof
  const maxSteps = 50;

  // Colors for agents
  const AGENT_COLORS = ['#38bdf8', '#f43f5e', '#a855f7', '#fb923c', '#10b981'];

  // Initialize landmarks and paths
  const landmarks: LandmarkState[] = [
    { id: 0, x: -0.6, y: 0.5 },
    { id: 1, x: 0.6, y: 0.5 },
    { id: 2, x: 0.0, y: -0.6 },
  ].slice(0, numLandmarks);

  useEffect(() => {
    let interval: any;
    if (isPlaying) {
      interval = setInterval(() => {
        setStep((prev) => (prev + 1) % maxSteps);
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isPlaying, maxSteps]);

  // Render on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const scale = width / 2.4; // Map [-1.2, 1.2] to canvas pixels
    const cx = width / 2;
    const cy = height / 2;

    const toCanvasX = (x: number) => cx + x * scale;
    const toCanvasY = (y: number) => cy - y * scale;

    // Clear background
    ctx.fillStyle = '#0b0f19';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let x = -1; x <= 1; x += 0.5) {
      ctx.beginPath();
      ctx.moveTo(toCanvasX(x), 0);
      ctx.lineTo(toCanvasX(x), height);
      ctx.stroke();
    }
    for (let y = -1; y <= 1; y += 0.5) {
      ctx.beginPath();
      ctx.moveTo(0, toCanvasY(y));
      ctx.lineTo(width, toCanvasY(y));
      ctx.stroke();
    }

    // Draw landmarks
    landmarks.forEach((lm) => {
      const lx = toCanvasX(lm.x);
      const ly = toCanvasY(lm.y);

      // Outer ring
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(lx, ly, 28, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // Landmark center
      ctx.fillStyle = '#334155';
      ctx.beginPath();
      ctx.arc(lx, ly, 12, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`L${lm.id + 1}`, lx, ly);
    });

    // Calculate agent positions based on step
    const progress = step / maxSteps;
    const agents: AgentState[] = [];

    for (let i = 0; i < numAgents; i++) {
      const angle = (i * 2 * Math.PI) / numAgents + progress * Math.PI;
      const targetLm = landmarks[i % landmarks.length];
      
      // Interpolate towards target with some orbit/avoidance
      const startX = Math.cos(angle) * 0.8;
      const startY = Math.sin(angle) * 0.8;
      const curX = startX + (targetLm.x - startX) * Math.min(progress * 1.4, 1.0);
      const curY = startY + (targetLm.y - startY) * Math.min(progress * 1.4, 1.0);

      agents.push({
        id: i,
        x: curX,
        y: curY,
        vx: (targetLm.x - curX) * 0.1,
        vy: (targetLm.y - curY) * 0.1,
        color: AGENT_COLORS[i % AGENT_COLORS.length],
        token: (i + Math.floor(step / 10)) % 5,
      });
    }

    // Draw communication lines between agents
    if (showCommChannel && agents.length > 1) {
      for (let i = 0; i < agents.length; i++) {
        for (let j = i + 1; j < agents.length; j++) {
          const a1 = agents[i];
          const a2 = agents[j];
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 6]);
          ctx.beginPath();
          ctx.moveTo(toCanvasX(a1.x), toCanvasY(a1.y));
          ctx.lineTo(toCanvasX(a2.x), toCanvasY(a2.y));
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }

    // Draw Perturbation Ghost Perception (if perturbed)
    if (isPerturbed && showGhostPerception && agents.length >= 2) {
      const a1 = agents[0];
      const a2 = agents[1];

      // If partner_full: partner relative position is negated: (a2 - a1) -> -(a2 - a1)
      const relX = a2.x - a1.x;
      const relY = a2.y - a1.y;
      const ghostX = a1.x - relX;
      const ghostY = a1.y - relY;

      const gx = toCanvasX(ghostX);
      const gy = toCanvasY(ghostY);

      // Ghost partner representation
      ctx.fillStyle = 'rgba(244, 63, 94, 0.25)';
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(gx, gy, 16, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#f43f5e';
      ctx.font = 'bold 9px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Ghost (Illusion)', gx, gy - 22);

      // Dotted vector connecting Agent 1 to its distorted belief
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.6)';
      ctx.beginPath();
      ctx.moveTo(toCanvasX(a1.x), toCanvasY(a1.y));
      ctx.lineTo(gx, gy);
      ctx.stroke();
    }

    // Draw real agents & their emitted language tokens
    agents.forEach((ag) => {
      const ax = toCanvasX(ag.x);
      const ay = toCanvasY(ag.y);

      // Agent body
      ctx.fillStyle = ag.color;
      ctx.shadowColor = ag.color;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(ax, ay, 15, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Agent label
      ctx.fillStyle = '#0f172a';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`A${ag.id + 1}`, ax, ay);

      // Emitted Token Speech Bubble
      const bubbleX = ax;
      const bubbleY = ay - 26;
      ctx.fillStyle = '#1e293b';
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(bubbleX - 22, bubbleY - 10, 44, 18, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#38bdf8';
      ctx.font = 'bold 9px monospace';
      ctx.fillText(`tok_${ag.token}`, bubbleX, bubbleY);
    });
  }, [step, isPerturbed, showGhostPerception, showCommChannel, numAgents, numLandmarks, maxSteps]);

  return (
    <div className="trajectory-canvas-wrap">
      <div className="canvas-header">
        <div className="canvas-title">
          <Radio size={16} className="text-emerald" />
          <span>Interactive 2D Trajectory Simulation ({numAgents} Agents / {numLandmarks} Landmarks)</span>
        </div>

        <div className="canvas-options">
          <label className="toggle-badge">
            <input
              type="checkbox"
              checked={showCommChannel}
              onChange={(e) => setShowCommChannel(e.target.checked)}
            />
            <span>Show Comm Channel</span>
          </label>

          {isPerturbed && (
            <label className="toggle-badge warning">
              <input
                type="checkbox"
                checked={showGhostPerception}
                onChange={(e) => setShowGhostPerception(e.target.checked)}
              />
              <ShieldAlert size={13} />
              <span>Partner Illusion ({perturbationScope})</span>
            </label>
          )}
        </div>
      </div>

      <div className="canvas-viewport">
        <canvas ref={canvasRef} width={480} height={420} className="sim-canvas" />
      </div>

      <div className="canvas-controls">
        <button className="icon-btn-sm" onClick={() => setIsPlaying(!isPlaying)}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <button
          className="icon-btn-sm"
          onClick={() => {
            setIsPlaying(false);
            setStep(0);
          }}
        >
          <RotateCcw size={14} />
        </button>

        <input
          type="range"
          min={0}
          max={maxSteps - 1}
          value={step}
          onChange={(e) => {
            setIsPlaying(false);
            setStep(parseInt(e.target.value));
          }}
          className="step-slider"
        />
        <span className="step-counter">Step {step + 1}/{maxSteps}</span>
      </div>
    </div>
  );
};
