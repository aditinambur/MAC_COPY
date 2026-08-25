import os
import sys
import subprocess
import json
import csv
import glob
import shutil
from typing import Dict, List, Optional
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="MAC Multi-Agent Emergent Communication API")

# Allow CORS for development & Docker container networking
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_ROOT = "onpolicy/scripts/results/MPE/simple_spread/mappo"
REGISTRY_PATH = os.path.join(RESULTS_ROOT, "run_registry.json")

# Global process tracking
active_process: Optional[subprocess.Popen] = None
active_run_id: Optional[str] = None
active_process_type: Optional[str] = None  # "training" or "repair"

class RunConfig(BaseModel):
    experiment_name: str = "check"
    seed: int = 1
    num_agents: int = 2
    num_landmarks: int = 3
    num_env_steps: int = 100000
    episode_length: int = 25
    n_rollout_threads: int = 32
    eval_interval: int = 5
    disable_messages: bool = False
    eval_disable_messages: bool = False
    eval_noise_std: float = 0.25
    use_eval: bool = True

class RepairConfig(BaseModel):
    checkpoint_name: str  # e.g. "checkpoint_1958400"
    mirror_scope: str = "partner_full"  # "partner_full", "partner", "all"
    repair_target: str = "auto"  # "auto", "embedding", "comm", "full", "noncomm"
    controller: str = "causal"  # "causal", "reward_only"
    measure_episodes: int = 6
    repair_iters: int = 15
    seed: int = 1

def load_registry() -> Dict:
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_registry(registry: Dict):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    try:
        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        print(f"Error saving registry: {e}")

def get_next_run(experiment_name: str) -> tuple:
    exp_dir = os.path.join(RESULTS_ROOT, experiment_name)
    if not os.path.exists(exp_dir):
        return "run1", os.path.join(exp_dir, "run1").replace("\\", "/")
    
    exst_run_nums = []
    for item in os.listdir(exp_dir):
        if item.startswith("run") and os.path.isdir(os.path.join(exp_dir, item)):
            try:
                num = int(item[3:])
                exst_run_nums.append(num)
            except ValueError:
                pass
                
    if not exst_run_nums:
        next_run = "run1"
    else:
        next_run = f"run{max(exst_run_nums) + 1}"
        
    return next_run, os.path.join(exp_dir, next_run).replace("\\", "/").replace("//", "/")

def scan_runs() -> List[Dict]:
    registry = load_registry()
    runs = []
    
    pattern = os.path.join(RESULTS_ROOT, "*", "run*")
    run_dirs = glob.glob(pattern)
    
    for run_dir in run_dirs:
        run_dir = run_dir.replace("\\", "/")
        path_parts = run_dir.split("/")
        run_name = path_parts[-1]
        exp_name = path_parts[-2]
        run_id = f"{exp_name}_{run_name}"
        
        info = registry.get(run_id, {})
        
        csv_path = os.path.join(run_dir, "causal_influence.csv")
        has_metrics = os.path.exists(csv_path)
        gif_path = os.path.join(run_dir, "render.gif")
        has_gif = os.path.exists(gif_path)
        
        # Check if checkpoints exist
        models_dir = os.path.join(run_dir, "models")
        checkpoints_count = 0
        if os.path.exists(models_dir) and os.path.isdir(models_dir):
            checkpoints_count = len([d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))])
            
        repair_log = os.path.join(run_dir, "repair_output.log")
        has_repair = os.path.exists(repair_log)
        
        is_active = (active_run_id == run_id and active_process is not None and active_process.poll() is None)
        status = "running" if is_active else info.get("status", "completed" if has_metrics else "unknown")

        runs.append({
            "run_id": run_id,
            "experiment_name": exp_name,
            "run_name": run_name,
            "path": run_dir,
            "status": status,
            "config": info.get("config", {}),
            "has_metrics": has_metrics,
            "has_gif": has_gif,
            "has_repair": has_repair,
            "checkpoints_count": checkpoints_count,
            "archived": info.get("archived", False)
        })
        
    def sort_key(run):
        is_running = 1 if run["status"] == "running" else 0
        try:
            num = int(run["run_name"][3:])
            return (is_running, run["experiment_name"], num)
        except ValueError:
            return (is_running, run["experiment_name"], 0)
            
    runs.sort(key=sort_key, reverse=True)
    return runs

@app.get("/api/system/info")
def get_system_info():
    """Return GPU, CUDA, and hardware acceleration status."""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    
    global active_process, active_run_id, active_process_type
    is_busy = active_process is not None and active_process.poll() is None
    
    return {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "active_run_id": active_run_id if is_busy else None,
        "process_type": active_process_type if is_busy else None,
        "status": "busy" if is_busy else "idle"
    }

@app.get("/api/config-schema")
def get_config_schema():
    """Return default hyperparameter options for MPE."""
    return {
        "experiment_name": {"type": "string", "default": "mpe_experiment", "description": "Identifier for the training run"},
        "num_agents": {"type": "integer", "default": 2, "description": "Number of learning agents"},
        "num_landmarks": {"type": "integer", "default": 3, "description": "Number of landmark targets"},
        "seed": {"type": "integer", "default": 1, "description": "Random seed for reproducibility"},
        "num_env_steps": {"type": "integer", "default": 100000, "description": "Total env steps to train"},
        "episode_length": {"type": "integer", "default": 25, "description": "Max length of each episode"},
        "n_rollout_threads": {"type": "integer", "default": 32, "description": "Number of parallel environments during training"},
        "eval_interval": {"type": "integer", "default": 5, "description": "Interval between evaluations (in episodes)"},
        "disable_messages": {"type": "boolean", "default": False, "description": "Disable communication during training (ablation)"},
        "eval_disable_messages": {"type": "boolean", "default": False, "description": "Disable communication during evaluation"},
        "eval_noise_std": {"type": "number", "default": 0.25, "description": "Standard deviation of noise added to messages"},
        "use_eval": {"type": "boolean", "default": True, "description": "Whether to perform causal evaluations during training"}
    }

@app.get("/api/runs")
def list_runs():
    """List all completed and active runs."""
    return scan_runs()

@app.get("/api/runs/active")
def get_active_run():
    """Get the current running process details."""
    global active_process, active_run_id, active_process_type
    if active_process is not None and active_process.poll() is None:
        return {"run_id": active_run_id, "process_type": active_process_type, "status": "running"}
    return {"run_id": None, "process_type": None, "status": "idle"}

@app.post("/api/runs/start")
def start_run(config: RunConfig, background_tasks: BackgroundTasks):
    """Start a new training run with GPU acceleration enabled by default."""
    global active_process, active_run_id, active_process_type
    
    if active_process is not None and active_process.poll() is None:
        raise HTTPException(status_code=400, detail="Another training or repair task is currently running.")
        
    next_run_name, next_run_dir = get_next_run(config.experiment_name)
    run_id = f"{config.experiment_name}_{next_run_name}"
    
    os.makedirs(next_run_dir, exist_ok=True)
    log_file_path = os.path.join(next_run_dir, "output.log")
    
    # Notice: --cuda is store_false, so omitting --cuda enables GPU execution
    cmd = [
        sys.executable, "-u", "onpolicy/scripts/train/train_mpe.py",
        "--env_name", "MPE",
        "--scenario_name", "simple_spread",
        "--algorithm_name", "mappo",
        "--use_wandb",
        "--experiment_name", config.experiment_name,
        "--num_agents", str(config.num_agents),
        "--num_landmarks", str(config.num_landmarks),
        "--seed", str(config.seed),
        "--num_env_steps", str(config.num_env_steps),
        "--episode_length", str(config.episode_length),
        "--n_rollout_threads", str(config.n_rollout_threads),
        "--eval_interval", str(config.eval_interval),
        "--eval_noise_std", str(config.eval_noise_std)
    ]
    
    if config.disable_messages:
        cmd.append("--disable_messages")
    if config.eval_disable_messages:
        cmd.append("--eval_disable_messages")
    if config.use_eval:
        cmd.append("--use_eval")
        
    log_file = open(log_file_path, "w")
    try:
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONUNBUFFERED"] = "1"
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        env["WANDB_MODE"] = "disabled"
        env["WANDB_SILENT"] = "true"
        
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=os.getcwd(),
            env=env,
            start_new_session=True
        )
    except Exception as e:
        log_file.close()
        raise HTTPException(status_code=500, detail=f"Failed to launch training subprocess: {e}")
        
    active_process = proc
    active_run_id = run_id
    active_process_type = "training"
    
    registry = load_registry()
    registry[run_id] = {
        "status": "running",
        "config": config.dict(),
        "pid": proc.pid
    }
    save_registry(registry)
    
    def monitor_process(p, r_id):
        p.wait()
        log_file.close()
        
        global active_process, active_run_id, active_process_type
        if active_run_id == r_id:
            active_process = None
            active_run_id = None
            active_process_type = None
            
        reg = load_registry()
        if r_id in reg:
            csv_path = os.path.join(next_run_dir, "causal_influence.csv")
            reg[r_id]["status"] = "completed" if (os.path.exists(csv_path) and p.returncode == 0) else "failed"
            save_registry(reg)
            
    background_tasks.add_task(monitor_process, proc, run_id)
    
    return {"run_id": run_id, "status": "running", "log_file": log_file_path}

@app.post("/api/runs/{run_id}/repair")
def run_repair(run_id: str, config: RepairConfig, background_tasks: BackgroundTasks):
    """Launch Phase 2/3 Causal Break, Selective Detection, and Online Repair on a checkpoint."""
    global active_process, active_run_id, active_process_type
    
    if active_process is not None and active_process.poll() is None:
        raise HTTPException(status_code=400, detail="Another task is currently running.")
        
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    model_dir = os.path.join(run["path"], "models", config.checkpoint_name)
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=400, detail=f"Checkpoint directory {config.checkpoint_name} does not exist.")
        
    repair_log_path = os.path.join(run["path"], "repair_output.log")
    
    cmd = [
        sys.executable, "-u", "onpolicy/scripts/phase2_3_repair.py",
        "--env_name", "MPE",
        "--scenario_name", "simple_spread",
        "--algorithm_name", "mappo",
        "--seed", str(config.seed),
        "--model_dir", model_dir,
        "--mirror_scope", config.mirror_scope,
        "--measure_episodes", str(config.measure_episodes),
        "--repair_iters", str(config.repair_iters),
        "--n_rollout_threads", "32",
        "--n_eval_rollout_threads", "1"
    ]
    
    if config.controller != "causal":
        cmd.extend(["--controller", config.controller])
    if config.repair_target != "auto":
        cmd.extend(["--repair_target", config.repair_target])
        
    log_file = open(repair_log_path, "w")
    try:
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONUNBUFFERED"] = "1"
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        env["WANDB_MODE"] = "disabled"
        env["WANDB_SILENT"] = "true"
        
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=os.getcwd(),
            env=env,
            start_new_session=True
        )
    except Exception as e:
        log_file.close()
        raise HTTPException(status_code=500, detail=f"Failed to launch repair subprocess: {e}")
        
    active_process = proc
    active_run_id = run_id
    active_process_type = "repair"
    
    def monitor_repair(p, r_id):
        p.wait()
        log_file.close()
        
        global active_process, active_run_id, active_process_type
        if active_run_id == r_id:
            active_process = None
            active_run_id = None
            active_process_type = None
            
    background_tasks.add_task(monitor_repair, proc, run_id)
    
    return {"run_id": run_id, "status": "running", "repair_log": repair_log_path}

@app.post("/api/runs/stop")
def stop_run():
    """Stop the active training or repair subprocess."""
    global active_process, active_run_id, active_process_type
    
    if active_process is None or active_process.poll() is not None:
        raise HTTPException(status_code=400, detail="No active process is in progress.")
        
    try:
        if sys.platform != "win32":
            os.killpg(os.getpgid(active_process.pid), 15)
        else:
            active_process.terminate()
        active_process.wait(timeout=5)
    except Exception:
        try:
            active_process.kill()
        except Exception:
            pass
            
    registry = load_registry()
    if active_run_id in registry:
        registry[active_run_id]["status"] = "stopped"
        save_registry(registry)
        
    stopped_id = active_run_id
    active_process = None
    active_run_id = None
    active_process_type = None
    
    return {"run_id": stopped_id, "status": "stopped"}

@app.get("/api/runs/{run_id}/logs")
def get_run_logs(run_id: str, lines: int = 200):
    """Fetch latest training logs."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    log_path = os.path.join(run["path"], "output.log")
    if not os.path.exists(log_path):
        return {"logs": "[No training logs found]"}
        
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_lines = f.readlines()
            return {"logs": "".join(log_lines[-lines:])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {e}")

@app.get("/api/runs/{run_id}/repair_logs")
def get_repair_logs(run_id: str, lines: int = 200):
    """Fetch latest Phase 2/3 repair logs."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    log_path = os.path.join(run["path"], "repair_output.log")
    if not os.path.exists(log_path):
        return {"logs": "[No repair experiments executed on this run yet]"}
        
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_lines = f.readlines()
            return {"logs": "".join(log_lines[-lines:])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read repair logs: {e}")

@app.get("/api/runs/{run_id}/metrics")
def get_run_metrics(run_id: str):
    """Fetch causal influence metrics from causal_influence.csv."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    csv_path = os.path.join(run["path"], "causal_influence.csv")
    if not os.path.exists(csv_path):
        return {"metrics": []}
        
    try:
        metrics = []
        with open(csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_row = {}
                for k, v in row.items():
                    try:
                        parsed_row[k] = float(v) if '.' in v or 'e' in v else int(v)
                    except ValueError:
                        parsed_row[k] = v
                metrics.append(parsed_row)
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse metrics: {e}")

@app.get("/api/runs/{run_id}/checkpoints")
def get_run_checkpoints(run_id: str):
    """List saved checkpoints under models/ directory."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    models_dir = os.path.join(run["path"], "models")
    if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
        return {"checkpoints": []}
        
    checkpoints = []
    for d in os.listdir(models_dir):
        full_p = os.path.join(models_dir, d)
        if os.path.isdir(full_p):
            is_best = (d == "checkpoint_best")
            step = 0
            if d.startswith("checkpoint_") and not is_best:
                try:
                    step = int(d.replace("checkpoint_", ""))
                except ValueError:
                    pass
            checkpoints.append({
                "name": d,
                "path": full_p.replace("\\", "/"),
                "step": step,
                "is_best": is_best
            })
            
    checkpoints.sort(key=lambda x: (not x["is_best"], x["step"]), reverse=True)
    return {"checkpoints": checkpoints}

@app.post("/api/runs/{run_id}/render")
def render_run(run_id: str):
    """Run rendering for a completed run to generate render.gif."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    models_dir = os.path.join(run["path"], "models")
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        raise HTTPException(status_code=400, detail="No model weights found. Check if training is completed.")
        
    config = run.get("config", {})
    num_agents = config.get("num_agents")
    num_landmarks = config.get("num_landmarks")

    # Auto-detect num_agents and num_landmarks if missing from config
    if num_agents is None:
        csv_path = os.path.join(run["path"], "causal_influence.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f:
                    header = f.readline().strip().split(",")
                    agent_cols = [c for c in header if c.startswith("causal_influence_kl_agent")]
                    if agent_cols:
                        num_agents = len(agent_cols)
            except Exception:
                pass

    if num_agents is None:
        num_agents = 2

    if num_landmarks is None:
        actor_path = os.path.join(models_dir, "actor.pt")
        if os.path.exists(actor_path):
            try:
                sd = torch.load(actor_path, map_location="cpu")
                if "message_head.weight" in sd:
                    obs_dim = sd["message_head.weight"].shape[1]
                    calc_landmarks = (obs_dim - 4 - 4 * (num_agents - 1)) // 2
                    if calc_landmarks > 0:
                        num_landmarks = calc_landmarks
            except Exception:
                pass

    if num_landmarks is None:
        num_landmarks = 3

    cmd = [
        sys.executable, "onpolicy/scripts/render/render_mpe.py",
        "--env_name", "MPE",
        "--scenario_name", "simple_spread",
        "--algorithm_name", "mappo",
        "--use_render",
        "--model_dir", models_dir,
        "--save_gifs",
        "--gif_dir", run["path"],
        "--n_rollout_threads", "1",
        "--n_training_threads", "1",
        "--render_episodes", "1",
        "--num_agents", str(num_agents),
        "--num_landmarks", str(num_landmarks)
    ]
    
    try:
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )
        gif_path = os.path.join(run["path"], "render.gif")
        if not os.path.exists(gif_path):
            raise HTTPException(status_code=500, detail=f"Rendering finished, but render.gif was not found. Error: {result.stderr}")
            
        return {"status": "success", "gif_path": gif_path}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Rendering timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute rendering: {e}")

@app.get("/api/runs/{run_id}/gif")
def get_run_gif(run_id: str):
    """Serve the render.gif visualization of the run."""
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    gif_path = os.path.join(run["path"], "render.gif")
    if not os.path.exists(gif_path):
        raise HTTPException(status_code=404, detail="No render.gif found for this run.")
        
    return FileResponse(gif_path, media_type="image/gif")

@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    """Delete a run from the filesystem and registry."""
    global active_process, active_run_id
    if active_run_id == run_id and active_process is not None and active_process.poll() is None:
        raise HTTPException(status_code=400, detail="Cannot delete an active process. Terminate it first.")
        
    runs = scan_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
        
    run_dir = run["path"]
    if os.path.exists(run_dir):
        try:
            shutil.rmtree(run_dir)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete directory: {e}")
            
    registry = load_registry()
    if run_id in registry:
        del registry[run_id]
        save_registry(registry)
        
    return {"status": "success", "detail": f"Successfully deleted run {run_id}"}

@app.post("/api/runs/{run_id}/archive")
def archive_run(run_id: str):
    """Toggle the archived status of a run in the registry."""
    registry = load_registry()
    
    if run_id not in registry:
        runs = scan_runs()
        run = next((r for r in runs if r["run_id"] == run_id), None)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        registry[run_id] = {
            "status": run["status"],
            "config": run["config"],
            "archived": True
        }
    else:
        current_status = registry[run_id].get("archived", False)
        registry[run_id]["archived"] = not current_status
        
    save_registry(registry)
    return {"status": "success", "archived": registry[run_id]["archived"]}
