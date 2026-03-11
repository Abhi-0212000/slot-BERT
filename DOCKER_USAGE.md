# Slot-BERT Docker Usage Guide

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Building the Image](#building-the-image)
- [Training Modes](#training-modes)
- [Method 1: Detached Mode (Recommended)](#method-1-detached-mode-recommended)
- [Method 2: tmux/screen + Interactive Container](#method-2-tmuxscreen--interactive-container)
- [Method 3: Interactive (Dev/Debug Only)](#method-3-interactive-devdebug-only)
- [Command Reference Breakdown](#command-reference-breakdown)
- [Monitoring & Logs](#monitoring--logs)
- [Visdom (Live Training Viz)](#visdom-live-training-viz)
- [Data Persistence](#data-persistence)
- [Docker Exec (Entering a Running Container)](#docker-exec-entering-a-running-container)
- [Cleanup](#cleanup)
- [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement        | Version          |
| :----------------- | :--------------- |
| Docker Engine      | ≥ 20.10          |
| Docker Compose V2  | ≥ 2.0            |
| NVIDIA Driver      | ≥ 535            |
| nvidia-container-toolkit | installed   |

Verify GPU access:
```bash
docker run --rm --runtime=nvidia nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

---

## Project Structure

```
slot-BERT/
├── Dockerfile              # CUDA 12.2.2 + PyTorch 2.5.1+cu121
├── docker-compose.yml      # Service definition (GPU, volumes, ports)
├── main.py                 # Training entry point
├── model_config.yml        # Model hyperparameters
├── scripts/
│   └── train.sh            # Starts Visdom + training
├── src/                    # Source code (mounted as volume)
│   ├── model/
│   ├── working_para/       # Training configs per dataset
│   ├── Data_samples/       # Dataset .pkl files
│   └── Model_checkpoint/   # → mapped to ./checkpoints/ on host
│
│  ──── Host-only dirs (created by bind mounts) ────
├── checkpoints/            # Persisted model checkpoints
├── visdom_data/            # Persisted Visdom state
└── outputs/                # Training outputs
```

---

## Building the Image

```bash
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/slot-BERT

# First-time build (or after Dockerfile / requirements.txt changes):
docker compose build

# Force rebuild from scratch (no cache):
docker compose build --no-cache
```

> **Note:** You only need to rebuild when `Dockerfile` or `src/requirements.txt` changes.
> Source code in `src/`, `main.py`, `model_config.yml`, and `scripts/` are bind-mounted,
> so edits on the host take effect immediately — no rebuild needed.

---

## Training Modes

Available `--mode` values (defined in `main.py`):

| Mode              | Dataset    | Config file                                       |
| :---------------- | :--------- | :------------------------------------------------ |
| `train_cholec`    | Cholec80   | `working_para/working_dir_root_train_cholec_p3.py` |
| `train_miccai`    | MICCAI     | `working_para/working_dir_root_train_miccai_p3.py` |
| `train_Thoracic`  | Thoracic   | `working_para/working_dir_root_train_Thoracic_p3.py` |

---

## Method 1: Detached Mode (Recommended)

**What it is:** Docker's built-in background execution. The container runs as a daemon
process managed by the Docker engine. No tmux or screen needed. Survives SSH disconnects
because the process lives in Docker, not in your shell session.

### Start training:
```bash
docker compose run -d --name qte9489-slotbert-train train \
  bash scripts/train.sh --mode train_cholec
```

### Parameter breakdown:

| Part | What it does |
| :--- | :----------- |
| `docker compose run` | Creates a **new one-off container** from the service definition in `docker-compose.yml`. Unlike `docker compose up`, this is for running a single command. |
| `-d` | **Detached mode.** Runs the container in the background. Your terminal is freed immediately. The container keeps running even if you close your SSH session. |
| `--name qte9489-slotbert-train` | Assigns a **human-readable name** to the container. Without this, Docker generates a random name like `jolly_curie`. The name is used for `docker logs`, `docker stop`, `docker exec`, etc. |
| `train` | The **service name** from `docker-compose.yml`. Tells Docker which service definition to use (image, volumes, GPU, ports, etc.). |
| `bash scripts/train.sh` | The **command to execute** inside the container. This **overrides** any `command:` in docker-compose.yml. It runs `train.sh` which starts Visdom, then runs `python main.py`. |
| `--mode train_cholec` | **Argument passed to `train.sh`**, which parses it and passes it to `python main.py --mode train_cholec`. |

### What happens after you run this:
1. Docker prints the container ID and returns you to your shell immediately.
2. Inside the container: Visdom starts on port 8097, then training begins.
3. You can disconnect SSH — the container keeps running.
4. Reconnect anytime and check with `docker logs` or `docker exec`.

### Follow logs:
```bash
docker logs -f qte9489-slotbert-train          # live stream (Ctrl+C to stop watching)
docker logs --tail 100 qte9489-slotbert-train   # last 100 lines
```

### Stop and clean up:
```bash
docker stop qte9489-slotbert-train
docker rm qte9489-slotbert-train
```

---

## Method 2: tmux/screen + Interactive Container

**What it is:** You create a persistent terminal session (tmux/screen) on the server,
then run the container interactively inside it. The tmux session survives SSH disconnects
because tmux is a server-side process.

**When to use:** When you want a live interactive shell inside the container (e.g., for
debugging, running multiple commands, inspecting files) AND need SSH-disconnect survival.

### Using tmux:
```bash
# SSH into server
ssh qte9489@<server>

# Create a named tmux session:
tmux new -s slotbert

# Inside tmux — start the container interactively:
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/slot-BERT
docker compose run --rm train bash

# Now you're inside the container shell:
bash scripts/train.sh --mode train_cholec

# ─── Detach from tmux (training continues): ───
#   Press: Ctrl+B, then D
#
# You can now safely close your SSH connection.

# ─── Reattach later: ───
ssh qte9489@<server>
tmux attach -t slotbert

# ─── List all tmux sessions: ───
tmux ls

# ─── Kill the session when done: ───
tmux kill-session -t slotbert
```

### Using screen (alternative to tmux):
```bash
# Create a named screen session:
screen -S slotbert

# Run container inside screen:
cd /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/slot-BERT
docker compose run --rm train bash
bash scripts/train.sh --mode train_cholec

# Detach: Ctrl+A, then D
# Reattach:
screen -r slotbert
```

### Detached Mode vs tmux — comparison:

| Feature | Detached (`-d`) | tmux/screen |
| :--- | :--- | :--- |
| Survives SSH disconnect | Yes | Yes |
| Interactive shell access | No (use `docker exec`) | Yes (reattach) |
| See live output directly | `docker logs -f` | Yes (in tmux pane) |
| Run ad-hoc commands | `docker exec -it ... bash` | Just type in shell |
| Setup complexity | None | Need tmux/screen installed |
| Multiple terminals | No | Yes (tmux panes/windows) |
| Best for | Fire-and-forget training | Interactive development |

---

## Method 3: Interactive (Dev/Debug Only)

**Warning:** This ties the container to your shell. If SSH dies, the container dies too.
Only use for quick debugging.

```bash
# Start interactive container:
docker compose run --rm train bash

# You're now inside the container. Run anything:
bash scripts/train.sh --mode train_cholec

# Or run Python directly:
python main.py --mode train_cholec

# Or poke around:
ls src/Data_samples/
python -c "import torch; print(torch.cuda.is_available())"

# Exit (stops and removes container due to --rm):
exit
```

---

## Command Reference Breakdown

### `docker compose run` vs `docker compose up`

| | `docker compose run` | `docker compose up` |
| :--- | :--- | :--- |
| Purpose | Run a **one-off command** | Start **all services** |
| Creates | One container | All containers in the file |
| Command | You specify the command | Uses `command:` from yml |
| Use case | Training runs, debugging | Multi-service apps |

### Key flags:

| Flag | Meaning |
| :--- | :--- |
| `-d` | Detached (background). Container runs without occupying your terminal. |
| `--rm` | Auto-remove container when it exits. Good for interactive/debug sessions. |
| `--name <name>` | Give the container a name. Required for `-d` so you can find it later. |
| `--service-ports` | Publish ports defined in yml. (`run` does NOT publish ports by default unless `-d` is used or `--service-ports` is specified.) |

**Important:** `docker compose run -d` publishes ports automatically. `docker compose run` (without `-d` and without `--service-ports`) does NOT publish ports.

---

## Monitoring & Logs

```bash
# Live log stream (Ctrl+C to stop watching, container keeps running):
docker logs -f qte9489-slotbert-train

# Last N lines:
docker logs --tail 50 qte9489-slotbert-train

# Check if container is running:
docker ps | grep slotbert

# Check all containers (including stopped):
docker ps -a | grep slotbert

# Resource usage (CPU, memory, GPU):
docker stats qte9489-slotbert-train
```

---

## Visdom (Live Training Viz)

`train.sh` automatically starts a Visdom server inside the container on port 8097.

### Access Visdom:

**If on the same machine:**
```
http://localhost:8097
```

**If SSH'd into a remote server (recommended):**
```bash
# On your LOCAL machine, create an SSH tunnel:
ssh -L 8097:localhost:8097 qte9489@<server-hostname>

# Then open in your local browser:
http://localhost:8097
```

**If behind a corporate proxy:** Port 8097 is likely blocked. Use the SSH tunnel
approach above — it routes traffic through the SSH connection on port 22.

---

## Data Persistence

All training artifacts are persisted on the host via bind mounts:

| What | Container Path | Host Path |
| :--- | :--- | :--- |
| Model checkpoints | `/app/src/Model_checkpoint/` | `./checkpoints/` |
| Visdom state | `/root/.visdom/` | `./visdom_data/` |
| Training outputs | `/app/outputs/` | `./outputs/` |
| Source code | `/app/src/` | `./src/` (editable) |
| Training script | `/app/scripts/` | `./scripts/` (editable) |
| Main entry | `/app/main.py` | `./main.py` (editable) |
| Model config | `/app/model_config.yml` | `./model_config.yml` (editable) |

> **Key insight:** Because `src/`, `main.py`, `model_config.yml`, and `scripts/` are
> mounted as volumes, you can edit code on the host and it takes effect inside
> the container immediately — no rebuild needed. Only `Dockerfile` or `requirements.txt`
> changes need a rebuild.

---

## Docker Exec (Entering a Running Container)

When a container is running in detached mode, you can open a shell inside it:

```bash
# Open an interactive bash shell inside the running container:
docker exec -it qte9489-slotbert-train bash

# Now you're inside the container — inspect files, check GPU, etc.:
nvidia-smi
ls /app/src/Model_checkpoint/
python -c "import torch; print(torch.cuda.memory_summary())"

# Exit the exec shell (container keeps running):
exit
```

You can also run one-off commands without entering a shell:
```bash
# Check GPU memory:
docker exec qte9489-slotbert-train nvidia-smi

# Check if Python process is running:
docker exec qte9489-slotbert-train ps aux | grep python

# Check disk usage:
docker exec qte9489-slotbert-train df -h /app
```

---

## Cleanup

```bash
# Stop a running container:
docker stop qte9489-slotbert-train

# Remove a stopped container:
docker rm qte9489-slotbert-train

# Stop + remove in one line:
docker stop qte9489-slotbert-train && docker rm qte9489-slotbert-train

# Remove the built image (forces rebuild next time):
docker compose down --rmi local

# Remove ALL stopped containers, unused images, and build cache:
docker system prune -f

# Nuclear option — remove everything (images, volumes, containers):
# docker system prune -a --volumes -f   # ⚠️ DANGEROUS
```

---

## Quick Reference Cheat Sheet

```bash
# ─── Build ───────────────────────────────────────────────
docker compose build

# ─── Train (detached) ───────────────────────────────────
docker compose run -d --name qte9489-slotbert-train train \
  bash scripts/train.sh --mode train_cholec

# ─── Logs ────────────────────────────────────────────────
docker logs -f qte9489-slotbert-train

# ─── Enter container ─────────────────────────────────────
docker exec -it qte9489-slotbert-train bash

# ─── Status ──────────────────────────────────────────────
docker ps | grep slotbert

# ─── Stop ────────────────────────────────────────────────
docker stop qte9489-slotbert-train && docker rm qte9489-slotbert-train

# ─── Train with tmux ─────────────────────────────────────
tmux new -s slotbert
docker compose run --rm train bash
bash scripts/train.sh --mode train_cholec
# Ctrl+B, D to detach | tmux attach -t slotbert to reattach
```

---

## Troubleshooting

| Problem | Solution |
| :--- | :--- |
| `Container name already in use` | `docker rm qte9489-slotbert-train` then retry |
| `nvidia-container-cli: device error` | Check `nvidia-smi` on host, restart Docker: `sudo systemctl restart docker` |
| `CUDA out of memory` | Reduce `Batch_size` in `src/working_para/*.py` or reduce `img_size` |
| `Visdom 502/503` | Use SSH tunnel: `ssh -L 8097:localhost:8097 qte9489@<server>` |
| `FileNotFoundError: model0.pth` | Set `Continue_flag = False` in `src/working_para/*.py` |
| `Port 8097 already in use` | `docker ps` to find conflicting container, stop it |
| Checkpoints not appearing on host | Verify `./checkpoints/` exists: `mkdir -p checkpoints` |
| Code changes not reflected | Only applies to mounted files. If you changed `Dockerfile` or `requirements.txt`, rebuild with `docker compose build` |
