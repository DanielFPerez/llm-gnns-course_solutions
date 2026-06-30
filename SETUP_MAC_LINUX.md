# Mac & Linux: ML Development Setup Guide

A complete, ordered guide for setting up a GPU-accelerated machine learning
development environment on **macOS** (Apple Silicon or Intel) and **native Linux**
(Ubuntu/Debian). Pick the path that matches your machine.

> **Why no virtualization layer?** Unlike Windows, Mac and Linux run the ML
> ecosystem natively. There is no WSL equivalent to set up — you work directly in
> a terminal.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install System Dependencies](#2-install-system-dependencies)
3. [GPU Drivers and CUDA (Linux with NVIDIA only)](#3-gpu-drivers-and-cuda-linux-with-nvidia-only)
4. [Set Up the Python Environment](#4-set-up-the-python-environment)
5. [Install and Verify PyTorch](#5-install-and-verify-pytorch)
6. [VS Code Setup](#6-vs-code-setup)
7. [Git & SSH Setup (GitHub)](#7-git--ssh-setup-github)
8. [Clone and Run the Course Project](#8-clone-and-run-the-course-project)
9. [JupyterLab](#9-jupyterlab)
10. [Daily Workflow Cheat Sheet](#10-daily-workflow-cheat-sheet)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

### macOS

- **Apple Silicon Mac** (M1/M2/M3/M4) — recommended for GPU-accelerated ML via
  Apple's MPS backend. Training speed is excellent for the course workloads.
- **Intel Mac** — works for all labs; PyTorch runs on CPU only (no GPU
  acceleration). Expect slower training times.
- macOS 13 Ventura or later recommended.
- [Homebrew](https://brew.sh) installed (the guide installs it if not present).

### Linux (Ubuntu/Debian)

- Ubuntu 22.04 LTS or 24.04 LTS recommended (other Debian-based distros work).
- An **NVIDIA GPU** if you want CUDA acceleration (optional — CPU works for all
  course labs, just slower). AMD GPU support via ROCm is possible but not covered
  here.
- `sudo` privileges for installing packages and drivers.

---

## 2. Install System Dependencies

### macOS

Install Homebrew if you don't have it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow any post-install instructions printed at the end (especially the `eval`
line for Apple Silicon that adds Homebrew to your PATH).

Install Python and Git:

```bash
brew install python git
```

Verify the versions:

```bash
python3 --version   # should be 3.11 or 3.12
git --version
```

> **Don't use the system Python** at `/usr/bin/python3` — it's managed by Apple
> and pip installs into a read-only path. The Homebrew Python at
> `/opt/homebrew/bin/python3` (Apple Silicon) or `/usr/local/bin/python3` (Intel)
> is the right one. Run `which python3` to confirm.

### Linux (Ubuntu/Debian)

Update the package index and install Python tooling:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv build-essential git
```

Verify:

```bash
python3 --version   # 3.10+ on Ubuntu 22.04, 3.12+ on 24.04
git --version
```

---

## 3. GPU Drivers and CUDA (Linux with NVIDIA only)

> **macOS users: skip this section entirely.** MPS acceleration requires no
> driver installation — PyTorch talks to the GPU through macOS system frameworks.
>
> **Linux users without an NVIDIA GPU:** also skip. CPU-only training works fine
> for all course labs.

### 3a. Install the NVIDIA driver

The safest method is through Ubuntu's official repositories:

```bash
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot, verify the driver is loaded:

```bash
nvidia-smi
```

You should see your GPU, driver version, and a CUDA version number in the
top-right corner. **Do not proceed to the next step until this works.**

> The CUDA version shown by `nvidia-smi` is the *maximum* the driver supports —
> not what's installed. PyTorch ships its own CUDA runtime in the pip wheel, so
> you do not need to install the CUDA toolkit separately unless you plan to
> compile custom CUDA kernels.

### 3b. (Optional) Install the CUDA toolkit for custom kernel compilation

Plain PyTorch training does **not** require this. Skip unless you know you need
`nvcc`.

```bash
# Find your driver-compatible CUDA version at developer.nvidia.com/cuda-downloads
# Example for CUDA 12.4 on Ubuntu 22.04:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

Add `nvcc` to your PATH by appending to `~/.bashrc`:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

---

## 4. Set Up the Python Environment

Always use a virtual environment — it isolates the project's packages from the
system Python and avoids version conflicts between projects.

```bash
cd ~
mkdir ml-project && cd ml-project
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt gains a `(.venv)` prefix when the environment is active. You need to
re-run `source .venv/bin/activate` in each new terminal session.

> **macOS alternative — `pyenv`:** If you manage multiple Python versions,
> [pyenv](https://github.com/pyenv/pyenv) (`brew install pyenv`) is a cleaner
> way to pin a specific Python version per project. The venv workflow above still
> applies on top of it.

---

## 5. Install and Verify PyTorch

### macOS (Apple Silicon — MPS acceleration)

```bash
pip install torch torchvision torchaudio
```

The standard pip wheel supports MPS on Apple Silicon automatically — no special
index URL needed.

**Verify:**

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x = torch.randn(100, 100, device='mps')
    print('MPS tensor sum:', x.sum().item())
"
```

Expected output on Apple Silicon:

```
PyTorch: 2.x.x
MPS available: True
MPS tensor sum: <some number>
```

> **Intel Mac:** `MPS available: False` is expected — training runs on CPU.
> Everything in the course still works, just slower. No action needed.

### Linux with NVIDIA GPU (CUDA acceleration)

```bash
pip install torch torchvision torchaudio
```

> **Newer GPUs (e.g. RTX 40/50-series) need a current PyTorch build.**
> Older wheels lack the right CUDA kernels and fail with
> `CUDA error: no kernel image is available for execution on the device`.
> The default `pip install torch` now ships a recent enough wheel. If you ever
> pin a version, grab the exact command from
> <https://pytorch.org/get-started/locally/> and choose the latest CUDA option.

**Verify:**

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    x = torch.randn(100, 100, device='cuda')
    print('CUDA tensor sum:', x.sum().item())
"
```

### Linux CPU-only

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

`torch.cuda.is_available()` will return `False` — that's expected.

---

## 6. VS Code Setup

VS Code runs natively on both Mac and Linux — no remote extension needed.

### Install VS Code

- **macOS:** `brew install --cask visual-studio-code`, or download from the VS
  Code website and drag to Applications.
- **Linux:** Download the `.deb` from the VS Code website, then:
  ```bash
  sudo dpkg -i code_*.deb
  sudo apt install -f   # fixes any missing dependencies
  ```

### Open the project from the terminal

```bash
cd ~/ml-project
code .
```

### Recommended extensions

Install these from the Extensions panel (`Cmd+Shift+X` / `Ctrl+Shift+X`):

- **Python** (Microsoft) — linting, IntelliSense, debugging
- **Jupyter** (Microsoft) — run `.ipynb` notebooks directly in VS Code

### Select the venv as your Python interpreter

1. Open any `.py` or `.ipynb` file.
2. Click the Python version in the bottom-right status bar (or open the Command
   Palette and run `Python: Select Interpreter`).
3. Pick the one that shows `.venv` in its path, e.g.
   `~/ml-project/.venv/bin/python`.

The Jupyter extension uses the same interpreter as the kernel for notebooks.

---

## 7. Git & SSH Setup (GitHub)

### 7a. Configure Git identity

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

### 7b. Create an SSH key (skip if you already have one)

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Accept the default location (`~/.ssh/id_ed25519`) or give it a custom name.
A passphrase is optional (see [7d](#7d-do-you-need-the-ssh-agent) for the
trade-off). Two files are created: a private key (no extension) and a public key
(`.pub`).

### 7c. Add the public key to GitHub

Print the public key and copy the entire line:

```bash
cat ~/.ssh/id_ed25519.pub
```

It starts with `ssh-ed25519 AAAA...`. Paste it into
**GitHub → Settings → SSH and GPG keys → New SSH key**.

> Common mistake: copying only part of the key, or pasting the **private** key.
> Always copy the full contents of the `.pub` file.

### 7d. Tell SSH which key to use (required for custom-named keys)

If you used a custom filename, SSH won't offer it automatically. Create or edit
`~/.ssh/config`:

```bash
nano ~/.ssh/config
```

Add:

```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/your-key-name
  IdentitiesOnly yes
```

Lock down permissions:

```bash
chmod 600 ~/.ssh/config
chmod 700 ~/.ssh
chmod 600 ~/.ssh/your-key-name
chmod 644 ~/.ssh/your-key-name.pub
```

`IdentitiesOnly yes` prevents `Too many authentication failures` when you have
several keys.

### 7e. Do you need the ssh-agent?

With `~/.ssh/config` pointing at your key, SSH reads it from disk each session —
**no agent needed.** If your key has a **passphrase** and you want it cached so
you don't retype it, add this to `~/.bashrc` (Linux) or `~/.zshrc` (macOS):

```bash
if [ -z "$SSH_AUTH_SOCK" ]; then
  eval "$(ssh-agent -s)" > /dev/null
  ssh-add ~/.ssh/your-key-name 2>/dev/null
fi
```

> **macOS bonus:** you can store the passphrase in the macOS Keychain so it
> survives reboots without needing the agent snippet:
> ```bash
> ssh-add --apple-use-keychain ~/.ssh/your-key-name
> ```
> And add `UseKeychain yes` to the `Host github.com` block in `~/.ssh/config`.

### 7f. Verify and clone

```bash
ssh -T git@github.com
```

Success:

```
Hi YourUsername! You've successfully authenticated, but GitHub does not provide shell access.
```

Then clone:

```bash
cd ~
git clone git@github.com:YourUsername/your-repo.git
```

---

## 8. Clone and Run the Course Project

```bash
cd ~
git clone git@github.com:YourUsername/llm-gnns-course_solutions.git
cd llm-gnns-course_solutions

# Activate your venv (or create a fresh one here)
source ~/.venv/bin/activate   # adjust path if your venv lives elsewhere

pip install -r environment/requirements.txt
```

**Smoke test** — save as `smoke_test.py` and run it to verify the full stack:

```python
import torch

print("PyTorch version:", torch.__version__)

# macOS Apple Silicon
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Backend: MPS (Apple Silicon GPU)")
# Linux NVIDIA
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Backend: CUDA —", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Backend: CPU (no GPU acceleration)")

x = torch.randn(10000, 10000, device=device)
result = (x @ x).sum()
print("Matrix multiply result:", result.item())

model = torch.nn.Linear(100, 1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
data = torch.randn(64, 100, device=device)
target = torch.randn(64, 1, device=device)

for step in range(5):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(data), target)
    loss.backward()
    optimizer.step()
    print(f"step {step}: loss = {loss.item():.4f}")

print("Everything works.")
```

```bash
python smoke_test.py
```

Decreasing loss values and no errors confirm the full pipeline.

---

## 9. JupyterLab

```bash
cd ~/llm-gnns-course_solutions
source ~/.venv/bin/activate   # if not already active
jupyter lab
```

Jupyter prints a URL like:

```
http://localhost:8888/lab?token=abc123...
```

Open it in your browser. The token in the URL is the authentication step — paste
the entire URL and it logs you straight in.

**Notes:**
- Your GPU is available in notebooks — first cell: `import torch; torch.backends.mps.is_available()` (Mac) or `torch.cuda.is_available()` (Linux).
- Port busy? Use `jupyter lab --port=8889`.
- **Alternative:** open `.ipynb` files directly in VS Code with the Jupyter
  extension and select `.venv` as the kernel — no port or token needed.

---

## 10. Daily Workflow Cheat Sheet

| Task | Command |
|------|---------|
| Go to project | `cd ~/llm-gnns-course_solutions` |
| Activate venv | `source .venv/bin/activate` |
| Open in VS Code | `code .` |
| Check GPU (Linux/NVIDIA) | `nvidia-smi` |
| Watch GPU live (Linux) | `watch -n 1 nvidia-smi` |
| Start JupyterLab | `jupyter lab` |
| Install new packages | `pip install <package>` (venv active) |
| Update requirements | `pip install -r environment/requirements.txt` |

**Remember:** the `(.venv)` prefix disappears in each new terminal session —
re-run `source .venv/bin/activate` before working.

---

## 11. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `which python3` shows `/usr/bin/python3` | Using system Python — activate your venv or check your PATH |
| `torch.backends.mps.is_available()` is `False` on Apple Silicon | macOS < 13, or running on Intel Mac (expected) — use CPU |
| `torch.cuda.is_available()` is `False` on Linux | `nvidia-smi` failing, or CPU-only torch wheel installed — check the driver first |
| `CUDA error: no kernel image is available` | torch wheel too old for your GPU — install a current build |
| `nvidia-smi: command not found` | Driver not installed or not in PATH — redo section 3a |
| `Permission denied (publickey)` on `git clone` | Custom-named SSH key not offered — add `~/.ssh/config` with `IdentityFile` (section 7d) |
| Port 8888 already in use | `jupyter lab --port=8889` |
| `ModuleNotFoundError` in notebook | Package not installed in active venv — `pip install <package>` with venv active |
| Jupyter kernel dies immediately on Mac | MPS out-of-memory — reduce batch size or close other GPU-heavy apps |
| `pip install` says "error: externally-managed-environment" | You're using the system Python — activate your venv first |

---

*Setup summary: install system deps (Homebrew/apt) → GPU drivers if using Linux
+ NVIDIA → Python venv → PyTorch (MPS on Apple Silicon, CUDA on Linux/NVIDIA,
CPU as fallback) → VS Code natively → Git SSH. No virtualization layer needed.*
