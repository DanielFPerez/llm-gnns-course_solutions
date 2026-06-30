# WSL2 + CUDA + VS Code: ML Development Setup Guide

A complete, ordered guide for setting up a GPU-accelerated machine learning
development environment on Windows using WSL2 (Ubuntu), NVIDIA CUDA, and VS Code.

> **Why WSL2?** The ML ecosystem (PyTorch, JAX, most research tooling) is
> Linux-first. WSL2 gives you a near-native Linux environment with full GPU
> access, while letting you keep Windows for everything else. You avoid the
> recurring friction of Windows-only build steps and missing Linux wheels.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install WSL2](#2-install-wsl2)
3. [Install the NVIDIA Driver (Windows side only)](#3-install-the-nvidia-driver-windows-side-only)
4. [Verify GPU Passthrough into WSL](#4-verify-gpu-passthrough-into-wsl)
5. [Set Up the Python Environment](#5-set-up-the-python-environment)
6. [Install and Verify PyTorch](#6-install-and-verify-pytorch)
7. [Important WSL Configurations](#7-important-wsl-configurations)
8. [VS Code + WSL](#8-vs-code--wsl)
9. [Git & SSH Setup (GitHub)](#9-git--ssh-setup-github)
10. [Running the Example Project](#10-running-the-example-project)
11. [JupyterLab in WSL](#11-jupyterlab-in-wsl)
12. [Daily Workflow Cheat Sheet](#12-daily-workflow-cheat-sheet)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

- **Windows 11** (recommended) or Windows 10 version 21H2+.
  Note: Windows 11 reports its version with a `10.0` prefix (e.g. build
  `10.0.26200`) for legacy reasons — that is still Windows 11.
- An **NVIDIA GPU** with administrative permission to install drivers.
- A broadband connection (you'll download several GB).

> **The one rule that prevents most problems:** You install exactly **one** GPU
> driver, and it goes on the **Windows** side. The Linux side gets the CUDA
> *toolkit/runtime*, never the driver. Installing an NVIDIA driver inside WSL
> breaks GPU passthrough.

---
    
## 2. Install WSL2

Open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

This enables the required features, sets WSL2 as the default, and installs
Ubuntu. **Reboot** when prompted. On first launch, Ubuntu asks you to create a
UNIX username and password (separate from your Windows login).

> If you see `A distribution with the supplied name already exists`, that just
> means Ubuntu is **already installed** — nothing is wrong, skip ahead.

Make sure everything is current (GPU passthrough needs a recent kernel):

```powershell
wsl --update
wsl --version
```

Also check **Settings → Windows Update** — the WSL kernel is patched through
Windows Update, and you need kernel **5.10.43.3 or higher** for GPU features.

Confirm your distro is on version 2:

```powershell
wsl -l -v
```

You want `Ubuntu` with `VERSION` showing `2`. (If it shows `1`, convert it with
`wsl --set-version Ubuntu 2`.)

**Opening Ubuntu later**, any of these work:
- Press the Windows key, type `ubuntu`, Enter.
- Click the `˅` dropdown in Windows Terminal and pick **Ubuntu**.
- Type `wsl` in any PowerShell/CMD window (type `exit` to return).

You know you're in Linux when the prompt changes from `PS C:\...>` to
`username@machine:~$`.

---

## 3. Install the NVIDIA Driver (Windows side only)

Download the standard **Game Ready** or **Studio** driver for your GPU from
[nvidia.com](https://www.nvidia.com/Download/index.aspx) and install it in
Windows normally. That single driver is automatically exposed into WSL2 through
`/usr/lib/wsl/lib/`.

**Do NOT** run anything like `sudo apt install nvidia-driver-xxx` inside Ubuntu.
That installs a conflicting Linux driver and breaks GPU passthrough.

Verify the Windows side first. In **PowerShell**:

```powershell
nvidia-smi
```

You should see your GPU, the driver version, and a "CUDA Version" in the
top-right. That CUDA number is the **maximum** the driver supports — not
something you've installed — so don't be surprised if it's higher than expected.

---

## 4. Verify GPU Passthrough into WSL

This is the gate that actually matters for ML. Open **Ubuntu** and run the same
command *inside Linux*:

```bash
nvidia-smi
```

The prompt should look like `dpere@kukulkan:~$` (Linux), **not** `PS C:\...`
(Windows). The output should report the **same driver version** and your GPU.

> A tiny version difference between the Windows `nvidia-smi` and the WSL one
> (e.g. `610.47` vs `610.43.02`) is normal — that's just the userspace tool
> build. The kernel driver (KMD) version is what matters and will match.

**Do not proceed until `nvidia-smi` works inside Ubuntu.** Everything downstream
depends on it. If it says `command not found`, update your Windows NVIDIA driver.

---

## 5. Set Up the Python Environment

Inside Ubuntu, update the system and install Python tooling:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv build-essential
```

(It will ask for the UNIX password you created during Ubuntu setup.)

Create your project with an isolated virtual environment. Modern Ubuntu blocks
system-wide pip installs, so a venv is the standard path:

```bash
mkdir ~/ml-project && cd ~/ml-project
python3 -m venv .venv
source .venv/bin/activate
```

You'll know the venv is active when your prompt gains a `(.venv)` prefix.

> **Critical filesystem rule:** Keep all code under your Linux home (`~/`),
> **not** on the Windows drive at `/mnt/c/...`. Cross-filesystem access is
> dramatically slower and is the #1 cause of "why is everything so slow?"
> If your prompt ever shows `/mnt/c/...`, run `cd ~` to get back home.
> To reach WSL files *from Windows*, open `\\wsl$\Ubuntu\home\<you>` in Explorer.

---

## 6. Install and Verify PyTorch

For standard PyTorch work you do **not** need a system-wide CUDA toolkit — the
pip wheels bundle their own CUDA runtime. You only need the Windows driver
(already installed) plus PyTorch.

With your venv active:

```bash
pip install torch torchvision torchaudio
```

> **Newer GPUs (e.g. RTX 50-series / Blackwell) need a current PyTorch build.**
> Older wheels lack the right CUDA kernels and fail with
> `CUDA error: no kernel image is available for execution on the device`.
> The default `pip install torch` now ships a recent enough wheel. If you ever
> pin a version, grab the exact command from
> <https://pytorch.org/get-started/locally/> and choose the latest CUDA option.

**Verify the full stack:**

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:

```
True
NVIDIA GeForce RTX 5060 Laptop GPU
```

`True` plus your GPU name means CUDA is wired up end to end. If you get `False`,
the usual cause is that `nvidia-smi` wasn't working in step 4, or a CPU-only
wheel got installed.

**When do you need the full CUDA toolkit (`nvcc`)?** Only if you **compile
custom CUDA code** (e.g. building flash-attention from source). If so, install
**only** the toolkit metapackage from NVIDIA's WSL repo — select **"WSL-Ubuntu"**
as the distribution on the [CUDA downloads page](https://developer.nvidia.com/cuda-downloads),
and never the `cuda`, `cuda-drivers`, or full `cuda-12-x`/`cuda-13-x`
metapackages (those try to install a Linux driver). Plain PyTorch training does
**not** require this.

---

## 7. Important WSL Configurations

### 7a. Cap WSL2 resource usage (`.wslconfig`)

Left unconfigured, WSL2 can balloon its RAM use. Create a file at
`C:\Users\<YourName>\.wslconfig` (on the **Windows** side) with something like:

```ini
[wsl2]
memory=24GB
processors=8
swap=8GB
```

Tune these to your machine and **leave headroom for Windows itself**. Apply
changes by running `wsl --shutdown` in PowerShell, then reopen Ubuntu.

### 7b. Remember VRAM is shared

Your GPU's VRAM is shared with whatever Windows is rendering. On an 8 GB card,
close GPU-heavy apps before large training runs. Watch usage live in a second
Ubuntu tab:

```bash
watch -n 1 nvidia-smi
```

### 7c. Working within an 8 GB VRAM budget

8 GB is great for learning, small-model fine-tuning, and experimentation, but
tight for larger training. Lean on: smaller batch sizes, mixed precision
(`bf16`/`fp16`), gradient checkpointing, and quantization (8-bit/4-bit loading
via `bitsandbytes`).

### 7d. Keep projects on the Linux filesystem

(Repeated because it matters most.) Code lives under `~/`, not `/mnt/c/...`.

---

## 8. VS Code + WSL

**Do not install a second VS Code inside Ubuntu.** Keep your Windows VS Code and
connect it into WSL. The GUI runs on Windows; a lightweight server runs inside
WSL and handles your terminal, extensions, language servers, and files on the
Linux side.

### One-time setup

1. In Windows VS Code, open the Extensions panel and install the **WSL**
   extension (official Microsoft one, formerly "Remote - WSL").

### Option A — launch from the WSL terminal (fastest)

```bash
cd ~/ml-project
code .
```

The first time, this auto-installs the server into Linux and opens a connected
window.

### Option B — start from Windows VS Code

Order matters: **connect to WSL first, then open the folder.**

1. Open VS Code on Windows.
2. Connect via any of: Command Palette (`Ctrl+Shift+P` → `WSL: Connect to WSL`),
   the blue/green `><` indicator in the bottom-left, or the Remote Explorer.
3. Once the bottom-left badge reads **WSL: Ubuntu**, open your project with
   **File → Open Folder** (`Ctrl+K Ctrl+O`). Because you're already connected,
   the picker browses the **Linux** filesystem — navigate to
   `/home/dpere/ml-project`.

> If you Open Folder *before* connecting, the picker shows the Windows `C:\`
> drive and you'll end up on the slow `\\wsl$\` path. Connect first.

After opening a WSL project once, it appears in **File → Open Recent** tagged
`[WSL: Ubuntu]` for one-click reconnection.

### Confirm you're connected correctly

- Bottom-left badge shows **WSL: Ubuntu**.
- Open a terminal with `` Ctrl+` `` — the prompt should be
  `dpere@kukulkan:~/ml-project$`, not PowerShell.
- Install functional extensions (Python, Jupyter) **into WSL** when prompted
  (VS Code offers an "Install in WSL" button). Themes can stay on Windows.

---

## 9. Git & SSH Setup (GitHub)

To clone your own repositories over SSH, you need an SSH key in WSL whose public
half is registered on GitHub.

### 9a. Create a key (skip if you already have one)

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter to accept the default location, or give it a custom name. A
passphrase is optional (see [9d](#9d-do-you-need-the-ssh-agent) for the
trade-off). This produces two files in `~/.ssh/`: a private key (no extension)
and a public key (`.pub`).

### 9b. Add the public key to GitHub

Print the **public** key and copy the entire single line:

```bash
cat ~/.ssh/your-key-name.pub
```

It starts with `ssh-ed25519 AAAA...`. Paste it into
**GitHub → Settings → SSH and GPG keys → New SSH key**. Make sure you're adding
it to the *correct* GitHub account (the one that owns the repos you'll clone).

> Common mistake: copying only part of the key, or pasting the **private** key.
> Always copy the full contents of the `.pub` file.

### 9c. Tell SSH which key to use (the reliable fix)

**This is essential if your key has a custom name.** SSH only auto-offers keys
with default names (`id_ed25519`, `id_rsa`); a custom-named key is ignored unless
you point SSH at it explicitly. This is the #1 cause of
`Permission denied (publickey)` even when the key is correctly added to GitHub.

Create or edit `~/.ssh/config`:

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

Replace `your-key-name` with your actual **private**-key filename (no `.pub`).
Then lock down permissions — SSH ignores configs and keys with loose perms:

```bash
chmod 600 ~/.ssh/config
chmod 700 ~/.ssh
chmod 600 ~/.ssh/your-key-name
chmod 644 ~/.ssh/your-key-name.pub
```

The `IdentitiesOnly yes` line makes SSH offer *only* this key for GitHub, which
also avoids `Too many authentication failures` errors once you have several keys.

### 9d. Do you need the ssh-agent?

With the `~/.ssh/config` above, SSH reads the key straight from disk every
session — **no agent, no `ssh-add` needed.** Reach for the agent only if your key
has a **passphrase** and you don't want to retype it each session. (If you ever
see `Could not open a connection to your authentication agent`, it just means the
agent isn't running — but with the config in place you don't need it.)

If you *do* want the agent to auto-start and load a passphrase-protected key,
add this to the end of `~/.bashrc`:

```bash
if [ -z "$SSH_AUTH_SOCK" ]; then
  eval "$(ssh-agent -s)" > /dev/null
  ssh-add ~/.ssh/your-key-name 2>/dev/null
fi
```

### 9e. Verify and clone

Test the connection:

```bash
ssh -T git@github.com
```

Success looks like:

```
Hi YourUsername! You've successfully authenticated, but GitHub does not provide shell access.
```

(The "does not provide shell access" part is **expected** — not an error.)
Confirm the username matches the account that owns the repo. Then clone:

```bash
cd ~
git clone git@github.com:YourUsername/your-repo.git
```

> **Filesystem reminder:** clone into your Linux home (`~/`), not `/mnt/c/...`,
> for the same performance reasons as everything else.

---

## 10. Running the Example Project

A known-good baseline that verifies the entire stack (GPU detection + real GPU
compute). Save as `~/ml-project/smoke_test.py`:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    # Push tensors to the GPU and do real math
    x = torch.randn(10000, 10000, device=device)
    result = (x @ x).sum()
    torch.cuda.synchronize()
    print("GPU matmul result:", result.item())

    # Tiny training loop to confirm autograd + GPU work together
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

    print("Everything works. GPU is doing real compute.")
else:
    print("CUDA not available — check nvidia-smi inside WSL and your torch build.")
```

Run it (venv active):

```bash
cd ~/ml-project
source .venv/bin/activate
python smoke_test.py
```

A clean run with decreasing loss values confirms the full pipeline: driver →
passthrough → CUDA runtime → PyTorch → autograd on the GPU.

---

## 11. JupyterLab in WSL

WSL2 forwards localhost automatically, so a Jupyter server inside Ubuntu is
directly reachable from your Windows browser — **no trick required.**

```bash
cd ~/ml-project
source .venv/bin/activate
pip install jupyterlab
jupyter lab
```

Jupyter prints a URL like:

```
http://localhost:8888/lab?token=abc123...
```

Copy the **entire** URL (token included) into your Windows browser, or
Ctrl-click it in the terminal. The token is the auth step — paste the whole line
and it logs you in.

**Notes:**
- You do **not** need `--ip=0.0.0.0` or `--allow-root`. The default `localhost`
  binding is already reachable from Windows and stays private to your machine.
  Only use `0.0.0.0` if you want *other devices on your network* to connect.
- Your GPU is available in notebooks — first cell: `import torch;
  torch.cuda.is_available()`.
- Port busy? Use `jupyter lab --port=8889` (WSL forwards whatever port it lands
  on).
- **Alternative:** open `.ipynb` files directly in WSL-connected VS Code and
  pick `.venv` as the kernel — no port, no token, GPU still available.

---

## 12. Daily Workflow Cheat Sheet

| Task | Command / Action |
|------|------------------|
| Open Ubuntu | Type `ubuntu` or `wsl`, or pick it in Windows Terminal |
| Go to project home | `cd ~/ml-project` |
| Activate venv | `source .venv/bin/activate` |
| Open in VS Code (from WSL) | `code .` |
| Check GPU | `nvidia-smi` |
| Watch GPU live | `watch -n 1 nvidia-smi` |
| Start JupyterLab | `jupyter lab` |
| Apply `.wslconfig` changes | `wsl --shutdown` (in PowerShell), then reopen |

**Remember:** the `(.venv)` prefix is gone in each new terminal session — just
re-run `source .venv/bin/activate`.

---

## 13. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `nvidia-smi` works in PowerShell but not WSL | Update the Windows NVIDIA driver; never install a driver inside WSL |
| `torch.cuda.is_available()` is `False` | `nvidia-smi` failing in WSL, or a CPU-only torch wheel was installed |
| `CUDA error: no kernel image is available` | torch wheel too old for your GPU — install a current build / latest CUDA wheel |
| Everything is slow | Your code is under `/mnt/c/...`; move it to `~/` |
| `localhost:8888` won't connect (but Jupyter is running) | WSL2 localhost forwarding hiccup after sleep/update — run `wsl --shutdown`, reopen |
| Port already in use | `jupyter lab --port=8889` |
| VS Code opened the Windows drive | You opened the folder before connecting — connect to WSL first |
| WSL eating all RAM | Add memory caps in `C:\Users\<you>\.wslconfig`, then `wsl --shutdown` |
| `git clone` → `Permission denied (publickey)` | Custom-named key not offered — add `~/.ssh/config` with `IdentityFile` (section 9c) |
| `Could not open a connection to your authentication agent` | ssh-agent not running; not needed if you use `~/.ssh/config` (section 9d) |

---

*Setup summary: Windows 11 → WSL2 (Ubuntu) → NVIDIA driver on Windows →
GPU passthrough verified → Python venv → PyTorch + CUDA → VS Code connected to
WSL. GPU-heavy ML work lives in WSL; Windows stays available for everything else.*