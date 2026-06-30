# Mac y Linux: Guía de Configuración para Desarrollo en ML

Una guía completa y ordenada para configurar un entorno de desarrollo de machine learning con aceleración GPU en **macOS** (Apple Silicon o Intel) y **Linux nativo** (Ubuntu/Debian). Elige el camino que corresponda a tu máquina.

> **¿Por qué no hay capa de virtualización?** A diferencia de Windows, Mac y Linux ejecutan el ecosistema de ML de forma nativa. No hay un equivalente a WSL que configurar — trabajas directamente en una terminal.

---

## Tabla de Contenidos

1. [Requisitos Previos](#1-requisitos-previos)
2. [Instalar Dependencias del Sistema](#2-instalar-dependencias-del-sistema)
3. [Drivers de GPU y CUDA (solo Linux con NVIDIA)](#3-drivers-de-gpu-y-cuda-solo-linux-con-nvidia)
4. [Configurar el Entorno de Python](#4-configurar-el-entorno-de-python)
5. [Instalar y Verificar PyTorch](#5-instalar-y-verificar-pytorch)
6. [Configuración de VS Code](#6-configuración-de-vs-code)
7. [Git y SSH (GitHub)](#7-git-y-ssh-github)
8. [Clonar y Ejecutar el Proyecto del Curso](#8-clonar-y-ejecutar-el-proyecto-del-curso)
9. [JupyterLab](#9-jupyterlab)
10. [Guía Rápida de Trabajo Diario](#10-guía-rápida-de-trabajo-diario)
11. [Resolución de Problemas](#11-resolución-de-problemas)

---

## 1. Requisitos Previos

### macOS

- **Mac con Apple Silicon** (M1/M2/M3/M4) — recomendado para ML con aceleración GPU mediante el backend MPS de Apple. La velocidad de entrenamiento es excelente para las cargas de trabajo del curso.
- **Mac con Intel** — funciona para todos los labs; PyTorch corre solo en CPU (sin aceleración GPU). Espera tiempos de entrenamiento más lentos.
- Se recomienda macOS 13 Ventura o posterior.
- [Homebrew](https://brew.sh) instalado (la guía lo instala si no está presente).

### Linux (Ubuntu/Debian)

- Se recomienda Ubuntu 22.04 LTS o 24.04 LTS (otras distribuciones basadas en Debian también funcionan).
- Una **GPU NVIDIA** si quieres aceleración CUDA (opcional — la CPU funciona para todos los labs del curso, solo más lento). El soporte de GPU AMD mediante ROCm es posible pero no está cubierto aquí.
- Privilegios `sudo` para instalar paquetes y drivers.

---

## 2. Instalar Dependencias del Sistema

### macOS

Instala Homebrew si no lo tienes:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Sigue las instrucciones post-instalación que aparecen al final (especialmente la línea `eval` para Apple Silicon que añade Homebrew a tu PATH).

Instala Python y Git:

```bash
brew install python git
```

Verifica las versiones:

```bash
python3 --version   # debería ser 3.11 o 3.12
git --version
```

> **No uses el Python del sistema** en `/usr/bin/python3` — está gestionado por Apple y pip instala en una ruta de solo lectura. El Python de Homebrew en `/opt/homebrew/bin/python3` (Apple Silicon) o `/usr/local/bin/python3` (Intel) es el correcto. Ejecuta `which python3` para confirmarlo.

### Linux (Ubuntu/Debian)

Actualiza el índice de paquetes e instala las herramientas de Python:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv build-essential git
```

Verifica:

```bash
python3 --version   # 3.10+ en Ubuntu 22.04, 3.12+ en 24.04
git --version
```

---

## 3. Drivers de GPU y CUDA (solo Linux con NVIDIA)

> **Usuarios de macOS: omite esta sección completamente.** La aceleración MPS no requiere instalación de drivers — PyTorch se comunica con la GPU a través de los frameworks del sistema de macOS.
>
> **Usuarios de Linux sin GPU NVIDIA:** también omite. El entrenamiento solo en CPU funciona perfectamente para todos los labs del curso.

### 3a. Instalar el driver de NVIDIA

El método más seguro es a través de los repositorios oficiales de Ubuntu:

```bash
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

Tras el reinicio, verifica que el driver está cargado:

```bash
nvidia-smi
```

Deberías ver tu GPU, la versión del driver y un número de versión de CUDA en la esquina superior derecha. **No continúes al siguiente paso hasta que esto funcione.**

> La versión de CUDA que muestra `nvidia-smi` es el *máximo* que soporta el driver — no lo que está instalado. PyTorch incluye su propio runtime de CUDA en el wheel de pip, así que no necesitas instalar el toolkit de CUDA por separado a menos que planees compilar kernels CUDA personalizados.

### 3b. (Opcional) Instalar el toolkit de CUDA para compilación de kernels personalizados

El entrenamiento estándar con PyTorch **no** requiere esto. Omite a menos que necesites `nvcc`.

```bash
# Encuentra la versión de CUDA compatible con tu driver en developer.nvidia.com/cuda-downloads
# Ejemplo para CUDA 12.4 en Ubuntu 22.04:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

Añade `nvcc` a tu PATH añadiendo lo siguiente a `~/.bashrc`:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

---

## 4. Configurar el Entorno de Python

Usa siempre un entorno virtual — aísla los paquetes del proyecto del Python del sistema y evita conflictos de versiones entre proyectos.

```bash
cd ~
mkdir ml-project && cd ml-project
python3 -m venv .venv
source .venv/bin/activate
```

Tu prompt gana el prefijo `(.venv)` cuando el entorno está activo. Debes volver a ejecutar `source .venv/bin/activate` en cada nueva sesión de terminal.

> **Alternativa en macOS — `pyenv`:** Si gestionas múltiples versiones de Python, [pyenv](https://github.com/pyenv/pyenv) (`brew install pyenv`) es una forma más limpia de fijar una versión específica de Python por proyecto. El flujo de trabajo con venv anterior sigue aplicándose encima de él.

---

## 5. Instalar y Verificar PyTorch

### macOS (Apple Silicon — aceleración MPS)

```bash
pip install torch torchvision torchaudio
```

El wheel estándar de pip soporta MPS en Apple Silicon automáticamente — no se necesita URL de índice especial.

**Verifica:**

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

Salida esperada en Apple Silicon:

```
PyTorch: 2.x.x
MPS available: True
MPS tensor sum: <algún número>
```

> **Mac con Intel:** `MPS available: False` es lo esperado — el entrenamiento corre en CPU. Todo el curso sigue funcionando, solo más lento. No se requiere ninguna acción.

### Linux con GPU NVIDIA (aceleración CUDA)

```bash
pip install torch torchvision torchaudio
```

> **Las GPUs más nuevas (p. ej. RTX serie 40/50) necesitan una versión actual de PyTorch.**
> Los wheels más antiguos carecen de los kernels CUDA correctos y fallan con
> `CUDA error: no kernel image is available for execution on the device`.
> El `pip install torch` predeterminado ahora incluye un wheel suficientemente reciente. Si en algún momento fijas una versión, obtén el comando exacto de
> <https://pytorch.org/get-started/locally/> y elige la opción CUDA más reciente.

**Verifica:**

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

### Linux solo CPU

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

`torch.cuda.is_available()` devolverá `False` — es lo esperado.

---

## 6. Configuración de VS Code

VS Code corre de forma nativa tanto en Mac como en Linux — no se necesita extensión remota.

### Instalar VS Code

- **macOS:** `brew install --cask visual-studio-code`, o descárgalo del sitio web de VS Code y arrástralo a Aplicaciones.
- **Linux:** Descarga el `.deb` del sitio web de VS Code, luego:
  ```bash
  sudo dpkg -i code_*.deb
  sudo apt install -f   # corrige cualquier dependencia faltante
  ```

### Abrir el proyecto desde la terminal

```bash
cd ~/ml-project
code .
```

### Extensiones recomendadas

Instálalas desde el panel de Extensiones (`Cmd+Shift+X` / `Ctrl+Shift+X`):

- **Python** (Microsoft) — linting, IntelliSense, depuración
- **Jupyter** (Microsoft) — ejecuta notebooks `.ipynb` directamente en VS Code

### Seleccionar el venv como intérprete de Python

1. Abre cualquier archivo `.py` o `.ipynb`.
2. Haz clic en la versión de Python en la barra de estado inferior derecha (o abre la Paleta de Comandos y ejecuta `Python: Select Interpreter`).
3. Elige el que muestra `.venv` en su ruta, p. ej. `~/ml-project/.venv/bin/python`.

La extensión de Jupyter usa el mismo intérprete como kernel para los notebooks.

---

## 7. Git y SSH (GitHub)

### 7a. Configurar la identidad de Git

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu_email@example.com"
```

### 7b. Crear una clave SSH (omitir si ya tienes una)

```bash
ssh-keygen -t ed25519 -C "tu_email@example.com"
```

Acepta la ubicación predeterminada (`~/.ssh/id_ed25519`) o dale un nombre personalizado. La contraseña es opcional (ver [7d](#7d-¿necesitas-el-ssh-agent) para las ventajas e inconvenientes). Se crean dos archivos: una clave privada (sin extensión) y una clave pública (`.pub`).

### 7c. Añadir la clave pública a GitHub

Imprime la clave pública y copia la línea completa:

```bash
cat ~/.ssh/id_ed25519.pub
```

Empieza con `ssh-ed25519 AAAA...`. Pégala en **GitHub → Settings → SSH and GPG keys → New SSH key**.

> Error común: copiar solo parte de la clave, o pegar la clave **privada**. Copia siempre el contenido completo del archivo `.pub`.

### 7d. Indicar a SSH qué clave usar (necesario para claves con nombre personalizado)

Si usaste un nombre de archivo personalizado, SSH no lo ofrecerá automáticamente. Crea o edita `~/.ssh/config`:

```bash
nano ~/.ssh/config
```

Añade:

```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/nombre-de-tu-clave
  IdentitiesOnly yes
```

Ajusta los permisos:

```bash
chmod 600 ~/.ssh/config
chmod 700 ~/.ssh
chmod 600 ~/.ssh/nombre-de-tu-clave
chmod 644 ~/.ssh/nombre-de-tu-clave.pub
```

`IdentitiesOnly yes` evita `Too many authentication failures` cuando tienes varias claves.

### 7e. ¿Necesitas el ssh-agent?

Con `~/.ssh/config` apuntando a tu clave, SSH la lee desde el disco en cada sesión — **no se necesita agente.** Si tu clave tiene una **contraseña** y quieres que esté en caché para no reescribirla, añade esto a `~/.bashrc` (Linux) o `~/.zshrc` (macOS):

```bash
if [ -z "$SSH_AUTH_SOCK" ]; then
  eval "$(ssh-agent -s)" > /dev/null
  ssh-add ~/.ssh/nombre-de-tu-clave 2>/dev/null
fi
```

> **Bonus en macOS:** puedes guardar la contraseña en el Llavero de macOS para que sobreviva a los reinicios sin necesitar el fragmento del agente:
> ```bash
> ssh-add --apple-use-keychain ~/.ssh/nombre-de-tu-clave
> ```
> Y añade `UseKeychain yes` al bloque `Host github.com` en `~/.ssh/config`.

### 7f. Verificar y clonar

```bash
ssh -T git@github.com
```

Éxito:

```
Hi TuUsuario! You've successfully authenticated, but GitHub does not provide shell access.
```

Luego clona:

```bash
cd ~
git clone git@github.com:TuUsuario/tu-repositorio.git
```

---

## 8. Clonar y Ejecutar el Proyecto del Curso

```bash
cd ~
git clone git@github.com:TuUsuario/llm-gnns-course_solutions.git
cd llm-gnns-course_solutions

# Activa tu venv (o crea uno nuevo aquí)
source ~/.venv/bin/activate   # ajusta la ruta si tu venv está en otro lugar

pip install -r environment/requirements.txt
```

**Prueba de humo** — guarda como `smoke_test.py` y ejecútalo para verificar la pila completa:

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
    print("Backend: CPU (sin aceleración GPU)")

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

Valores de loss decrecientes y sin errores confirman la pipeline completa.

---

## 9. JupyterLab

```bash
cd ~/llm-gnns-course_solutions
source ~/.venv/bin/activate   # si no está ya activo
jupyter lab
```

Jupyter imprime una URL como:

```
http://localhost:8888/lab?token=abc123...
```

Ábrela en tu navegador. El token en la URL es el paso de autenticación — pega la URL completa y accederás directamente.

**Notas:**
- Tu GPU está disponible en los notebooks — primera celda: `import torch; torch.backends.mps.is_available()` (Mac) o `torch.cuda.is_available()` (Linux).
- ¿Puerto ocupado? Usa `jupyter lab --port=8889`.
- **Alternativa:** abre archivos `.ipynb` directamente en VS Code con la extensión Jupyter y selecciona `.venv` como kernel — sin puerto ni token.

---

## 10. Guía Rápida de Trabajo Diario

| Tarea | Comando |
|-------|---------|
| Ir al proyecto | `cd ~/llm-gnns-course_solutions` |
| Activar venv | `source .venv/bin/activate` |
| Abrir en VS Code | `code .` |
| Verificar GPU (Linux/NVIDIA) | `nvidia-smi` |
| Ver GPU en tiempo real (Linux) | `watch -n 1 nvidia-smi` |
| Iniciar JupyterLab | `jupyter lab` |
| Instalar nuevos paquetes | `pip install <paquete>` (con venv activo) |
| Actualizar dependencias | `pip install -r environment/requirements.txt` |

**Recuerda:** el prefijo `(.venv)` desaparece en cada nueva sesión de terminal — vuelve a ejecutar `source .venv/bin/activate` antes de trabajar.

---

## 11. Resolución de Problemas

| Síntoma | Causa probable / solución |
|---------|--------------------------|
| `which python3` muestra `/usr/bin/python3` | Usando el Python del sistema — activa tu venv o verifica tu PATH |
| `torch.backends.mps.is_available()` devuelve `False` en Apple Silicon | macOS < 13, o corriendo en Mac con Intel (esperado) — usa CPU |
| `torch.cuda.is_available()` devuelve `False` en Linux | `nvidia-smi` falla, o se instaló un wheel de torch solo para CPU — verifica el driver primero |
| `CUDA error: no kernel image is available` | El wheel de torch es demasiado antiguo para tu GPU — instala una versión actual |
| `nvidia-smi: command not found` | Driver no instalado o no está en el PATH — repite la sección 3a |
| `Permission denied (publickey)` al hacer `git clone` | Clave SSH con nombre personalizado no ofrecida — añade `~/.ssh/config` con `IdentityFile` (sección 7d) |
| Puerto 8888 ya en uso | `jupyter lab --port=8889` |
| `ModuleNotFoundError` en el notebook | Paquete no instalado en el venv activo — `pip install <paquete>` con el venv activo |
| El kernel de Jupyter muere inmediatamente en Mac | Sin memoria en MPS — reduce el tamaño del batch o cierra otras apps con uso intensivo de GPU |
| `pip install` dice "error: externally-managed-environment" | Estás usando el Python del sistema — activa tu venv primero |

---

*Resumen de la configuración: instala dependencias del sistema (Homebrew/apt) → drivers de GPU si usas Linux + NVIDIA → Python venv → PyTorch (MPS en Apple Silicon, CUDA en Linux/NVIDIA, CPU como alternativa) → VS Code de forma nativa → Git SSH. No se necesita capa de virtualización.*
