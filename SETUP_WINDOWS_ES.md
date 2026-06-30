# WSL2 + CUDA + VS Code: Guía de Configuración para Desarrollo en ML

Una guía completa y ordenada para configurar un entorno de desarrollo de machine learning con aceleración GPU en Windows usando WSL2 (Ubuntu), NVIDIA CUDA y VS Code.

> **¿Por qué WSL2?** El ecosistema de ML (PyTorch, JAX, la mayoría de herramientas de investigación) es Linux-first. WSL2 te proporciona un entorno Linux casi nativo con acceso completo a la GPU, mientras te permite seguir usando Windows para todo lo demás. Evita la fricción recurrente de los pasos de compilación exclusivos de Windows y los wheels de Linux faltantes.

---

## Tabla de Contenidos

1. [Requisitos Previos](#1-requisitos-previos)
2. [Instalar WSL2](#2-instalar-wsl2)
3. [Instalar el Driver de NVIDIA (solo en Windows)](#3-instalar-el-driver-de-nvidia-solo-en-windows)
4. [Verificar el Acceso a la GPU desde WSL](#4-verificar-el-acceso-a-la-gpu-desde-wsl)
5. [Configurar el Entorno de Python](#5-configurar-el-entorno-de-python)
6. [Instalar y Verificar PyTorch](#6-instalar-y-verificar-pytorch)
7. [Configuraciones Importantes de WSL](#7-configuraciones-importantes-de-wsl)
8. [VS Code + WSL](#8-vs-code--wsl)
9. [Git y SSH (GitHub)](#9-git-y-ssh-github)
10. [Ejecutar el Proyecto de Ejemplo](#10-ejecutar-el-proyecto-de-ejemplo)
11. [JupyterLab en WSL](#11-jupyterlab-en-wsl)
12. [Guía Rápida de Trabajo Diario](#12-guía-rápida-de-trabajo-diario)
13. [Resolución de Problemas](#13-resolución-de-problemas)

---

## 1. Requisitos Previos

- **Windows 11** (recomendado) o Windows 10 versión 21H2+.
  Nota: Windows 11 reporta su versión con el prefijo `10.0` (p. ej. build
  `10.0.26200`) por razones de compatibilidad — sigue siendo Windows 11.
- Una **GPU NVIDIA** con permisos de administrador para instalar drivers.
- Conexión de banda ancha (descargarás varios GB).

> **La regla de oro para evitar la mayoría de los problemas:** Instala exactamente **un** driver de GPU, y va en el lado de **Windows**. El lado de Linux recibe el *toolkit/runtime* de CUDA, nunca el driver. Instalar un driver de NVIDIA dentro de WSL rompe el acceso a la GPU.

---

## 2. Instalar WSL2

Abre **PowerShell como Administrador** y ejecuta:

```powershell
wsl --install
```

Esto habilita las funcionalidades necesarias, establece WSL2 como predeterminado e instala Ubuntu. **Reinicia** cuando se te indique. Al iniciar por primera vez, Ubuntu te pedirá crear un nombre de usuario y contraseña UNIX (distintos de tu cuenta de Windows).

> Si ves `A distribution with the supplied name already exists`, significa que Ubuntu **ya está instalado** — no hay ningún problema, continúa al siguiente paso.

Asegúrate de que todo esté actualizado (el acceso a la GPU requiere un kernel reciente):

```powershell
wsl --update
wsl --version
```

También verifica **Configuración → Windows Update** — el kernel de WSL se actualiza mediante Windows Update, y necesitas la versión **5.10.43.3 o superior** para las funciones de GPU.

Confirma que tu distribución está en la versión 2:

```powershell
wsl -l -v
```

Deberías ver `Ubuntu` con `VERSION` mostrando `2`. (Si muestra `1`, conviértelo con `wsl --set-version Ubuntu 2`.)

**Para abrir Ubuntu más adelante**, cualquiera de estas opciones funciona:
- Pulsa la tecla Windows, escribe `ubuntu`, Enter.
- Haz clic en el desplegable `˅` de Windows Terminal y elige **Ubuntu**.
- Escribe `wsl` en cualquier ventana de PowerShell/CMD (escribe `exit` para volver).

Sabrás que estás en Linux cuando el prompt cambie de `PS C:\...>` a `username@machine:~$`.

---

## 3. Instalar el Driver de NVIDIA (solo en Windows)

Descarga el driver estándar **Game Ready** o **Studio** para tu GPU desde [nvidia.com](https://www.nvidia.com/Download/index.aspx) e instálalo en Windows normalmente. Ese único driver se expone automáticamente en WSL2 a través de `/usr/lib/wsl/lib/`.

**NO** ejecutes nada como `sudo apt install nvidia-driver-xxx` dentro de Ubuntu. Eso instala un driver de Linux en conflicto y rompe el acceso a la GPU.

Verifica el lado de Windows primero. En **PowerShell**:

```powershell
nvidia-smi
```

Deberías ver tu GPU, la versión del driver y una "CUDA Version" en la esquina superior derecha. Ese número de CUDA es el **máximo** que soporta el driver — no algo que hayas instalado — así que no te sorprendas si es mayor de lo esperado.

---

## 4. Verificar el Acceso a la GPU desde WSL

Este es el punto crítico para el ML. Abre **Ubuntu** y ejecuta el mismo comando *dentro de Linux*:

```bash
nvidia-smi
```

El prompt debería ser como `dpere@kukulkan:~$` (Linux), **no** `PS C:\...` (Windows). La salida debería mostrar la **misma versión del driver** y tu GPU.

> Una pequeña diferencia de versión entre el `nvidia-smi` de Windows y el de WSL (p. ej. `610.47` vs `610.43.02`) es normal — es solo la versión de la herramienta en espacio de usuario. La versión del driver del kernel (KMD) es la que importa y coincidirá.

**No continúes hasta que `nvidia-smi` funcione dentro de Ubuntu.** Todo lo demás depende de esto. Si dice `command not found`, actualiza tu driver de NVIDIA en Windows.

---

## 5. Configurar el Entorno de Python

Dentro de Ubuntu, actualiza el sistema e instala las herramientas de Python:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv build-essential
```

(Pedirá la contraseña UNIX que creaste durante la configuración de Ubuntu.)

Crea tu proyecto con un entorno virtual aislado. Ubuntu moderno bloquea las instalaciones globales de pip, por lo que un venv es el camino estándar:

```bash
mkdir ~/ml-project && cd ~/ml-project
python3 -m venv .venv
source .venv/bin/activate
```

Sabrás que el venv está activo cuando tu prompt tenga el prefijo `(.venv)`.

> **Regla crítica del sistema de archivos:** Mantén todo el código bajo tu home de Linux (`~/`), **no** en la unidad de Windows en `/mnt/c/...`. El acceso entre sistemas de archivos es dramáticamente más lento y es la causa #1 del "¿por qué todo va tan lento?". Si tu prompt alguna vez muestra `/mnt/c/...`, ejecuta `cd ~` para volver a casa.
> Para acceder a los archivos de WSL *desde Windows*, abre `\\wsl$\Ubuntu\home\<tu-usuario>` en el Explorador.

---

## 6. Instalar y Verificar PyTorch

Para el trabajo estándar con PyTorch **no** necesitas el toolkit CUDA a nivel de sistema — los wheels de pip incluyen su propio runtime de CUDA. Solo necesitas el driver de Windows (ya instalado) más PyTorch.

Con tu venv activo:

```bash
pip install torch torchvision torchaudio
```

> **Las GPUs más nuevas (p. ej. RTX serie 50 / Blackwell) necesitan una versión actual de PyTorch.**
> Los wheels más antiguos carecen de los kernels CUDA correctos y fallan con
> `CUDA error: no kernel image is available for execution on the device`.
> El `pip install torch` predeterminado ahora incluye un wheel suficientemente reciente. Si en algún momento fijas una versión, obtén el comando exacto de
> <https://pytorch.org/get-started/locally/> y elige la opción CUDA más reciente.

**Verifica la pila completa:**

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Salida esperada:

```
True
NVIDIA GeForce RTX 5060 Laptop GPU
```

`True` más el nombre de tu GPU significa que CUDA está conectado de extremo a extremo. Si obtienes `False`, la causa habitual es que `nvidia-smi` no funcionaba en el paso 4, o se instaló un wheel solo de CPU.

**¿Cuándo necesitas el toolkit CUDA completo (`nvcc`)?** Solo si **compilas código CUDA personalizado** (p. ej. compilando flash-attention desde el código fuente). En ese caso, instala **solo** el metapaquete del toolkit del repositorio WSL de NVIDIA — selecciona **"WSL-Ubuntu"** como distribución en la [página de descargas de CUDA](https://developer.nvidia.com/cuda-downloads), y nunca los metapaquetes `cuda`, `cuda-drivers`, o `cuda-12-x`/`cuda-13-x` completos (intentan instalar un driver de Linux). El entrenamiento estándar con PyTorch **no** requiere esto.

---

## 7. Configuraciones Importantes de WSL

### 7a. Limitar el uso de recursos de WSL2 (`.wslconfig`)

Sin configurar, WSL2 puede disparar su uso de RAM. Crea un archivo en `C:\Users\<TuNombre>\.wslconfig` (en el lado de **Windows**) con algo como:

```ini
[wsl2]
memory=24GB
processors=8
swap=8GB
```

Ajusta estos valores a tu máquina y **deja margen para Windows**. Aplica los cambios ejecutando `wsl --shutdown` en PowerShell y luego vuelve a abrir Ubuntu.

### 7b. Recuerda que la VRAM es compartida

La VRAM de tu GPU se comparte con lo que Windows está renderizando. En una tarjeta de 8 GB, cierra las apps con uso intensivo de GPU antes de entrenamientos grandes. Monitoriza el uso en tiempo real en una segunda pestaña de Ubuntu:

```bash
watch -n 1 nvidia-smi
```

### 7c. Trabajar con un presupuesto de 8 GB de VRAM

8 GB son geniales para aprender, fine-tuning de modelos pequeños y experimentación, pero ajustado para entrenamientos más grandes. Apóyate en: tamaños de batch más pequeños, precisión mixta (`bf16`/`fp16`), gradient checkpointing y cuantización (carga de 8 bits/4 bits con `bitsandbytes`).

### 7d. Mantén los proyectos en el sistema de archivos de Linux

(Se repite porque es lo más importante.) El código vive en `~/`, no en `/mnt/c/...`.

---

## 8. VS Code + WSL

**No instales un segundo VS Code dentro de Ubuntu.** Mantén tu VS Code de Windows y conéctalo a WSL. La interfaz gráfica corre en Windows; un servidor ligero corre dentro de WSL y gestiona tu terminal, extensiones, servidores de lenguaje y archivos en el lado de Linux.

### Configuración inicial

1. En VS Code de Windows, abre el panel de Extensiones e instala la extensión **WSL** (la oficial de Microsoft, antes llamada "Remote - WSL").

### Opción A — lanzar desde la terminal de WSL (más rápido)

```bash
cd ~/ml-project
code .
```

La primera vez, esto instala automáticamente el servidor en Linux y abre una ventana conectada.

### Opción B — iniciar desde VS Code de Windows

El orden importa: **conéctate a WSL primero, luego abre la carpeta.**

1. Abre VS Code en Windows.
2. Conéctate mediante cualquiera de estas opciones: Paleta de Comandos (`Ctrl+Shift+P` → `WSL: Connect to WSL`), el indicador azul/verde `><` en la esquina inferior izquierda, o el Explorador Remoto.
3. Una vez que el badge inferior izquierdo diga **WSL: Ubuntu**, abre tu proyecto con **Archivo → Abrir Carpeta** (`Ctrl+K Ctrl+O`). Como ya estás conectado, el selector muestra el sistema de archivos de **Linux** — navega hasta `/home/dpere/ml-project`.

> Si abres la carpeta *antes* de conectarte, el selector muestra la unidad `C:\` de Windows y terminarás en la ruta lenta `\\wsl$\`. Conéctate primero.

Después de abrir un proyecto WSL una vez, aparece en **Archivo → Abrir Reciente** marcado con `[WSL: Ubuntu]` para reconexión con un clic.

### Confirma que estás conectado correctamente

- El badge inferior izquierdo muestra **WSL: Ubuntu**.
- Abre una terminal con `` Ctrl+` `` — el prompt debería ser `dpere@kukulkan:~/ml-project$`, no PowerShell.
- Instala las extensiones funcionales (Python, Jupyter) **en WSL** cuando se te pida (VS Code ofrece un botón "Install in WSL"). Los temas pueden quedarse en Windows.

---

## 9. Git y SSH (GitHub)

Para clonar tus propios repositorios por SSH, necesitas una clave SSH en WSL cuya mitad pública esté registrada en GitHub.

### 9a. Crear una clave (omitir si ya tienes una)

```bash
ssh-keygen -t ed25519 -C "tu_email@example.com"
```

Pulsa Enter para aceptar la ubicación predeterminada, o dale un nombre personalizado. La contraseña es opcional (ver [9d](#9d-¿necesitas-el-ssh-agent) para las ventajas e inconvenientes). Esto produce dos archivos en `~/.ssh/`: una clave privada (sin extensión) y una clave pública (`.pub`).

### 9b. Añadir la clave pública a GitHub

Imprime la clave **pública** y copia la línea completa:

```bash
cat ~/.ssh/nombre-de-tu-clave.pub
```

Empieza con `ssh-ed25519 AAAA...`. Pégala en **GitHub → Settings → SSH and GPG keys → New SSH key**. Asegúrate de añadirla a la cuenta de GitHub *correcta* (la que posee los repositorios que clonarás).

> Error común: copiar solo parte de la clave, o pegar la clave **privada**. Copia siempre el contenido completo del archivo `.pub`.

### 9c. Indicar a SSH qué clave usar (la solución fiable)

**Esto es esencial si tu clave tiene un nombre personalizado.** SSH solo ofrece automáticamente las claves con nombres predeterminados (`id_ed25519`, `id_rsa`); una clave con nombre personalizado se ignora a menos que apuntes SSH a ella explícitamente. Esta es la causa #1 de `Permission denied (publickey)` incluso cuando la clave está correctamente añadida a GitHub.

Crea o edita `~/.ssh/config`:

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

Reemplaza `nombre-de-tu-clave` con el nombre real de tu archivo de clave **privada** (sin `.pub`). Luego ajusta los permisos — SSH ignora las configuraciones y claves con permisos demasiado abiertos:

```bash
chmod 600 ~/.ssh/config
chmod 700 ~/.ssh
chmod 600 ~/.ssh/nombre-de-tu-clave
chmod 644 ~/.ssh/nombre-de-tu-clave.pub
```

La línea `IdentitiesOnly yes` hace que SSH ofrezca *solo* esta clave para GitHub, lo que también evita errores de `Too many authentication failures` cuando tienes varias claves.

### 9d. ¿Necesitas el ssh-agent?

Con el `~/.ssh/config` anterior, SSH lee la clave directamente desde el disco en cada sesión — **no se necesita agente ni `ssh-add`.** Recurre al agente solo si tu clave tiene una **contraseña** y no quieres reescribirla en cada sesión. (Si alguna vez ves `Could not open a connection to your authentication agent`, significa que el agente no está corriendo — pero con la configuración en su lugar no lo necesitas.)

Si *sí* quieres que el agente se inicie automáticamente y cargue una clave protegida por contraseña, añade esto al final de `~/.bashrc`:

```bash
if [ -z "$SSH_AUTH_SOCK" ]; then
  eval "$(ssh-agent -s)" > /dev/null
  ssh-add ~/.ssh/nombre-de-tu-clave 2>/dev/null
fi
```

### 9e. Verificar y clonar

Prueba la conexión:

```bash
ssh -T git@github.com
```

El éxito se ve así:

```
Hi TuUsuario! You've successfully authenticated, but GitHub does not provide shell access.
```

(La parte "does not provide shell access" es **esperada** — no es un error.) Confirma que el nombre de usuario corresponde a la cuenta que posee el repositorio. Luego clona:

```bash
cd ~
git clone git@github.com:TuUsuario/tu-repositorio.git
```

> **Recordatorio del sistema de archivos:** clona en tu home de Linux (`~/`), no en `/mnt/c/...`, por las mismas razones de rendimiento que todo lo demás.

---

## 10. Ejecutar el Proyecto de Ejemplo

Una base de referencia que verifica toda la pila (detección de GPU + cómputo real en GPU). Guarda como `~/ml-project/smoke_test.py`:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    # Empujar tensores a la GPU y hacer matemáticas reales
    x = torch.randn(10000, 10000, device=device)
    result = (x @ x).sum()
    torch.cuda.synchronize()
    print("GPU matmul result:", result.item())

    # Bucle de entrenamiento mínimo para confirmar que autograd + GPU funcionan juntos
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

Ejecútalo (con el venv activo):

```bash
cd ~/ml-project
source .venv/bin/activate
python smoke_test.py
```

Una ejecución limpia con valores de loss decrecientes confirma la pipeline completa: driver → passthrough → runtime de CUDA → PyTorch → autograd en la GPU.

---

## 11. JupyterLab en WSL

WSL2 redirige localhost automáticamente, así que un servidor Jupyter dentro de Ubuntu es directamente accesible desde tu navegador de Windows — **sin ningún truco especial.**

```bash
cd ~/ml-project
source .venv/bin/activate
pip install jupyterlab
jupyter lab
```

Jupyter imprime una URL como:

```
http://localhost:8888/lab?token=abc123...
```

Copia la URL **completa** (incluyendo el token) en tu navegador de Windows, o haz Ctrl+clic en la terminal. El token es el paso de autenticación — pega toda la línea y te iniciará sesión.

**Notas:**
- **No** necesitas `--ip=0.0.0.0` ni `--allow-root`. El enlace predeterminado a `localhost` ya es accesible desde Windows y permanece privado en tu máquina. Usa `0.0.0.0` solo si quieres que *otros dispositivos de tu red* se conecten.
- Tu GPU está disponible en los notebooks — primera celda: `import torch; torch.cuda.is_available()`.
- ¿Puerto ocupado? Usa `jupyter lab --port=8889` (WSL redirige cualquier puerto que use).
- **Alternativa:** abre archivos `.ipynb` directamente en VS Code conectado a WSL y elige `.venv` como kernel — sin puerto, sin token, la GPU sigue disponible.

---

## 12. Guía Rápida de Trabajo Diario

| Tarea | Comando / Acción |
|-------|-----------------|
| Abrir Ubuntu | Escribe `ubuntu` o `wsl`, o elige en Windows Terminal |
| Ir al home del proyecto | `cd ~/ml-project` |
| Activar venv | `source .venv/bin/activate` |
| Abrir en VS Code (desde WSL) | `code .` |
| Verificar GPU | `nvidia-smi` |
| Ver GPU en tiempo real | `watch -n 1 nvidia-smi` |
| Iniciar JupyterLab | `jupyter lab` |
| Aplicar cambios de `.wslconfig` | `wsl --shutdown` (en PowerShell), luego reabrir |

**Recuerda:** el prefijo `(.venv)` desaparece en cada nueva sesión de terminal — vuelve a ejecutar `source .venv/bin/activate`.

---

## 13. Resolución de Problemas

| Síntoma | Causa probable / solución |
|---------|--------------------------|
| `nvidia-smi` funciona en PowerShell pero no en WSL | Actualiza el driver de NVIDIA en Windows; nunca instales un driver dentro de WSL |
| `torch.cuda.is_available()` devuelve `False` | `nvidia-smi` falla en WSL, o se instaló un wheel de torch solo para CPU |
| `CUDA error: no kernel image is available` | El wheel de torch es demasiado antiguo para tu GPU — instala una versión actual |
| Todo va lento | Tu código está en `/mnt/c/...`; muévelo a `~/` |
| `localhost:8888` no conecta (pero Jupyter está corriendo) | Problema de redirección de localhost en WSL2 tras suspensión/actualización — ejecuta `wsl --shutdown` y vuelve a abrir |
| Puerto ya en uso | `jupyter lab --port=8889` |
| VS Code abrió la unidad de Windows | Abriste la carpeta antes de conectarte — conéctate a WSL primero |
| WSL consume toda la RAM | Añade límites de memoria en `C:\Users\<tu-usuario>\.wslconfig`, luego `wsl --shutdown` |
| `git clone` → `Permission denied (publickey)` | Clave con nombre personalizado no ofrecida — añade `~/.ssh/config` con `IdentityFile` (sección 9c) |
| `Could not open a connection to your authentication agent` | ssh-agent no está corriendo; no es necesario si usas `~/.ssh/config` (sección 9d) |

---

*Resumen de la configuración: Windows 11 → WSL2 (Ubuntu) → driver de NVIDIA en Windows → acceso a GPU verificado → Python venv → PyTorch + CUDA → VS Code conectado a WSL. El trabajo de ML intensivo en GPU vive en WSL; Windows queda disponible para todo lo demás.*
