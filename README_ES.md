# LLMs y GNNs para Razonamiento Avanzado — Soluciones del Curso

Notebooks de solución para el curso **"LLMs y GNNs para Razonamiento Avanzado sobre Datos Relacionales"**.  
El curso va desde los fundamentos del ML clásico hasta sistemas híbridos que combinan modelos de lenguaje de gran escala con redes neuronales de grafos.

---

## Estructura del curso

| Módulo | Tema | Bibliotecas principales | Labs |
|---|---|---|---|
| **1 — ML** | Exploración de datos y ML clásico sobre el dataset IBM Telco Churn | pandas, scikit-learn, PyTorch | 2 |
| **2 — LLM** | LLMs de código abierto, chatbot local, RAG, profundización en HuggingFace | transformers, sentence-transformers, faiss-cpu, Ollama | 4 |
| **3 — GNN** | Datos de grafos, entrenamiento de GCN, atención con GAT sobre el dataset Cora | torch-geometric, networkx | 3 |
| **4 — Híbrido** | Grafos con atributos de texto, GraphRAG, G-Retriever | construye sobre los módulos 2 + 3 | 3 |

---

## Resumen de los labs

### Módulo 1 — ML Clásico
| Lab | Título | Temas |
|---|---|---|
| `lab1_1` | Exploración de Datos | inspección con pandas, limpieza de datos, desbalance de clases, visualización de características |
| `lab1_2` | Primer Modelo ML | codificación one-hot, división train/val/test, Regresión Logística, matriz de confusión, F1, Árbol de Decisión, MLP (sklearn + PyTorch), regularización |

### Módulo 2 — LLMs y RAG
| Lab | Título | Temas |
|---|---|---|
| `lab2_1` | LLMs de Código Abierto: Primer Contacto | `SimpleLLM`, prompts de sistema, temperatura, alucinación, ventanas de contexto |
| `lab2_2` | Chatbot Local | historial de conversación, chat multi-turno, clase `Chatbot` |
| `lab2_3` | RAG Básico | fragmentación de documentos, índice FAISS, recuperación semántica, generación de respuestas |
| `lab2_4` | HuggingFace en Profundidad | `pipeline()`, tokenización, `AutoModelForCausalLM`, estrategias de generación, embeddings con `SentenceTransformer` |

### Módulo 3 — Redes Neuronales de Grafos
| Lab | Título | Temas |
|---|---|---|
| `lab3_1` | Datos de Grafos | objetos `Data` de PyG, dataset Cora, máscaras train/val/test, visualización de grafos |
| `lab3_2` | Entrenamiento de una GCN | `GCNConv`, bucle de entrenamiento, clasificación de nodos, evaluación |
| `lab3_3` | Atención con GAT | `GATConv`, atención multi-cabeza, visualización de pesos de atención |

### Módulo 4 — Sistemas Híbridos
| Lab | Título | Temas |
|---|---|---|
| `lab4_1` | Grafos con Atributos de Texto | combinación de características textuales con estructura de grafo |
| `lab4_2` | GraphRAG | generación aumentada por recuperación con conciencia de grafos |
| `lab4_3` | G-Retriever | sistema de razonamiento extremo a extremo con grafos + LLM |

---

## Inicio rápido (local)

```bash
# 1 — clonar
git clone https://github.com/DanielFPerez/llm-gnns-course_solutions.git
cd llm-gnns-course_solutions

# 2 — crear y activar un entorno virtual
python3 -m venv .venv
source .venv/bin/activate        # Windows (WSL2): mismo comando

# 3 — instalar todas las dependencias
pip install -r environment/requirements.txt

# 4 — lanzar JupyterLab
jupyter lab
```

Abre la URL que se muestra en la terminal (p. ej. `http://localhost:8888/lab?token=...`) en tu navegador y navega al notebook que deseas ejecutar.

### Guías de configuración por plataforma

Instrucciones detalladas paso a paso para tu sistema operativo:

- **Windows (WSL2 + NVIDIA CUDA)** → [`SETUP_WINDOWS.md`](SETUP_WINDOWS.md)
- **macOS (Apple Silicon / Intel) o Linux nativo** → [`SETUP_MAC_LINUX.md`](SETUP_MAC_LINUX.md)

Ambas guías cubren: dependencias del sistema, controladores GPU, entornos virtuales de Python, verificación de PyTorch, configuración de VS Code + Jupyter, y configuración de Git/SSH.

---

## Ejecución en Google Colab

Cada notebook tiene un badge **Abrir en Colab** en la parte superior. Haz clic para abrirlo y ejecutarlo en un entorno cloud gratuito — sin necesidad de instalación local.

En Colab, la celda de configuración al inicio de cada notebook clona automáticamente el repositorio e instala todas las dependencias. La primera ejecución tarda unos minutos; las siguientes usan la caché de Colab.

> **Consejo GPU:** En Colab, ve a **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4** antes de ejecutar los labs de los Módulos 3 o 4 para obtener aceleración por hardware.

---

## Módulo 2 — Configuración de Ollama (solo local)

Los labs 2.1–2.3 utilizan un wrapper `SimpleLLM` que intenta Ollama primero y recurre automáticamente a un modelo de HuggingFace. Para usar Ollama localmente:

```bash
# Instalar Ollama (Linux / WSL2)
curl -fsSL https://ollama.com/install.sh | sh

# Descargar un modelo
ollama pull llama3.2:1b    # ~1.3 GB, rápido en CPU
ollama pull llama3.2:3b    # ~2.0 GB, mejor calidad

# Iniciar el servidor (deja esta terminal abierta)
ollama serve
```

Si `ollama serve` no está en ejecución, `SimpleLLM()` cambia automáticamente a `HuggingFaceTB/SmolLM2-1.7B-Instruct` de HuggingFace (~3.4 GB, se descarga y almacena en caché en el primer uso).

Consulta **Lab 2.1, Sección 0** para una guía completa sobre cómo interactuar con Ollama desde la terminal, incluida la API REST.

---

## Estructura del repositorio

```
llm-gnns-course_solutions/
├── environment/
│   └── requirements.txt          # todas las dependencias para los cuatro módulos
├── module-1-ml/
│   ├── lab1_1_data_exploration.ipynb
│   └── lab1_2_first_ml_model.ipynb
├── module-2-llm/
│   ├── lab2_1_open_source_llms.ipynb
│   ├── lab2_2_local_chatbot.ipynb
│   ├── lab2_3_basic_rag.ipynb
│   └── lab2_4_huggingface_deep_dive.ipynb
├── module-3-gnn/
│   ├── lab3_1_graph_data.ipynb
│   ├── lab3_2_training_gcn.ipynb
│   └── lab3_3_attention_gat.ipynb
├── module-4-hybrid/
│   ├── lab4_1_text_attributed_graphs.ipynb
│   ├── lab4_2_graphrag.ipynb
│   └── lab4_3_g_retriever.ipynb
├── utils/                        # helpers compartidos importados por todos los notebooks
│   ├── data.py                   # load_telco_churn(), load_company_kb()
│   ├── llm.py                    # SimpleLLM (Ollama primero, fallback a HuggingFace)
│   ├── checks.py                 # validadores de ejercicios (check_dataframe, check_model, …)
│   ├── graph.py                  # helpers de visualización para GNNs
│   └── plotting.py               # utilidades de gráficos para los módulos 1–2
├── SETUP_WINDOWS.md
├── SETUP_MAC_LINUX.md
└── README.md
```

### Paquete `utils/`

| Módulo | Exportaciones principales | Usado en |
|---|---|---|
| `data.py` | `load_telco_churn()`, `load_company_kb()` | Módulos 1, 2 |
| `llm.py` | `SimpleLLM` — `.generate()`, `.chat()`, `.count_tokens()` | Módulo 2 |
| `checks.py` | `check_dataframe`, `check_split`, `check_model`, `check_gnn_model`, `check_graph` | Todos los módulos |
| `graph.py` | `plot_graph`, `plot_embeddings`, `plot_attention_subgraph` | Módulos 3, 4 |
| `plotting.py` | `plot_class_distribution`, `plot_confusion_matrix`, `plot_training_curves`, `plot_feature_importance` | Módulo 1 |

Los datasets se descargan de forma diferida y se almacenan en caché en `~/.llm_gnns_course/data/` en el primer uso.

---

## Dependencias

Todos los paquetes están en `environment/requirements.txt`. Los principales por módulo:

| Módulo | Paquetes principales |
|---|---|
| 1 — ML | `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn` |
| 2 — LLM | `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`, `accelerate`, `datasets` |
| 3 — GNN | `torch-geometric`, `networkx` |
| 4 — Híbrido | sin paquetes nuevos — construye sobre los módulos 2 + 3 |
