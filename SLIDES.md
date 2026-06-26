# Diapositivas — Proyecto DAG Reward Humanoide

> Guion de **3 diapositivas**. Cada bloque `---` es una diapositiva.
> Los bloques **`🖼️ IMAGEN`** indican qué archivo poner, dónde y por qué.
> Todos los archivos de imagen/video están en la raíz del repo.

---

## 🟦 DIAPOSITIVA 1 — El problema y la idea del paper

### Título: "De rodar a caminar: recompensas DAG para un humanoide"

**Encabezado (arriba, texto pequeño):**
- Clase: Robótica Cognitiva — UAEM
- Autores: Alan Cabrera y Enrique Albarrán

**Contenido (lado izquierdo o arriba):**
- **Paper base:** Meng & Xiao (2023), *From Rolling Over to Walking* (arXiv:2303.02581)
- **Tarea:** enseñar a un humanoide de 17 DoF (`HumanoidStandup-v4`, MuJoCo) a
  incorporarse desde el suelo, con **PPO** en una **laptop RTX 4050**.
- **Idea central — recompensa como grafo (DAG) con *achievement triggers*:**
  se encadenan habilidades (`lift → standup → upright`); el bono de una
  habilidad **solo se activa** cuando la anterior supera un *passing score*.
  Esto evita la recompensa dispersa.

```
A_ij,t = max(A_ij,t-1,  R_i,t − PS_ij)        R_t = Σ_i Σ_j  A_ij,t · R_j,t
```

> 🖼️ **DIAGRAMA 1 — `dag_diagram.html`** (se abre en el navegador)
> **Dónde:** centro/abajo de la diapositiva, ancho grande (es el visual principal aquí).
> **Qué muestra:** la estructura del DAG (pelvis_lift → standup → upright + señales auxiliares).
> **Para qué:** explicar visualmente el mecanismo de recompensa encadenada.
> **Cómo usarlo:** ábrelo en el navegador a pantalla completa (F11) y proyéctalo,
> o captura la pantalla para pegarlo como imagen en la diapositiva.

**Pie (texto pequeño):** Adaptación a hardware de consumo: MuJoCo CPU (no Isaac
Gym), 17 DoF (no iCub 32 DoF), ~3–6M steps (no decenas de millones), + CPG,
*action clamping* y representación ego-céntrica.

---

## 🟩 DIAPOSITIVA 2 — El proceso: depurar el reward hasta que el robot se levanta

### Título: "El valor está en el debugging, no solo en el resultado"

**Tabla (lado izquierdo):**

| Iter. | Qué pasó | Lección |
|-------|----------|---------|
| v1 | *passing score* inalcanzable | el bono nunca se activaba |
| v2 | reward de altura **ciego**: `clip(z−0.30)` = gradiente 0 en z∈[0.08, 0.15] | señal densa que es **sparse de facto** |
| **v3** | **fix:** rampa de altura continua → **el robot se sienta** ✅ | medir la cantidad física (`z`), no solo métricas de PPO |
| v4 | `feet_tuck` no escapa la pose en "L" ❌ | un bono tímido no saca de un óptimo local |

**Dato clave (resaltar):** v3 → `z`: 0.15 → **0.45** · 42 % del tiempo con el
torso despegado del suelo · `std` de PPO estable tras corregir hiperparámetros.

> 🖼️ **IMAGEN 2 (la estrella) — `compare_v2_v3.png`**
> **Dónde:** lado derecho o mitad inferior, lo más grande posible.
> **Qué muestra:** ANTES (v2, retorciéndose en el piso) vs DESPUÉS (v3, sentado erguido).
> **Para qué:** es el "¡ah!" visual del proyecto — el antes/después del fix.
>
> 🎬 **VIDEO (opcional, muy recomendado) — `demo_v3_height.mp4`**
> **Dónde:** reproducir en vivo en esta diapositiva (doble clic o insertar como video).
> **Para qué:** ver al robot moverse impacta mucho más que una foto estática.

---

## 🟨 DIAPOSITIVA 3 — Hallazgos, límites y conclusión

### Título: "Lo que aprendimos sobre diseño de recompensas"

**Hallazgos de RL (lado izquierdo, bullets):**
1. **Sparse de facto:** un reward "denso" no sirve si su gradiente está fuera de
   la región donde el agente opera (bug v2, hallado midiendo `z`).
2. **Reward hacking:** el agente explota la señal más fácil (ángulos, pose en "L").
3. **Proxy ≠ objetivo:** en v4 el reward total *subió* mientras la altura *bajó*.
4. Salir de un óptimo local estable exige más que un bono débil: penalización
   activa, más exploración o *curriculum*.

**Limitaciones (texto pequeño):** 1 sola semilla por config · presupuesto de
muestras 2–3 órdenes de magnitud menor que el paper · robot y simulador
distintos · no llega a pararse del todo (techo del presupuesto, no del método).

**Conclusión (resaltar):** se implementó la contribución central del paper (DAG
con *achievement triggers* + CPG + ego-centric) y se llevó al robot de
*acostado* a *sentado erguido* en hardware de consumo, con un proceso de
diagnóstico riguroso documentado iteración por iteración.

> 🖼️ **IMAGEN 3 — `results_v3_height.png`**
> **Dónde:** lado derecho, arriba.
> **Qué muestra:** curvas de entrenamiento v3 — `standup` despega de 0 (evidencia cuantitativa).
> **Para qué:** respaldar con datos que el robot sí aprendió a subir.
>
> 🖼️ **IMAGEN 4 (cierre) — `pose_sequence_v3.png`**
> **Dónde:** lado derecho, abajo (o como banda inferior de ancho completo).
> **Qué muestra:** la secuencia de incorporación del robot (acostado → sentado).
> **Para qué:** cierre visual de la progresión.

**Pie:** Repo: github.com/AlanoHater/robot-ppo · rama `feature/multipath-dag-egocentric`

---

## 📋 Resumen rápido — qué imagen en cada diapositiva

| Diapositiva | Imagen principal | Extra |
|-------------|------------------|-------|
| 1 (idea) | `dag_diagram.html` (navegador) | — |
| 2 (proceso/resultado) | `compare_v2_v3.png` | 🎬 video `demo_v3_height.mp4` en vivo |
| 3 (hallazgos/cierre) | `results_v3_height.png` | `pose_sequence_v3.png` |
