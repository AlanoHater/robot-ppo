# Guion de exposición — Proyecto DAG Reward Humanoide

**Robótica Cognitiva — UAEM · Alan Cabrera y Enrique Albarrán**
**Duración estimada:** ~5–6 min · 3 diapositivas

> Texto pensado para **decirse en voz alta**. Lo que está en *[corchetes]* son
> acciones (señalar, cambiar diapo, reproducir video), no se lee.
> División sugerida de roles (ajústenla a gusto):
> **Alan** → apertura + Diapo 1 · **Enrique** → Diapo 2 · **Alan** → Diapo 3 + cierre.

---

## 🎬 Apertura (Alan · ~20 s)

> "Buenas tardes. Nuestro proyecto se basa en un paper de robótica llamado
> *From Rolling Over to Walking* —de rodar a caminar—, que enseña a robots
> humanoides a desarrollar habilidades motoras complejas. Nosotros tomamos su
> idea central y la implementamos en una laptop normal, para enseñarle a un
> robot simulado a **levantarse del suelo**. Y, sobre todo, vamos a contarles
> el **proceso de depuración**, que es donde más aprendimos."

*[Avanzar a Diapositiva 1]*

---

## 🟦 DIAPOSITIVA 1 — La idea (Alan · ~1:20)

**Mientras se ve el diagrama del DAG (`dag_diagram.html`):**

> "El problema de fondo es la **recompensa dispersa**. Si al robot solo le das
> puntos por estar de pie, nunca aprende: acostado está tan lejos de pararse
> que por azar casi nunca llega, así que casi nunca recibe señal."

> "La solución del paper es partir la tarea en **habilidades encadenadas**, como
> los niveles de un videojuego: no ganas puntos del nivel 2 hasta que pasas el
> nivel 1. A esa estructura se le llama un **DAG** —un grafo dirigido acíclico—."

*[Señalar los nodos de arriba del diagrama]*

> "Aquí cada caja es una habilidad, medida por la altura del torso: primero
> `pelvis_lift`, despegar la pelvis del suelo; luego `standup`, incorporarse; y
> al final `upright`, pararse. Las flechas son 'desbloqueos'."

*[Señalar el número PS sobre una flecha]*

> "Cada flecha tiene una 'nota para pasar', el *passing score*. La recompensa de
> una habilidad **solo empieza a contar cuando la anterior supera su nota**. Y
> una vez desbloqueada, se queda desbloqueada aunque el robot tropiece. Así la
> recompensa lo guía paso a paso, de lo simple a lo complejo."

> "Lo adaptamos a hardware de consumo: simulador en CPU, un robot de 17 grados
> de libertad, y unos pocos millones de pasos, en vez del clúster gigante del
> paper original."

*[Avanzar a Diapositiva 2]*

---

## 🟩 DIAPOSITIVA 2 — El proceso y el resultado (Enrique · ~2:30)

**Mientras se ve la tabla de iteraciones y `compare_v2_v3.png`:**

> "Aquí está lo interesante. Esto no funcionó a la primera: pasamos por varias
> iteraciones, y cada error nos enseñó algo."

*[Señalar la tabla, fila por fila]*

> "En la versión 1, la 'nota para pasar' era inalcanzable, así que el siguiente
> nivel nunca se desbloqueaba."

> "En la versión 2 encontramos el bug más importante. La recompensa de altura
> usaba un umbral: 'te doy puntos si subes más de 0.30'. Pero al **medir la
> altura real** del robot, descubrimos que se movía entre 0.08 y 0.15 — o sea,
> **nunca llegaba al umbral**, y la recompensa valía cero en todo su rango. Una
> señal que parecía densa era, en la práctica, igual de inútil que la dispersa.
> El robot solo se retorcía en el piso."

*[Señalar la imagen izquierda del antes/después]*

> "Este es el 'antes': el robot revolcándose sin lograr nada."

> "El arreglo, en la versión 3, fue cambiar el umbral por una **rampa continua**:
> que cada centímetro que sube dé un poco más de recompensa, desde el suelo. Y
> ahí sí—"

*[Señalar la imagen derecha]*

> "—el robot pasó de revolcarse a **incorporarse y sentarse erguido**. La altura
> subió de 0.15 a 0.45, y pasa el 42 % del tiempo con el torso despegado."

*[Reproducir el video `demo_v3_height.mp4`]*

> "Aquí lo pueden ver en acción: dobla las piernas, arquea la espalda y se
> empuja hacia arriba hasta sentarse."

*[Avanzar a Diapositiva 3]*

---

## 🟨 DIAPOSITIVA 3 — Hallazgos y conclusión (Alan · ~1:30)

**Mientras se ven las curvas (`results_v3_height.png`) y la secuencia:**

> "De todo esto sacamos cuatro lecciones que aplican a cualquier proyecto de
> aprendizaje por refuerzo:"

> "Una: una recompensa 'densa' no sirve de nada si su gradiente está fuera de la
> zona donde el robot realmente se mueve. Eso solo lo vimos **midiendo la
> cantidad física**, no mirando las métricas del algoritmo, que parecían sanas."

> "Dos: el agente siempre explota la señal más fácil —a esto se le llama *reward
> hacking*—. En una versión, hasta encontró una pose de 'sentado en L' que era
> un callejón sin salida."

> "Tres, y muy contraintuitiva: en otra versión la recompensa total **subió**
> mientras la altura real **bajaba**. Optimizar el número no es lo mismo que
> lograr el objetivo."

> "Como limitación honesta: con nuestro presupuesto de cómputo el robot llega a
> sentarse, pero no a pararse del todo. Eso es un límite del hardware, no del
> método: el paper original usa miles de veces más simulaciones."

> "En conclusión: implementamos la contribución central del paper —la recompensa
> en DAG con desbloqueos— y llevamos al robot de estar tendido a sentarse
> erguido en una laptop, documentando cada paso del diagnóstico de forma
> rigurosa."

---

## 🎬 Cierre (Alan · ~15 s)

> "En resumen: el reward shaping es poderoso, pero peligrosamente sensible a los
> detalles; un solo umbral mal puesto puede congelar todo el aprendizaje.
> Gracias, ¿alguna pregunta?"

---

## ❓ Preguntas probables del público (y cómo responder)

**¿Por qué no usaron la GPU si tienen una RTX 4050?**
> "Porque la física de MuJoCo corre en CPU; la GPU solo aceleraría la red
> neuronal, que es pequeña. El cuello de botella es la simulación, no el
> entrenamiento. Por eso paralelizamos en varios núcleos de CPU."

**¿Qué es exactamente un *passing score*?**
> "Es el umbral que una habilidad debe superar para desbloquear la siguiente.
> Mientras el padre no lo pasa, la recompensa del hijo no cuenta."

**¿Por qué no se para del todo?**
> "Dos razones: el presupuesto de cómputo es 2 a 3 órdenes de magnitud menor que
> el del paper, y el robot cae en un óptimo local de 'sentado' del que salir
> requiere mucha más exploración. Lo intentamos en la versión 4 y lo
> documentamos honestamente."

**¿Qué aporta esto sobre un PPO normal?**
> "El PPO normal con una recompensa única sufre la recompensa dispersa. El DAG
> descompone la meta y da señal intermedia, que es justo lo que permite que el
> robot empiece a progresar."

**¿Es su propio algoritmo o el del paper?**
> "El mecanismo de recompensa —el DAG con achievement triggers— es del paper.
> Nuestra contribución es la implementación a escala de laptop, el diagnóstico
> de los bugs de diseño de recompensa, y las figuras/análisis."

---

## 💡 Tips de presentación
- **Practiquen el cambio de turno** (quién dice qué) — que se note coordinado.
- En la Diapo 2, **dejen que el video hable**: pausen de hablar 3–4 segundos
  mientras el robot se mueve.
- Si se ponen nerviosos con la fórmula, **no la lean**: expliquen la idea de
  "niveles que se desbloquean". Nadie quiere ver símbolos leídos en voz alta.
- Tengan el video ya abierto y listo antes de empezar (que no se vea buscarlo).
