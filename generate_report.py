"""
Genera el reporte del proyecto en PDF (3 páginas, español).
Uso:  python generate_report.py
Requiere: reportlab, y las figuras compare_v2_v3.png, results_v3_height.png,
          pose_sequence_v3.png en la misma carpeta.
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

styles = getSampleStyleSheet()
H1  = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=15, spaceAfter=4,
                     textColor=colors.HexColor('#1a3d5c'))
H2  = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=11.5, spaceBefore=7,
                     spaceAfter=3, textColor=colors.HexColor('#2c5f8a'))
SUB = ParagraphStyle('SUB', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER,
                     textColor=colors.HexColor('#555555'), spaceAfter=2)
BODY = ParagraphStyle('BODY', parent=styles['Normal'], fontSize=9.3, leading=12.6,
                      alignment=TA_JUSTIFY, spaceAfter=4)
MONO = ParagraphStyle('MONO', parent=styles['Code'], fontSize=8.5, leading=11,
                      textColor=colors.HexColor('#333333'), backColor=colors.HexColor('#f2f4f7'),
                      borderPadding=4, spaceBefore=2, spaceAfter=4)
CAP = ParagraphStyle('CAP', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER,
                     textColor=colors.HexColor('#666666'), spaceAfter=8)

def P(t): return Paragraph(t, BODY)

def make_table(data, col_widths, header=True):
    t = Table(data, colWidths=col_widths)
    cmds = [
        ('FONTSIZE', (0,0), (-1,-1), 8.4),
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#c0ccd8')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
    ]
    if header:
        cmds += [('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c5f8a')),
                 ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                 ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                 ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#eef2f6')])]
    t.setStyle(TableStyle(cmds))
    return t

story = []

# ============ PÁGINA 1 ============
story.append(Paragraph('Recompensas DAG con <i>Achievement Triggers</i> para un Humanoide', H1))
story.append(Paragraph('Adaptación del paper <i>From Rolling Over to Walking</i> (Meng &amp; Xiao, 2023) '
                       'a hardware de consumo con PPO y MuJoCo', SUB))
story.append(Paragraph('Proyecto de Maestría — Aprendizaje por Refuerzo &nbsp;·&nbsp; '
                       'RTX 4050 Laptop &nbsp;·&nbsp; junio 2026', SUB))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph('1. Introducción y objetivo', H2))
story.append(P(
    'Entrenar a un robot humanoide a incorporarse y caminar desde una posición tendida es un '
    'problema clásico de control con recompensa dispersa: la señal de éxito (estar de pie) está '
    'muy lejos del estado inicial, y una recompensa monolítica rara vez guía al agente hasta ahí. '
    'El paper de Meng &amp; Xiao (2023) propone descomponer la tarea en un <b>grafo dirigido acíclico '
    '(DAG) de habilidades</b> encadenadas mediante <i>achievement triggers</i>. Este proyecto '
    'implementa esa idea central a una escala entrenable en una laptop con GPU de consumo, y '
    'documenta de forma honesta hasta dónde llega el método con ese presupuesto.'))
story.append(P(
    'El objetivo no es reproducir el paper al pie de la letra, sino (a) implementar su mecanismo de '
    'recompensa, (b) entrenarlo sobre <font face="Courier">HumanoidStandup-v4</font> (17 grados de '
    'libertad, MuJoCo) con PPO, y (c) analizar rigurosamente el comportamiento resultante.'))

story.append(Paragraph('2. La idea del paper en una fórmula', H2))
story.append(P(
    'En lugar de una sola recompensa, se definen habilidades (p. ej. <i>lift → standup → upright</i>) '
    'donde el bono de una habilidad <b>solo se activa</b> cuando la anterior supera un umbral '
    '(<i>passing score</i>, PS). El "logro" (<i>achievement</i>) acumula el mejor avance histórico:'))
story.append(Paragraph(
    'A<sub>ij,t</sub> = max( A<sub>ij,t-1</sub> ,&nbsp; R<sub>i,t</sub> &minus; PS<sub>ij</sub> )'
    '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
    'R<sub>t</sub> = &Sigma;<sub>i</sub> &Sigma;<sub>j</sub> &nbsp; A<sub>ij,t</sub> &middot; R<sub>j,t</sub>', MONO))
story.append(P(
    'Así, la recompensa de la habilidad <i>j</i> solo contribuye al total una vez que su habilidad '
    'padre <i>i</i> ha sido "alcanzada", encadenando el aprendizaje de lo simple a lo complejo y '
    'evitando el problema de la recompensa dispersa.'))

story.append(Paragraph('3. Adaptaciones por limitación de hardware', H2))
story.append(make_table([
    ['Paper original', 'Esta implementación'],
    ['Isaac Gym, miles de envs en GPU', 'MuJoCo en CPU, hasta 10 envs en paralelo'],
    ['Robot iCub (32 DoF)', 'HumanoidStandup-v4 (17 DoF)'],
    ['Decenas de millones de steps', '3–6 millones de steps'],
    ['DAG completo hasta caminar', 'DAG de incorporación (lift → standup → upright)'],
], [7.2*cm, 8.6*cm]))
story.append(Spacer(1, 0.15*cm))
story.append(P(
    'Componentes del paper implementados además del DAG: <b>señales CPG</b> (8 senos/cosenos de '
    'frecuencias coprimas concatenados a la observación), <b>action clamping</b> proporcional al '
    'progreso del episodio, y <b>representación ego-céntrica</b> (velocidades rotadas al marco de '
    'referencia de la raíz del robot, verificada contra <font face="Courier">mj_objectVelocity</font>).'))

story.append(PageBreak())

# ============ PÁGINA 2 ============
story.append(Paragraph('4. Metodología y proceso experimental', H2))
story.append(P(
    'El sistema se construyó como un <i>wrapper</i> de Gymnasium que (i) calcula la recompensa de cada '
    'habilidad a partir del estado físico de MuJoCo, (ii) aplica las fórmulas del DAG, y (iii) añade '
    'CPG y transformación ego-céntrica a la observación. Se entrena con PPO (stable-baselines3, red '
    '[400,300,200,100]). El proyecto avanzó por iteraciones, y el mayor valor está en el '
    '<b>proceso de depuración</b>: cada fallo reveló un principio de diseño de recompensas.'))
story.append(make_table([
    ['Iter.', 'Qué ocurrió', 'Lección'],
    ['v1', 'Passing score inalcanzable: el bono nunca se activaba', 'Calibrar PS al rango real de la señal'],
    ['v2', 'Reward de altura "ciego": clip(z−0.30) tiene gradiente 0\nen z∈[0.08,0.15], donde el robot opera',
            'Una señal densa puede ser sparse de facto'],
    ['v3', 'FIX: rampa de altura continua desde el suelo →\nel robot se incorpora y se sienta erguido',
            'Medir la cantidad física (z), no solo métricas de PPO'],
    ['v4', 'feet_tuck + warm-start no escapa la pose en "L"',
            'Un bono débil no saca de un óptimo local estable'],
], [1.1*cm, 8.0*cm, 6.7*cm]))
story.append(Spacer(1, 0.15*cm))
story.append(P(
    'El bug de la iteración v2 fue el hallazgo central. Midiendo la altura real del torso <i>z</i> del '
    'modelo entrenado se descubrió que el robot vivía en <i>z</i>&isin;[0.076, 0.152], pero las tres '
    'señales de altura usaban umbrales (<i>z</i>&gt;0.30, &gt;0.80, &gt;1.00) que valen exactamente '
    'cero —y con gradiente cero— en todo ese rango. Las señales de altura estaban <b>muertas</b>: el '
    'agente solo podía optimizar las señales de ángulo, y las explotaba (<i>reward hacking</i>). La '
    'corrección fue reemplazar los umbrales por <b>rampas continuas</b> con gradiente no nulo desde el '
    'suelo, de modo que cada centímetro ganado en altura diera más recompensa.'))

if os.path.exists('compare_v2_v3.png'):
    story.append(Spacer(1, 0.1*cm))
    story.append(Image('compare_v2_v3.png', width=13.5*cm, height=5.4*cm))
    story.append(Paragraph('Figura 1. Comportamiento antes (v2, recompensa de altura ciega) y después '
                           '(v3, rampa continua) del fix. v2 se retuerce en el piso; v3 se incorpora.', CAP))

story.append(PageBreak())

# ============ PÁGINA 3 ============
story.append(Paragraph('5. Resultados', H2))
story.append(P(
    'La corrección del reward (v3) produjo un cambio cualitativo: el robot pasó de retorcerse acostado '
    'a hacer un esfuerzo coordinado de incorporarse hasta sentarse erguido. La iteración v4, que '
    'intentó llevarlo de sentado a de pie añadiendo una señal de "recoger los pies", no lo logró: la '
    'pose de "sentado en L" resultó ser un óptimo local demasiado estable.'))
story.append(make_table([
    ['Métrica', 'v2 (reward roto)', 'v3 (rampa altura)', 'v4 (feet_tuck)'],
    ['Recompensa media por episodio', '~67 (plana)', '~850 (sube)', '~900'],
    ['Altura z máxima', '0.152', '0.450', '0.429'],
    ['Altura z promedio', '0.116', '0.291', '0.209'],
    ['% del tiempo despegado (z>0.30)', '0 %', '42 %', '—'],
    ['% del tiempo de pie (z>0.80)', '0 %', '0 %', '0 %'],
    ['std de PPO (estabilidad)', '4.4 (explota)', '0.94', '0.93'],
], [6.0*cm, 3.3*cm, 3.3*cm, 3.2*cm]))
story.append(Spacer(1, 0.1*cm))
if os.path.exists('results_v3_height.png'):
    story.append(Image('results_v3_height.png', width=14.5*cm, height=6.4*cm))
    story.append(Paragraph('Figura 2. Curvas de v3. Izq.: recompensa máxima por habilidad — pelvis_lift '
                           'satura en 1.0 y standup despega de 0. Der.: recompensa total por episodio, '
                           'crecimiento sostenido de ~100 a ~870.', CAP))

story.append(Paragraph('6. Hallazgos y lecciones', H2))
story.append(P(
    '<b>(1) Sparse de facto:</b> una recompensa "densa" no guía si su gradiente está fuera de la región '
    'donde el agente opera. <b>(2) Reward hacking:</b> el agente explota sistemáticamente la señal más '
    'fácil (ángulos, pose en L) antes que el objetivo. <b>(3) Proxy &ne; objetivo:</b> en v4 la '
    'recompensa total subió mientras la altura real bajó. <b>(4)</b> Escapar un óptimo local estable '
    'requiere más que un bono positivo débil: penalización activa, mayor exploración o un curriculum.'))

story.append(Paragraph('7. Limitaciones y conclusión', H2))
story.append(P(
    '<b>Limitaciones:</b> una sola semilla por configuración (sin validación estadística); presupuesto '
    'de muestras 2–3 órdenes de magnitud menor que el del paper; robot y simulador distintos; el robot '
    'no llega a pararse del todo —un límite del presupuesto de cómputo, no del método—.'))
story.append(P(
    '<b>Conclusión:</b> se implementó la contribución central del paper (recompensa DAG con '
    '<i>achievement triggers</i>, junto con CPG, <i>action clamping</i> y representación ego-céntrica) '
    'y se llevó al humanoide de <i>tendido</i> a <i>sentado erguido</i> en hardware de consumo. Más '
    'allá del resultado, el proyecto aporta un proceso de diagnóstico riguroso que expone errores '
    'comunes (y sutiles) del diseño de recompensas en aprendizaje por refuerzo.'))

story.append(Paragraph('Referencia', H2))
story.append(Paragraph(
    'Meng, Y., &amp; Xiao, J. (2023). <i>From Rolling Over to Walking: Enabling Humanoid Robots to '
    'Develop Complex Motor Skills</i>. arXiv:2303.02581. &nbsp; Código del proyecto: '
    'github.com/AlanoHater/robot-ppo (rama feature/multipath-dag-egocentric).',
    ParagraphStyle('REF', parent=BODY, fontSize=8.4, textColor=colors.HexColor('#444444'))))

doc = SimpleDocTemplate('reporte_proyecto.pdf', pagesize=A4,
                        topMargin=1.4*cm, bottomMargin=1.3*cm,
                        leftMargin=1.7*cm, rightMargin=1.7*cm,
                        title='Reporte - DAG Reward Humanoide')
doc.build(story)
print('[OK] reporte_proyecto.pdf generado')
