"""Genera reports/Propuesta_TraficoIA.pdf (2 a 3 páginas).

Edita los textos de este archivo (equipo, integrantes, hallazgos) y vuelve a ejecutar:
    python docs/generar_propuesta.py
Requiere reportlab (pip install reportlab).
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (ListFlowable, ListItem, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

SALIDA = Path(__file__).resolve().parents[1] / "reports" / "Propuesta_TraficoIA.pdf"
AZUL = colors.HexColor("#1d4e89")

# ---------------- Datos del equipo (EDITAR) ----------------
EQUIPO = "N.º ____"
MATERIA = "______________________"
DOCENTE = "______________________"
INTEGRANTES = ["Integrante 1: ______________ (coordinación; lidera supervisado)",
               "Integrante 2: ______________ (datos y EDA; lidera no supervisado)",
               "Integrante 3: ______________ (simulación; lidera refuerzo)",
               "Integrante 4: ______________ (interfaz y repositorio; lidera CNN)"]

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=13,
                    textColor=AZUL, spaceBefore=10, spaceAfter=4)
TIT = ParagraphStyle("TIT", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20,
                     textColor=AZUL, spaceAfter=2)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontName="Helvetica", fontSize=10.5,
                     textColor=colors.HexColor("#555555"), alignment=1, spaceAfter=8)
P = ParagraphStyle("P", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6,
                   leading=13, alignment=TA_JUSTIFY, spaceAfter=4)
CELDA = ParagraphStyle("CELDA", parent=P, fontSize=8.6, leading=11, alignment=0, spaceAfter=0)
CELDAB = ParagraphStyle("CELDAB", parent=CELDA, fontName="Helvetica-Bold", textColor=colors.white)


def tabla(filas, anchos):
    datos = [[Paragraph(c, CELDAB if i == 0 else CELDA) for c in f] for i, f in enumerate(filas)]
    t = Table(datos, colWidths=[a * cm for a in anchos], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), AZUL),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b7c3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def viñetas(items):
    return ListFlowable([ListItem(Paragraph(i, P), leftIndent=12) for i in items],
                        bulletType="bullet", start="•", leftIndent=12)


def pie(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.2 * cm, "TráficoIA · Propuesta del equipo · Proyecto Final, versión 3")
    canvas.drawRightString(letter[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
    canvas.restoreState()


def construir():
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(SALIDA), pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title="Propuesta TráficoIA", author="Equipo " + EQUIPO)
    e = []
    e += [Paragraph("TráficoIA", TIT),
          Paragraph("Propuesta del equipo: predicción de tráfico, señales de tránsito y control de semáforos", SUB)]
    e.append(tabla([["Materia", "Docente", "Equipo"], [MATERIA, DOCENTE, EQUIPO]], [6.5, 6.5, 4]))
    e.append(Spacer(1, 4))
    e.append(viñetas(INTEGRANTES))

    e.append(Paragraph("1. Problema y justificación", H1))
    e.append(Paragraph(
        "La congestión urbana y las señales de tránsito mal interpretadas generan retrasos y accidentes. "
        "Un sistema de apoyo a la movilidad necesita responder cuatro preguntas distintas: <b>¿cuánto tráfico "
        "habrá?</b>, <b>¿qué tipos de día existen?</b>, <b>¿qué señal tengo enfrente?</b> y <b>¿cómo conviene "
        "operar el semáforo?</b>. TráficoIA es un prototipo que integra un módulo de aprendizaje automático para "
        "cada pregunta: aprendizaje supervisado, no supervisado, una red neuronal convolucional (CNN) y "
        "aprendizaje por refuerzo. Es un prototipo académico: se prioriza que funcione, que se entienda y que "
        "se explique, no la precisión de un producto profesional.", P))

    e.append(Paragraph("2. Objetivos", H1))
    e.append(Paragraph("<b>General.</b> Construir en Python, con un entorno de Anaconda, un prototipo que integre "
                       "los cuatro enfoques para apoyar la gestión del tráfico en una intersección.", P))
    e.append(viñetas([
        "Predecir el nivel de tráfico (bajo, medio, alto) con al menos tres algoritmos supervisados comparados.",
        "Agrupar días por su perfil horario y proponer un horario de semáforo para cada tipo de día.",
        "Reconocer 12 señales de tránsito con una CNN propia entrenada en PyTorch.",
        "Entrenar un agente de Q-learning que reduzca la espera promedio frente a un semáforo de tiempo fijo.",
        "Integrar los módulos en una aplicación Streamlit demostrable y documentada.",
    ]))

    e.append(Paragraph("3. Datos", H1))
    e.append(tabla([
        ["Componente", "Conjunto de datos", "Descripción y licencia"],
        ["Supervisado y no supervisado", "Metro Interstate Traffic Volume (UCI, #492)",
         "Volumen horario de tráfico en la autopista I-94 (sentido oeste, Minnesota, EE. UU.), 2012 a 2018, con "
         "clima y festivos. Licencia CC BY 4.0 (confirmar en la página del dataset antes de entregar)."],
        ["CNN", "GTSRB, señales alemanas (43 clases)",
         "Más de 39 000 imágenes de entrenamiento. Se usarán 12 clases: cinco límites de velocidad (30, 50, 60, 70 "
         "y 80 km/h), ceda el paso, alto, entrada prohibida, precaución general, obras, niños y mantenga la derecha. "
         "Condiciones de uso: ver página oficial y citar a Stallkamp et al. (2012)."],
        ["Refuerzo", "Cruce simulado por el equipo",
         "Entorno propio de Gymnasium; no requiere dataset."],
    ], [3.4, 4.6, 9.0]))
    e.append(Spacer(1, 3))
    e.append(Paragraph(
        "Las 12 clases incluyen cinco límites de velocidad a propósito: son visualmente muy parecidos y permiten "
        "un análisis de errores más interesante en la matriz de confusión.", P))

    e.append(Paragraph("4. Hallazgos preliminares del EDA", H1))
    e.append(Paragraph(
        "<i>[Completar con 3 a 4 hallazgos de 01_eda.ipynb que cambien decisiones: calidad de datos, patrón horario, "
        "efecto de festivos y clima, desbalance de imágenes. Cada hallazgo debe terminar con la decisión que se tomó.]</i>", P))

    e.append(Paragraph("5. Metodología y métricas por módulo", H1))
    e.append(tabla([
        ["Módulo", "Tarea y método", "Evaluación"],
        ["Supervisado", "Clasificar nivel de tráfico (terciles del volumen, calculados solo con entrenamiento) "
         "con hora, día, temperatura, lluvia y festivo. Árbol de decisión, Random Forest y KNN.",
         "80 % entrenamiento / 20 % prueba; validación cruzada de 5 particiones. Exactitud, F1 macro y matriz de confusión."],
        ["No supervisado", "Agrupar días por su perfil de 24 horas. StandardScaler, K-Means, DBSCAN y PCA para visualizar.",
         "Método del codo, coeficiente de silueta y proporción de ruido en DBSCAN."],
        ["CNN", "Clasificar 12 señales en imágenes a color de 32 × 32 o 48 × 48. Tres bloques Conv-ReLU-MaxPool, "
         "Dropout y capas densas; aumento de datos; separación por secuencia.",
         "Exactitud, curvas de pérdida y exactitud por época, matriz de confusión y análisis de señales confundidas."],
        ["Refuerzo", "Cruce de dos vías con semáforo de dos fases. Estado: colas (0 a 4) y fase. Acciones: mantener o cambiar. "
         "Recompensa: menos 1 por auto en espera y menos 1 por cambio de fase. Q-learning tabular.",
         "Espera promedio por auto frente a semáforo de tiempo fijo; recompensa por episodio; 3 combinaciones de hiperparámetros."],
    ], [2.6, 8.2, 6.2]))

    e.append(Paragraph("6. Plan de trabajo", H1))
    e.append(tabla([
        ["Entrega", "Fecha", "Contenido", "Responsable principal"],
        ["Semana 1", "14 y 16 nov.", "Entorno, repositorio, datos, EDA y propuesta", "Integrantes 1 y 2"],
        ["Semana 2", "21 y 23 nov.", "Modelos supervisados y no supervisado", "Integrantes 1 y 2"],
        ["Semana 3", "28 y 30 nov.", "CNN y agente de refuerzo", "Integrantes 4 y 3"],
        ["Semana 4", "5 y 7 dic.", "Prototipo, repositorio, video, informe y exposición", "Todo el equipo"],
    ], [2.4, 2.8, 8.0, 3.8]))
    e.append(Spacer(1, 3))
    e.append(Paragraph("Todos los integrantes hacen commits cada semana y deben poder explicar cualquier parte del proyecto. "
                       "Se hará un reporte semanal con el formato del Anexo B.", P))

    e.append(Paragraph("7. Riesgos y mitigación", H1))
    e.append(viñetas([
        "<b>PyTorch sin GPU:</b> entrenar la CNN en Google Colab y descargar el modelo (.pt); el resto corre en CPU.",
        "<b>Datos con huecos y duplicados:</b> limpieza documentada y reproducible en src/datos.py.",
        "<b>Validación inflada en GTSRB</b> (fotos casi idénticas por secuencia): separar entrenamiento y validación por secuencia.",
        "<b>Enlaces de datasets caídos:</b> descarga con respaldo manual y manifiesto de origen; proponer alternativa al docente.",
    ]))

    e.append(Paragraph("Referencias", H1))
    refs = [
        "Hogue, J. (2019). <i>Metro Interstate Traffic Volume</i> [Conjunto de datos]. UCI Machine Learning Repository. "
        "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
        "Stallkamp, J., Schlipsing, M., Salmen, J., y Igel, C. (2012). Man vs. computer: Benchmarking machine learning "
        "algorithms for traffic sign recognition. <i>Neural Networks, 32</i>, 323-332.",
        "Sutton, R. S., y Barto, A. G. (2018). <i>Reinforcement learning: An introduction</i> (2.ª ed.). MIT Press.",
    ]
    for r in refs:
        e.append(Paragraph(r, ParagraphStyle("ref", parent=P, leftIndent=14, firstLineIndent=-14, alignment=0)))

    doc.build(e, onFirstPage=pie, onLaterPages=pie)
    print("Generado:", SALIDA)


if __name__ == "__main__":
    construir()
