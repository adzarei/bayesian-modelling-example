# Guía de Estudio: Inferencia Bayesiana

## Contenidos

**Tema 1 del texto base:** secciones 1.1, 1.2, 1.3, 1.5, 1.8

En este tema se repasarán los conceptos básicos de Inferencia Bayesiana vistos con anterioridad en la asignatura de Modelado Estadístico de Datos y se profundizará en algunos de ellos con ejemplos tomados del primer tema del texto base.

### Temas principales

- Probabilidad e Inferencia
- Notación general de la inferencia estadística
- Inferencia Bayesiana
- La probabilidad como una medida de la incertidumbre
- Resultados útiles de Teoría de la Probabilidad

Este tema debe seguirse mediante el estudio del capítulo 1 del texto base, en particular de las secciones 1.1, 1.2, 1.3, 1.5 y 1.8. Una correspondencia parcial se puede encontrar en el texto *Bayes Rules!* capítulos 1 y 2.

#### Resumen de secciones

- **Sección 1.1:** Ventajas de la inferencia Bayesiana para científicos de datos. No se profundiza en el conflicto entre prácticas frecuentistas y bayesianas, pero se mencionan algunos aspectos.
- **Sección 1.2:** Notación y nomenclatura del curso. Atención especial a:
  - Distinción entre muestra y población.
  - Distinción entre parámetros y variables observables.
  - Notación matemática para escalares, vectores y matrices (vectores columna).
  - Concepto de intercambiabilidad de las variables.
  - Definición de variables explicativas.
  - Modelos jerárquicos.
- **Sección 1.3:** Notación para conceptos probabilísticos. Asegúrate de entender:
  - Distribución o densidad de probabilidad.
  - Probabilidad condicionada y marginal.
  - Regla de Bayes y probabilidades involucradas.
  - Probabilidad predictiva a priori y a posteriori.
  - Verosimilitud y principio de verosimilitud.
  - Cocientes de verosimilitud y probabilidad en general.
- **Sección 1.5:** Concepto de probabilidad y sus interpretaciones bayesianas y frecuentistas. Ejemplos de problemas de aplicación de la interpretación frecuentista. La probabilidad como medida de incertidumbre.
- **Sección 1.8:** Resultados clásicos de teoría de la probabilidad útiles para capítulos posteriores:
  - Cálculo de valores esperados y varianzas.
  - Probabilidad condicionada y su impacto en valores esperados y varianzas.
  - Cambios de variable en distribuciones de probabilidad.

---

## Modelos Jerárquicos

Ésta será nuestra primera toma de contacto con los modelos jerárquicos Bayesianos. Veremos cómo estos modelos permiten, entre otras cosas, disminuir la dependencia de los resultados de la inferencia respecto a la elección de los priors, haciendo a estos últimos sujetos de la propia inferencia.

### Temas principales

- Intercambiabilidad y modelos jerárquicos
- Análisis Bayesiano de modelos jerárquicos conjugados
- Ejemplo de las 8 escuelas
- Modelado jerárquico aplicado al meta-análisis
- Priors débilmente informativos para matrices de covarianza

Del capítulo 1 del texto base saltamos al capítulo 5 (secciones 5.1 a 5.7). Las secciones más importantes son 5.1, 5.2, 5.4 y 5.5.

#### Resumen de secciones

- **Sección 5.1:** Ejemplo paradigmático (incidencia de tumores en ratas) para ilustrar ventajas del modelado jerárquico. Uso de experimentos históricos para definir una distribución a priori. Discusión de ventajas e inconvenientes y modelado jerárquico de los datos.
- **Sección 5.2:** Profundización en el concepto de intercambiabilidad, su relación con independencia e identidad de distribuciones y el teorema de de Finetti. Se discuten casos de intercambiabilidad condicional y la definición de un modelo jerárquico simple (dos niveles), hiperprior y distribuciones a posteriori de interés.
- **Sección 5.3:** Modelos jerárquicos conjugados. Son importantes pero no siempre aplicables. Si no es posible construir un modelo conjugado, se usan técnicas computacionales (tema 4, capítulos 11 y 12). No se espera que los estudiantes creen estos modelos, solo que comprendan qué es un modelo conjugado y los tres pasos para la solución analítica. No es necesario seguir el desarrollo matemático en detalle.
- **Secciones 5.4 y 5.5:** Desarrollo completo de un modelo bayesiano jerárquico normal (observaciones gaussianas por grupos, parámetros también normales). Se discuten alternativas de *no pooling*, *complete pooling* y modelado jerárquico. Se desarrollan expresiones para la distribución a posteriori conjunta, condicional y marginal de hiperparámetros. Se calcula analíticamente la distribución a posteriori del hiperparámetro media dado el de varianza. Se discute la definición del prior para la varianza y el cálculo de distribuciones predictivas a posteriori. Aplicación al problema de las 8 escuelas.
- **Sección 5.6:** Uso de modelos jerárquicos para meta-análisis (comparar/integrar diferentes fuentes de datos). No será materia de evaluación.
- **Sección 5.7:** Opciones para distribuciones a priori de varianzas en modelos jerárquicos. Se recomienda conocer el concepto de calibración y las propuestas de la sección. Aproximación práctica en capítulos 5, 15 y 16 de *Bayes Rules!*.

---

## Evaluación y Comparación de Modelos

En este tema se aborda cómo comprobar la confiabilidad de la inferencia y si el modelo describe correctamente los datos, así como la comparación de modelos alternativos para un mismo conjunto de observaciones.

### Temas principales

- Medidas de la exactitud de las predicciones
- Criterios basados en Teoría de la Información y validación cruzada
- Comparación de modelos basada en el rendimiento predictivo
- Comparación de modelos mediante el factor de Bayes

Para el estudio de este tema, leer las secciones 7.1 a 7.4 del tema 7 del texto base. Todas son relevantes y materia de evaluación.

#### Resumen de secciones

- **Sección 7.1:** Utilidad y definición de medidas de exactitud predictiva. Definición de exactitud predictiva para un punto, valor esperado del logaritmo de la densidad predictiva. Problema/sesgo de optimismo al usar los mismos datos para ajuste y evaluación.
- **Sección 7.2:** Aproximaciones para disminuir el sesgo de evaluación sobre los mismos datos usados en el ajuste (*within-sample*). Correcciones como AIC, WAIC, DIC y validación cruzada.
- **Sección 7.3:** Aplicación de técnicas al caso de las 8 escuelas. Discusión del sesgo de selección y limitaciones de las técnicas.
- **Sección 7.4:** Alternativa basada en factores de Bayes. Metodología y ejemplos de aplicación, con discusión de limitaciones.

Los contenidos de este tema no están agrupados en un solo capítulo de *Bayes Rules!*, sino en varias subsecciones bajo *Model evaluation (& selection)*.

---

## Aspectos Computacionales de la Inferencia Bayesiana

Los modelos multiparamétricos y jerárquicos suelen requerir estimación aproximada (no analítica) de las distribuciones a posteriori. Se han desarrollado técnicas sofisticadas, pero aquí se exploran los fundamentos básicos.

### Temas principales

- Principios básicos de las cadenas de Markov
- El muestreador de Gibbs
- Algoritmos de Metropolis y Metropolis-Hastings
- Muestreadores de Gibbs y Metropolis como bloques básicos de la inferencia
- Evaluación de la convergencia
- Número efectivo de muestras
- [Hamiltonian Monte Carlo] (recomendado, fuera de temario)
- [Inferencia Variacional] (recomendado, fuera de temario)

> **Nota:** El texto base contiene material más avanzado que queda fuera de este tema por razones de carga de trabajo.

Para el estudio de este tema, leer las secciones 11.1 a 11.5. Las secciones 12.4 y 13.7 son ejemplos avanzados y quedan fuera de examen. Se recomienda una lectura superficial de Hamiltonian Monte Carlo e Inferencia Variacional solo para quienes estén interesados.

Aproximación práctica: capítulos 6, 7 y 8 de *Bayes Rules!*.

---

## Librerías de Modelado Probabilístico

En este último tema se estudiará una implementación del ejemplo de las ocho escuelas (descrito en el texto base) con la librería **TensorFlow Probability** como entorno de especificación de modelos para inferencia probabilística.

Para la práctica de evaluación continua (PEC) será necesario implementar un modelo jerárquico bayesiano para el problema descrito en el enunciado de la PEC. La implementación puede realizarse con cualquier librería de programación probabilística.

- Si se elige **TensorFlow Probability**, se puede seguir el modelo descrito en el repositorio de GitHub:  
  [`probability/tensorflow_probability/examples/jupyter_notebooks/Eight_Schools.ipynb`](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Eight_Schools.ipynb)

  El notebook contiene una implementación del ejemplo de las ocho escuelas, utilizado en repetidas ocasiones en el texto base. Se puede adaptar directamente para resolver el problema propuesto en el enunciado.

- Si se necesita un texto introductorio, se sugiere [Probabilistic Programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers), donde también hay ejemplos con PyMC.

- Para el problema de las 8 escuelas implementado en PyMC, consultar:  
  [Model comparison — PyMC v5.6.0 documentation](https://www.pymc.io/projects/docs/en/stable/pymc-examples/examples/model_comparison/Case_study_8_schools.html)

---