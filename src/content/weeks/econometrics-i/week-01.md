---
id: WEEK-ECONOMETRICS-I-01
title: "Introducción a la econometría, causalidad y tipos de datos"
slug: introduccion-causalidad-tipos-datos
summary: "Primera unidad de Econometrics I sobre resultados potenciales, el problema contrafactual, la selección y las estructuras fundamentales de datos económicos."
language: es
status: published
access: public
published_at: 2026-08-23
updated_at: 2026-08-18
domains: [economics]
featured: true
placeholder: false
demo: false
authors:
  - id: HAMILTON-TAVERAS
    display_name: "Hamilton Taveras"
    role_label: "Instructor"
course_id: COURSE-ECONOMETRICS-I
week_number: 1
question: "¿Cómo permiten los datos económicos distinguir una asociación observada de una comparación causal?"
overview: "La unidad introduce los resultados potenciales, el contrafactual, la selección y las estructuras de corte transversal, series de tiempo y panel."
resources: []
version: "1.0.0"
license: "All rights reserved"
concepts: ["causalidad", "resultados potenciales", "selección", "tipos de datos"]
learning_objectives:
  - "Distinguir preguntas predictivas y causales."
  - "Definir un efecto causal mediante resultados potenciales."
  - "Reconocer datos de corte transversal, series de tiempo y panel."
next_week_id: WEEK-ECONOMETRICS-I-02
source_path: "01-causalidad-tipos-datos/contenido.md"
highlight_sections:
  - el-efecto-causal-individual
  - lo-que-sabemos-y-lo-que-todavía-no-sabemos
---

## Introducción a la econometría, causalidad y tipos de datos

### ¿Ir a la universidad aumenta el salario?

Las personas con educación universitaria suelen ganar más que las personas que no fueron a la universidad. Esa observación parece sugerir una respuesta sencilla: estudiar aumenta los salarios.

Pero la pregunta económica que nos interesa es más exigente.

Queremos saber cuánto ganaría una persona si fuera a la universidad comparado con cuánto ganaría esa misma persona si no fuera. La diferencia entre esos dos escenarios es distinta de comparar el salario de dos personas diferentes, una que decidió estudiar y otra que no.

Esta dificultad aparece constantemente en economía.

¿Un programa de capacitación aumenta la probabilidad de conseguir empleo? ¿Una reducción del tamaño de las clases mejora el aprendizaje? ¿Un aumento del salario mínimo reduce el empleo? ¿Una transferencia monetaria aumenta el consumo de los hogares?

Los datos pueden mostrarnos diferencias entre personas, empresas, países o momentos del tiempo. Determinar cuándo esas diferencias permiten responder una pregunta causal requiere algo más.

Ese es uno de los problemas centrales de la econometría.

### ¿Qué hace la econometría?

La economía propone relaciones sobre el comportamiento de personas, empresas, mercados y gobiernos. Para confrontar esas ideas con la realidad necesitamos datos.

La econometría utiliza métodos estadísticos para estudiar relaciones económicas a partir de esos datos.

Dos preguntas aparecerán repetidamente durante el curso.

Una es predictiva. Queremos utilizar información disponible para anticipar una variable que todavía no conocemos. Podemos preguntarnos, por ejemplo, cuál será la inflación del próximo trimestre dadas ciertas variables observadas hoy.

La otra es causal. Queremos saber cómo cambiaría un resultado si modificáramos alguna condición. Nos interesa saber qué ocurriría con el salario de una persona si aumentara su educación, o con el aprendizaje de un estudiante si estudiara en una clase más pequeña.

Estas preguntas no son equivalentes.

Que una variable permita predecir otra no significa necesariamente que modificar la primera provoque un cambio en la segunda. Una relación observada en los datos puede ser útil para predecir y, al mismo tiempo, no tener una interpretación causal.

En esta primera clase nos concentraremos en entender qué queremos decir cuando hablamos de un efecto causal.

### Tres preguntas para comenzar

Las personas con educación universitaria suelen ganar más que quienes no fueron a la universidad. Los distritos escolares con clases más pequeñas pueden obtener mejores resultados académicos. En lugares donde los cigarrillos son más caros, su consumo puede ser menor.

Las tres son relaciones que podemos buscar en los datos. Pero las preguntas que interesan al economista son distintas.

- ¿Ir a la universidad aumenta el salario?
- ¿Reducir el tamaño de una clase mejora el aprendizaje?
- ¿Aumentar el precio de los cigarrillos reduce su consumo?

Pasar de la primera clase de afirmaciones a la segunda es pasar de describir lo que observamos a preguntarnos qué ocurriría si algo cambiara.

La educación y los salarios servirán como caso principal para introducir el contrafactual. El tamaño de clase permitirá entender qué resuelve un experimento aleatorio. El precio de los cigarrillos mostrará por qué muchas preguntas económicas importantes deben estudiarse con datos observacionales.

### Dos versiones de una misma persona

Volvamos a la educación.

Consideremos una persona $i$. Podemos imaginar dos escenarios.

En el primero, la persona recibe el tratamiento. En nuestro ejemplo, completa educación universitaria.

En el segundo, no recibe el tratamiento.

Definimos

$$
D_i =
\begin{cases}
1 & \text{si la persona } i \text{ recibe el tratamiento},\\
0 & \text{si la persona } i \text{ no recibe el tratamiento}.
\end{cases}
$$

Ahora imaginemos el salario que tendría esa persona bajo cada escenario.

Denotamos por

$$
Y_i(1)
$$

el salario que tendría la persona $i$ si recibiera el tratamiento, y por

$$
Y_i(0)
$$

el salario que tendría si no lo recibiera.

Estas cantidades se denominan **resultados potenciales**.

La palabra potencial es importante. No estamos diciendo todavía qué ocurrió. Estamos describiendo qué resultado correspondería a cada uno de los dos estados posibles.

Podemos representar la idea mediante una tabla.

| Persona | $Y_i(1)$ | $Y_i(0)$ |
|---|---:|---:|
| Ana | salario con universidad | salario sin universidad |
| Bruno | salario con universidad | salario sin universidad |
| Carla | salario con universidad | salario sin universidad |

Si pudiéramos observar ambas columnas para cada persona, medir un efecto causal sería sencillo.

La misma lógica puede trasladarse a otros problemas. Para un estudiante podemos pensar en el resultado académico con una clase pequeña y con una clase grande. Para un consumidor podemos pensar en el consumo de cigarrillos bajo un precio alto y bajo un precio bajo.

### El efecto causal individual

El efecto causal del tratamiento para la persona $i$ es

$$
\tau_i = Y_i(1)-Y_i(0).
$$

Si

$$
\tau_i>0,
$$

la persona tendría un resultado mayor bajo tratamiento que sin tratamiento.

Si

$$
\tau_i<0,
$$

el tratamiento reduciría su resultado.

No existe ninguna razón para exigir que el efecto sea idéntico para todas las personas. La universidad podría tener un efecto grande sobre el salario de una persona y pequeño sobre el de otra.

Esta definición captura la comparación que realmente queremos hacer.

No estamos comparando a Ana con Bruno. Estamos comparando a Ana bajo dos situaciones alternativas.

Ahí aparece inmediatamente el problema.

### Solo observamos uno de los dos resultados

Una persona no puede aparecer simultáneamente en nuestros datos habiendo ido y no habiendo ido a la universidad.

Si Ana fue a la universidad, observamos

$$
Y_i(1),
$$

pero no observamos

$$
Y_i(0).
$$

Si Bruno no fue a la universidad, observamos

$$
Y_i(0),
$$

pero no observamos

$$
Y_i(1).
$$

El resultado que habría ocurrido bajo la alternativa no realizada es el **resultado contrafactual**.

Podemos escribir el resultado observado como

$$
Y_i=D_iY_i(1)+(1-D_i)Y_i(0).
$$

Esta expresión contiene únicamente una regla de observación.

Cuando $D_i=1$,

$$
Y_i=Y_i(1).
$$

Cuando $D_i=0$,

$$
Y_i=Y_i(0).
$$

No hemos utilizado ningún supuesto estadístico ni causal. Simplemente hemos expresado algebraicamente cuál de los dos resultados potenciales observamos.

El problema causal puede verse entonces con claridad: para calcular

$$
\tau_i=Y_i(1)-Y_i(0)
$$

necesitaríamos observar simultáneamente dos resultados, pero los datos contienen solo uno.

Este es el problema fundamental de la inferencia causal.

### Del efecto individual al efecto promedio

Aunque no podamos observar el efecto causal de una persona, muchas preguntas económicas no requieren conocer cada efecto individual.

Un gobierno que evalúa un programa de capacitación puede estar interesado en saber cuánto aumenta, en promedio, el empleo de quienes reciben el programa.

Definimos el efecto promedio del tratamiento como

$$
ATE=E[Y_i(1)-Y_i(0)].
$$

Por las propiedades del valor esperado,

$$
ATE=E[Y_i(1)]-E[Y_i(0)].
$$

El ATE compara el resultado promedio que tendría la población bajo tratamiento con el resultado promedio que tendría esa misma población sin tratamiento.

La comparación sigue siendo contrafactual. La diferencia es que ahora buscamos un efecto promedio y no el efecto de una persona particular.

Esta transición resulta útil porque, bajo determinadas condiciones, los promedios sí pueden aprenderse comparando grupos de personas.

La pregunta es cuáles condiciones necesitamos.

### ¿Por qué no comparar simplemente los dos grupos?

Tenemos datos de salarios. Un grupo fue a la universidad y otro no.

Una primera aproximación podría ser calcular

$$
E[Y_i\mid D_i=1]-E[Y_i\mid D_i=0].
$$

El primer término es el salario promedio observado entre quienes estudiaron en la universidad. El segundo es el salario promedio observado entre quienes no lo hicieron.

Como para los tratados observamos $Y_i(1)$ y para los no tratados observamos $Y_i(0)$,

$$
E[Y_i\mid D_i=1]
=
E[Y_i(1)\mid D_i=1]
$$

y

$$
E[Y_i\mid D_i=0]
=
E[Y_i(0)\mid D_i=0].
$$

Por tanto, la diferencia observada es

$$
E[Y_i(1)\mid D_i=1]
-
E[Y_i(0)\mid D_i=0].
$$

Esta cantidad no es automáticamente un efecto causal.

Para entender el problema, sumamos y restamos

$$
E[Y_i(0)\mid D_i=1].
$$

Entonces,

$$
\begin{aligned}
&E[Y_i(1)\mid D_i=1]
-
E[Y_i(0)\mid D_i=0]
\\
=&
\left(
E[Y_i(1)\mid D_i=1]
-
E[Y_i(0)\mid D_i=1]
\right)
\\
&+
\left(
E[Y_i(0)\mid D_i=1]
-
E[Y_i(0)\mid D_i=0]
\right).
\end{aligned}
$$

El primer término compara, para quienes efectivamente recibieron el tratamiento, su resultado bajo tratamiento con el resultado que habrían tenido sin tratamiento.

Es un efecto causal promedio para ese grupo.

El segundo término compara lo que habrían obtenido sin tratamiento quienes decidieron tratarse con lo que obtuvieron sin tratamiento quienes no se trataron.

Ese término captura una diferencia previa entre los grupos.

En el ejemplo de educación, quienes van a la universidad pueden diferir de quienes no van en muchas dimensiones que también están relacionadas con sus salarios. Pueden provenir de hogares diferentes, tener distintas oportunidades, habilidades o preferencias.

Por eso,

$$
\text{diferencia observada}
=
\text{efecto causal}
+
\text{diferencias entre los grupos}.
$$

A estas últimas las llamaremos, en este punto, **selección**.

La comparación observada coincide con el efecto causal solamente cuando logramos eliminar esa segunda fuente de diferencias.

### El experimento ideal

Podemos entender qué necesitamos imaginando un experimento.

Consideremos ahora la pregunta sobre tamaño de clase. Queremos saber si reducir el número de estudiantes por aula mejora el rendimiento académico.

Si simplemente comparamos escuelas con clases pequeñas y escuelas con clases grandes, los grupos pueden diferir en recursos, composición socioeconómica, calidad docente u otras características.

El experimento ideal intenta eliminar esas diferencias sistemáticas.

Supongamos que tomamos un grupo suficientemente grande de estudiantes y asignamos aleatoriamente quién estudia en una clase pequeña y quién estudia en una clase regular.

Un grupo recibe el tratamiento,

$$
D_i=1,
$$

y el otro pertenece al grupo de control,

$$
D_i=0.
$$

La asignación no depende de las características de los estudiantes.

La aleatorización busca que los dos grupos sean comparables antes de recibir el tratamiento.

Si esto ocurre, las diferencias sistemáticas en sus resultados potenciales no determinan quién recibe el tratamiento. En promedio, el grupo tratado puede utilizarse para aprender qué ocurre bajo tratamiento y el grupo de control para aprender qué ocurre sin tratamiento.

Entonces la diferencia

$$
E[Y_i\mid D_i=1]-E[Y_i\mid D_i=0]
$$

puede interpretarse como un efecto causal promedio.

El experimento STAR de Tennessee ofrece un ejemplo concreto de esta lógica al asignar estudiantes y docentes aleatoriamente a clases de distintos tamaños.

La importancia del experimento aleatorio para este curso no está solamente en los experimentos que efectivamente podamos realizar.

El experimento nos proporciona un punto de referencia.

Nos muestra cómo serían los datos si pudiéramos construir una comparación en la que recibir el tratamiento no estuviera relacionado con las características que determinan los resultados.

### ¿Por qué no hacemos experimentos para todo?

Consideremos ahora el consumo de cigarrillos.

Queremos saber si aumentar su precio reduce el consumo.

Podemos imaginar el experimento ideal. Tomaríamos unidades comparables, variaríamos aleatoriamente el precio que enfrentan y observaríamos cómo cambia su consumo.

En la práctica, un experimento de este tipo puede ser inviable, costoso o poco representativo. Los precios que observamos suelen cambiar por impuestos, regulación y condiciones económicas.

La mayor parte de los datos económicos no proviene de experimentos controlados.

Las personas deciden si estudian. Las empresas deciden cuánto invertir. Los bancos deciden a quién prestar. Los gobiernos deciden dónde ejecutar programas. Los trabajadores deciden cuánto trabajar.

Estas decisiones producen datos.

Cuando observamos posteriormente los resultados, los grupos que queremos comparar pueden haber sido diferentes desde el comienzo.

Por eso encontrar una relación en los datos no basta para establecer causalidad.

Si observamos que

$$
E[Y_i\mid D_i=1]
>
E[Y_i\mid D_i=0],
$$

sabemos que los resultados promedio de los grupos son diferentes.

Para interpretar esa diferencia como el efecto de $D_i$ sobre $Y_i$, necesitamos una razón que permita considerar que la comparación recupera el contrafactual que no observamos.

Buena parte de la econometría que estudiaremos durante el curso puede entenderse desde esta dificultad.

¿Cómo podemos aprender sobre efectos causales cuando los datos no provienen del experimento que idealmente querríamos realizar?

Todavía no tenemos las herramientas para responder esa pregunta. Primero necesitamos construirlas.

### Los datos económicos

Antes de estudiar cómo extraer información de los datos necesitamos reconocer su estructura.

Un conjunto de datos contiene observaciones sobre determinadas unidades.

Una unidad puede ser una persona, un hogar, una empresa, una escuela, una provincia o un país. También observamos variables asociadas a esas unidades, como salario, educación, empleo, precios, producción o consumo.

La estructura del conjunto de datos depende de qué unidades observamos y de cuándo las observamos.

Tres estructuras aparecerán repetidamente durante el curso.

### Datos de corte transversal

Los datos de **corte transversal** contienen múltiples unidades observadas en un mismo período o aproximadamente en el mismo momento.

Podríamos observar, por ejemplo, una muestra de trabajadores dominicanos durante 2026.

| Persona | Salario | Educación | Edad |
|---|---:|---:|---:|
| 1 | ... | ... | ... |
| 2 | ... | ... | ... |
| 3 | ... | ... | ... |

Cada fila corresponde a una persona diferente.

La dimensión que cambia principalmente es la unidad observada.

Este tipo de datos permite estudiar cómo difieren unas unidades de otras.

Podemos preguntar si las personas con más educación tienen salarios mayores, si los hogares de mayores ingresos consumen más o si las empresas más grandes son más productivas.

Estas comparaciones pueden revelar asociaciones importantes.

Su interpretación causal requiere preguntarnos nuevamente si las unidades comparadas difieren en otros aspectos relacionados con el resultado.

### Series de tiempo

Una **serie de tiempo** contiene observaciones de una misma unidad en diferentes momentos.

Por ejemplo, podríamos observar la tasa de inflación mensual de República Dominicana.

| Mes | Inflación |
|---|---:|
| enero | ... |
| febrero | ... |
| marzo | ... |

Aquí la unidad permanece fija y cambia el tiempo.

También podríamos estudiar el PIB trimestral de un país, la tasa de cambio diaria o la tasa de desempleo mensual.

El orden temporal de las observaciones importa. Enero ocurre antes que febrero y febrero antes que marzo.

Esto distingue las series de tiempo de un corte transversal. Las observaciones no son simplemente unidades intercambiables colocadas una debajo de otra.

Los métodos específicos necesarios para estudiar la dependencia temporal pertenecen a otros cursos o a tratamientos posteriores. Por ahora basta reconocer la estructura.

### Datos de panel

Los datos de **panel** combinan las dos dimensiones anteriores.

Observamos múltiples unidades y seguimos cada una durante varios períodos.

Podríamos observar el salario de las mismas personas durante varios años.

| Persona | Año | Salario | Educación |
|---|---:|---:|---:|
| 1 | 2024 | ... | ... |
| 1 | 2025 | ... | ... |
| 1 | 2026 | ... | ... |
| 2 | 2024 | ... | ... |
| 2 | 2025 | ... | ... |
| 2 | 2026 | ... | ... |

Ahora podemos comparar personas diferentes y también observar cómo cambia una misma persona a través del tiempo.

Un panel puede contener hogares, empresas, escuelas, provincias o países.

La característica que lo define es que las mismas unidades aparecen repetidamente.

No estudiaremos todavía los métodos econométricos específicos para explotar esta estructura. Lo que necesitamos por ahora es ser capaces de reconocerla y entender qué información adicional contiene.

### Una misma pregunta, distintos datos

Consideremos nuevamente una pregunta causal:

¿La educación aumenta el salario?

- Podríamos disponer de un corte transversal con educación y salario de muchas personas en un año determinado.
- También podríamos tener información de salario promedio y educación para un país durante muchos años.
- O podríamos seguir a las mismas personas durante varios años y construir un panel.

Los tres conjuntos contienen información sobre educación y salarios, pero no contienen la misma información.

La forma en que fueron generados los datos determina qué comparaciones podemos hacer y qué métodos serán apropiados posteriormente.

Identificar las variables no basta.

También necesitamos entender quién fue observado, cuándo fue observado y por qué tenemos esas observaciones.

### Lo que sabemos y lo que todavía no sabemos

Podemos definir un efecto causal mediante resultados potenciales:

$$
\tau_i=Y_i(1)-Y_i(0).
$$

También sabemos por qué no podemos observarlo directamente para una persona. Solo uno de los dos resultados potenciales aparece en los datos.

Pasar a promedios permite formular una cantidad de interés para una población, pero comparar directamente tratados y no tratados puede mezclar el efecto causal con diferencias preexistentes entre los grupos.

El experimento aleatorio muestra una forma ideal de resolver ese problema haciendo comparables los grupos.

Los datos económicos reales, sin embargo, suelen ser observacionales y pueden aparecer como cortes transversales, series de tiempo o paneles.

Todavía nos falta una herramienta para describir sistemáticamente cómo cambia el comportamiento promedio de una variable cuando conocemos otra.

Ese será el punto de partida de la próxima clase.
