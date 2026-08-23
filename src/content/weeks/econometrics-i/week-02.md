---
id: WEEK-ECONOMETRICS-I-02
title: "Probabilidad aplicada y función de esperanza condicional"
slug: probabilidad-aplicada-funcion-esperanza-condicional
summary: "Segunda unidad de Econometrics I sobre sumatoria, esperanza, varianza, covarianza, probabilidad condicional y función de esperanza condicional."
language: es
status: published
access: public
published_at: 2026-08-23
updated_at: 2026-08-19
domains: [economics]
featured: true
placeholder: false
demo: false
authors:
  - id: HAMILTON-TAVERAS
    display_name: "Hamilton Taveras"
    role_label: "Instructor"
course_id: COURSE-ECONOMETRICS-I
week_number: 2
question: "¿Cómo permiten la probabilidad y la esperanza condicional describir con precisión la relación observada entre variables?"
overview: "La clase desarrolla sumatoria, esperanza, varianza, covarianza y probabilidad condicional hasta construir la función de esperanza condicional y distinguir asociación de causalidad."
resources: []
version: "1.0.0"
license: "All rights reserved"
concepts: ["probabilidad", "esperanza", "varianza", "covarianza", "esperanza condicional"]
learning_objectives:
  - "Interpretar esperanza, varianza y covarianza."
  - "Construir e interpretar una esperanza condicional."
  - "Aplicar la ley de expectativas iteradas."
previous_week_id: WEEK-ECONOMETRICS-I-01
source_path: "02-probabilidad-cef/contenido.md"
highlight_sections:
  - 10-esperanza-condicional
  - 23-límites-que-deben-quedar-claros
---

## Semana 02 — Probabilidad aplicada y función de esperanza condicional

### 1. De los datos observados a una relación entre variables

La semana anterior planteamos preguntas como si estudiar más años aumenta el salario o si reducir el tamaño de una clase mejora el rendimiento académico. También vimos una dificultad que acompañará buena parte del curso. Observar que dos grupos tienen resultados diferentes no basta para atribuir esa diferencia a una causa.

Antes de intentar resolver ese problema necesitamos una herramienta más básica. Tenemos que aprender a describir cómo se relacionan las variables que observamos.

Consideremos nuevamente educación y salarios. Una base de corte transversal podría contener información como esta:

| Trabajador | Educación $X$ | Salario $Y$ |
|---|---:|---:|
| A | 12 | 38 |
| B | 12 | 44 |
| C | 14 | 46 |
| D | 14 | 54 |
| E | 16 | 62 |
| F | 16 | 76 |

El salario está expresado aquí en miles de pesos para mantener sencillo el ejemplo.

Hay varias preguntas posibles. Podemos preguntar cuál es el salario promedio, cuánto varían los salarios entre trabajadores o si educación alta suele aparecer junto con salarios altos. También podemos hacer una pregunta distinta:

$$
\text{¿Cuál es el salario promedio entre trabajadores con }X=x\text{ años de educación?}
$$

Esa última pregunta será el destino de esta clase.

Para llegar a ella necesitamos primero construir las herramientas probabilísticas que permiten hablar con precisión de valores promedio, dispersión y relaciones entre variables.

---

### 2. Variables aleatorias

Cuando observamos una base de datos, cada trabajador tiene un salario y un nivel educativo concretos. Si seleccionáramos aleatoriamente una persona de la población antes de mirar sus características, no sabríamos de antemano qué salario ni qué educación tendría.

Podemos representar esas cantidades mediante variables aleatorias.

Una **variable aleatoria** asigna un valor numérico a cada resultado posible de un proceso incierto.

Si $Y$ representa el salario de un trabajador seleccionado al azar, diferentes trabajadores pueden producir diferentes valores de $Y$. Si $X$ representa sus años de educación, ocurre lo mismo con $X$.

Una vez seleccionamos a una persona concreta observamos realizaciones particulares, por ejemplo,

$$
X=16,\qquad Y=62.
$$

La variable aleatoria y el valor observado no son exactamente el mismo objeto. $Y$ representa la cantidad cuyo valor todavía puede variar antes de observar a la unidad; $Y=62$ es una realización posible.

Esta distinción permite pasar de describir una base particular a formular preguntas sobre una población.

#### Probabilidades asociadas a sus posibles valores

Para una variable discreta, podemos asociar probabilidades a los valores que puede tomar.

Consideremos una variable $X$ que representa años de educación y que, en una población simplificada, puede tomar tres valores:

| $x$ | $P(X=x)$ |
|---:|---:|
| 12 | 0.40 |
| 14 | 0.35 |
| 16 | 0.25 |

Estas probabilidades pueden interpretarse como proporciones de la población. Si elegimos una persona al azar, existe una probabilidad de 0.40 de seleccionar a alguien con 12 años de educación.

Naturalmente,

$$
0\leq P(X=x)\leq 1
$$

y, si hemos enumerado todos los valores posibles,

$$
\sum_x P(X=x)=1.
$$

No necesitamos desarrollar un curso general de teoría de probabilidad. Lo que interesa es utilizar estas probabilidades para construir cantidades que después aparecerán constantemente en econometría.

---

### 3. El operador de sumatoria

Antes de construir promedios poblacionales conviene introducir una notación que utilizaremos repetidamente durante el curso.

Supongamos que observamos una secuencia de valores

$$
Y_1,Y_2,\ldots,Y_n.
$$

Si queremos escribir la suma de todos ellos podríamos hacerlo como

$$
Y_1+Y_2+\cdots+Y_n.
$$

Cuando el número de términos aumenta, esta forma se vuelve poco práctica. El operador de sumatoria permite escribir la misma expresión de manera compacta:

$$
\boxed{
\sum_{i=1}^{n}Y_i
}
$$

El símbolo $\sum$ indica que debemos sumar una colección de términos.

El índice $i$ identifica cada término. El valor que aparece debajo del símbolo indica dónde comienza la suma y el valor superior indica dónde termina.

Por tanto,

$$
\sum_{i=1}^{n}Y_i
=
Y_1+Y_2+\cdots+Y_n.
$$

Por ejemplo,

$$
\sum_{i=1}^{4}Y_i
=
Y_1+Y_2+Y_3+Y_4.
$$

El índice utilizado es solamente una forma de llevar el conteo. Podríamos escribir

$$
\sum_{i=1}^{n}Y_i
$$

o

$$
\sum_{j=1}^{n}Y_j,
$$

y ambas expresiones representan exactamente la misma suma.

#### 3.1 Suma de una constante

Si $a$ es una constante que se repite $n$ veces,

$$
\sum_{i=1}^{n}a
=
a+a+\cdots+a.
$$

Como hay $n$ términos,

$$
\boxed{
\sum_{i=1}^{n}a=na
}
$$

Por ejemplo,

$$
\sum_{i=1}^{4}5
=
5+5+5+5
=
20.
$$

Esta propiedad será útil cuando una constante aparezca dentro de una expresión que estamos sumando.

#### 3.2 Una constante puede salir de la sumatoria

Supongamos que $b$ no cambia con $i$.

Entonces,

$$
\sum_{i=1}^{n}bY_i
=
bY_1+bY_2+\cdots+bY_n.
$$

Como $b$ aparece multiplicando todos los términos, podemos factorizarlo:

$$
\boxed{
\sum_{i=1}^{n}bY_i
=
b\sum_{i=1}^{n}Y_i
}
$$

Por ejemplo,

$$
\sum_{i=1}^{3}2Y_i
=
2Y_1+2Y_2+2Y_3
=
2(Y_1+Y_2+Y_3).
$$

La condición relevante es que $b$ sea constante respecto del índice de la suma.

#### 3.3 La sumatoria se distribuye sobre una suma

Si tenemos dos variables $X_i$ y $Y_i$,

$$
\sum_{i=1}^{n}(X_i+Y_i)
$$

significa

$$
(X_1+Y_1)+(X_2+Y_2)+\cdots+(X_n+Y_n).
$$

Reagrupando los términos,

$$
(X_1+\cdots+X_n)+(Y_1+\cdots+Y_n).
$$

Por tanto,

$$
\boxed{
\sum_{i=1}^{n}(X_i+Y_i)
=
\sum_{i=1}^{n}X_i
+
\sum_{i=1}^{n}Y_i
}
$$

De la misma manera,

$$
\boxed{
\sum_{i=1}^{n}(X_i-Y_i)
=
\sum_{i=1}^{n}X_i
-
\sum_{i=1}^{n}Y_i
}
$$

Esto permite separar expresiones dentro de una sumatoria cuando están conectadas mediante suma o resta.

#### 3.4 Suma de una transformación lineal

Las propiedades anteriores pueden combinarse.

Si $a$ y $b$ son constantes,

$$
\sum_{i=1}^{n}(a+bY_i)
$$

puede separarse como

$$
\sum_{i=1}^{n}a
+
\sum_{i=1}^{n}bY_i.
$$

Utilizando las propiedades anteriores,

$$
\boxed{
\sum_{i=1}^{n}(a+bY_i)
=
na+b\sum_{i=1}^{n}Y_i
}
$$

Este tipo de manipulación aparecerá repetidamente cuando trabajemos con promedios y, posteriormente, con estimadores.

#### 3.5 Dividir una suma por el número de observaciones

Una aplicación inmediata del operador de sumatoria es expresar un promedio.

Para los valores

$$
Y_1,Y_2,\ldots,Y_n,
$$

el promedio puede escribirse como

$$
\frac{Y_1+Y_2+\cdots+Y_n}{n}.
$$

Utilizando sumatoria,

$$
\boxed{
\frac{1}{n}\sum_{i=1}^{n}Y_i
}
$$

La sumatoria agrega los valores y el factor $1/n$ los convierte en un promedio.

Por ejemplo, si

$$
Y_1=40,\qquad Y_2=50,\qquad Y_3=60,
$$

entonces

$$
\frac{1}{3}\sum_{i=1}^{3}Y_i
=
\frac{40+50+60}{3}
=
50.
$$

Esta expresión será especialmente útil cuando más adelante distingamos entre cantidades poblacionales y cantidades construidas a partir de observaciones.

Por ahora basta entender la operación matemática.

#### 3.6 Suma de desviaciones respecto del promedio

Hay una propiedad que aparecerá con frecuencia en econometría.

Sea

$$
\bar Y
=
\frac{1}{n}\sum_{i=1}^{n}Y_i.
$$

Consideremos la suma de las desviaciones de cada observación respecto del promedio:

$$
\sum_{i=1}^{n}(Y_i-\bar Y).
$$

Distribuyendo la sumatoria,

$$
\sum_{i=1}^{n}(Y_i-\bar Y)
=
\sum_{i=1}^{n}Y_i
-
\sum_{i=1}^{n}\bar Y.
$$

Como $\bar Y$ es el mismo número para todas las observaciones,

$$
\sum_{i=1}^{n}\bar Y
=
n\bar Y.
$$

Entonces,

$$
\sum_{i=1}^{n}(Y_i-\bar Y)
=
\sum_{i=1}^{n}Y_i-n\bar Y.
$$

Pero, por definición,

$$
n\bar Y
=
\sum_{i=1}^{n}Y_i.
$$

Por tanto,

$$
\boxed{
\sum_{i=1}^{n}(Y_i-\bar Y)=0
}
$$

Las desviaciones positivas respecto del promedio compensan exactamente las negativas.

Esta propiedad ayuda a entender por qué no podemos medir dispersión simplemente sumando desviaciones. Si hiciéramos eso, obtendríamos siempre cero.

Esa dificultad será precisamente la que motive el uso de desviaciones al cuadrado cuando introduzcamos la varianza.

#### 3.7 Lo que no puede hacerse con una sumatoria

Las propiedades anteriores no permiten manipular arbitrariamente cualquier expresión.

Por ejemplo,

$$
\sum_{i=1}^{n}X_iY_i
$$

no es, en general,

$$
\left(\sum_{i=1}^{n}X_i\right)
\left(\sum_{i=1}^{n}Y_i\right).
$$

El lado izquierdo suma productos correspondientes:

$$
X_1Y_1+X_2Y_2+\cdots+X_nY_n.
$$

El lado derecho genera además productos cruzados que no aparecen en la expresión original.

De manera similar,

$$
\sum_{i=1}^{n}Y_i^2
$$

no es lo mismo que

$$
\left(\sum_{i=1}^{n}Y_i\right)^2.
$$

Por ejemplo,

$$
\sum_{i=1}^{2}Y_i^2
=
Y_1^2+Y_2^2,
$$

mientras que

$$
(Y_1+Y_2)^2
=
Y_1^2+2Y_1Y_2+Y_2^2.
$$

Esta distinción será relevante cuando aparezcan varianzas, covarianzas y, posteriormente, mínimos cuadrados.

#### 3.8 Por qué necesitamos esta notación

El operador de sumatoria no introduce una idea econométrica nueva. Es una herramienta para expresar de manera compacta operaciones que ya conocemos.

Nos permitirá escribir objetos como

$$
\frac{1}{n}\sum_{i=1}^{n}Y_i
$$

sin expandir cada observación y, cuando trabajemos con probabilidades, expresiones como

$$
\sum_y yP(Y=y).
$$

La lógica es la misma en ambos casos: recorrer los valores relevantes y agregarlos siguiendo una regla.

Con esta notación ya podemos definir con precisión uno de los objetos probabilísticos que utilizaremos durante todo el curso, la esperanza.

---

### 4. Esperanza

Supongamos que queremos resumir el salario de toda una población con un valor representativo.

Una posibilidad es calcular el promedio. En lenguaje probabilístico, el promedio poblacional de una variable aleatoria se expresa mediante su **esperanza** o **valor esperado**.

Para una variable aleatoria discreta $Y$,

$$
E[Y]
=
\sum_y yP(Y=y).
$$

El valor esperado es un promedio ponderado. Cada valor posible de $Y$ recibe como peso la probabilidad con que aparece.

Si una variable puede tomar los valores

$$
Y\in\{30,50,70\}
$$

con probabilidades

$$
0.20,\quad 0.50,\quad 0.30,
$$

entonces

$$
E[Y]
=
30(0.20)+50(0.50)+70(0.30)=52.
$$

Esto no significa que necesariamente exista una persona cuyo salario sea exactamente 52. La esperanza describe el centro de la variable en la población.

La misma idea se extiende a variables que pueden tomar muchos o incluso infinitos valores. La maquinaria matemática para el caso continuo no es necesaria para los objetivos de esta semana. Lo importante es conservar la interpretación de $E[Y]$ como promedio poblacional.

#### 4.1 La esperanza como operador

Si $a$ es una constante,

$$
E[a]=a.
$$

Si multiplicamos una variable por una constante $b$,

$$
E[bY]=bE[Y].
$$

Si además sumamos una constante,

$$
E[a+bY]
=
a+bE[Y].
$$

#### Derivación

Partiendo de la definición para una variable discreta,

$$
E[a+bY]
=
\sum_y (a+by)P(Y=y).
$$

Distribuyendo la suma,

$$
E[a+bY]
=
a\sum_yP(Y=y)
+
b\sum_y yP(Y=y).
$$

Como las probabilidades de todos los valores posibles suman uno,

$$
\sum_yP(Y=y)=1,
$$

y por definición,

$$
\sum_y yP(Y=y)=E[Y].
$$

Entonces,

$$
E[a+bY]=a+bE[Y].
$$

No hemos utilizado ningún supuesto económico, causal ni estadístico especial. Es una propiedad algebraica de la esperanza.

#### Interpretación económica

Supongamos que $Y$ es ingreso antes de una transferencia y que todas las personas reciben una transferencia fija de 10 unidades. El nuevo ingreso es

$$
Y^{nuevo}=10+Y.
$$

El ingreso promedio aumenta exactamente en 10:

$$
E[Y^{nuevo}]=10+E[Y].
$$

Si en cambio todos los ingresos se multiplicaran por 1.05,

$$
E[1.05Y]=1.05E[Y].
$$

La operación realizada sobre cada valor individual tiene una consecuencia directa sobre el promedio de la población.

---

### 5. El promedio no describe toda la variable

Dos poblaciones pueden tener el mismo salario promedio y, sin embargo, mostrar situaciones muy diferentes.

Consideremos:

$$
A=\{40,50,60\}
$$

y

$$
B=\{10,50,90\}.
$$

En ambos casos el promedio es 50.

Pero en la segunda población los valores están mucho más alejados del promedio.

Necesitamos una medida que capture esa dispersión.

---

### 6. Varianza

La **varianza** mide cuánto se alejan, en promedio cuadrático, los valores de una variable de su esperanza.

$$
\operatorname{Var}(Y)
=
E[(Y-E[Y])^2].
$$

Primero medimos la desviación respecto al promedio:

$$
Y-E[Y].
$$

Si solamente promediáramos estas desviaciones obtendríamos cero, porque las desviaciones positivas y negativas se compensan. La propiedad

$$
\sum_{i=1}^{n}(Y_i-\bar Y)=0
$$

mostrada anteriormente es la versión muestral de esta misma intuición.

Por eso elevamos cada desviación al cuadrado:

$$
(Y-E[Y])^2.
$$

Finalmente calculamos su esperanza.

Una variable cuyos valores permanecen cerca de $E[Y]$ tendrá una varianza relativamente pequeña. Una variable cuyos valores están muy dispersos tendrá una varianza mayor.

#### 6.1 Ejemplo

Para

$$
Y=\{40,50,60\}
$$

con cada valor igualmente probable,

$$
E[Y]=50.
$$

Entonces,

$$
\operatorname{Var}(Y)
=
\frac{(40-50)^2+(50-50)^2+(60-50)^2}{3}
=
\frac{200}{3}.
$$

Para

$$
Z=\{10,50,90\},
$$

también tenemos $E[Z]=50$, pero

$$
\operatorname{Var}(Z)
=
\frac{(10-50)^2+(50-50)^2+(90-50)^2}{3}
=
\frac{3200}{3}.
$$

El promedio no cambió. Lo que cambió fue la dispersión de los resultados alrededor del promedio.

#### 6.2 Qué ocurre cuando transformamos una variable

Si

$$
Z=a+bY,
$$

entonces

$$
\operatorname{Var}(Z)=b^2\operatorname{Var}(Y).
$$

La constante $a$ no aparece en el resultado porque sumar la misma cantidad a todos desplaza la variable, pero no cambia la distancia relativa entre sus valores.

Partiendo de $E[Z]=a+bE[Y]$,

$$
Z-E[Z]
=
a+bY-\left(a+bE[Y]\right)
=
b(Y-E[Y]).
$$

Elevando al cuadrado y tomando esperanza,

$$
\operatorname{Var}(Z)
=
E[b^2(Y-E[Y])^2]
=
b^2E[(Y-E[Y])^2].
$$

Por definición,

$$
\boxed{
\operatorname{Var}(a+bY)=b^2\operatorname{Var}(Y)
}
$$

Este resultado se deriva de las definiciones y de propiedades algebraicas de la esperanza. No depende de un supuesto causal.

---

### 7. De una variable a dos variables

Hasta aquí hemos descrito cada variable por separado.

Podemos saber que

$$
E[X]=14
$$

y que

$$
E[Y]=50,
$$

pero esos dos números no nos dicen si trabajadores con más educación tienden a encontrarse entre quienes ganan salarios mayores.

El problema aparece porque ahora interesa saber cómo se comportan **conjuntamente** $X$ y $Y$.

Necesitamos una medida que utilice simultáneamente las desviaciones de ambas variables respecto de sus medias.

---

### 8. Covarianza

La **covarianza** entre $X$ y $Y$ se define como

$$
\operatorname{Cov}(X,Y)
=
E[(X-E[X])(Y-E[Y])].
$$

Cuando $X>E[X]$ y al mismo tiempo $Y>E[Y]$, el producto de las desviaciones es positivo. También es positivo cuando ambas variables están por debajo de sus respectivos promedios. Si una está por encima de su promedio mientras la otra está por debajo, el producto es negativo.

Una covarianza positiva indica que valores relativamente altos de $X$ tienden a aparecer junto con valores relativamente altos de $Y$, y valores relativamente bajos de una variable tienden a aparecer junto con valores relativamente bajos de la otra.

Una covarianza negativa indica una tendencia opuesta.

Una covarianza cercana a cero indica que este tipo particular de movimiento conjunto es débil.

No debemos interpretar todavía ninguna de estas afirmaciones como causal.

#### 8.1 Una forma equivalente

La definición también puede escribirse como

$$
\operatorname{Cov}(X,Y)
=
E[XY]-E[X]E[Y].
$$

Partimos de

$$
\operatorname{Cov}(X,Y)
=
E[(X-E[X])(Y-E[Y])].
$$

Multiplicando,

$$
(X-E[X])(Y-E[Y])
=
XY-XE[Y]-E[X]Y+E[X]E[Y].
$$

Tomando esperanza y usando que $E[X]$ y $E[Y]$ son constantes,

$$
\boxed{
\operatorname{Cov}(X,Y)
=
E[XY]-E[X]E[Y]
}
$$

No hemos supuesto que una variable cause a la otra ni una forma particular para su relación.

#### 8.2 Qué nos dice y qué no nos dice

La covarianza es útil porque resume movimiento conjunto mediante un solo número.

Ese beneficio también constituye su limitación.

Saber que educación y salario tienen covarianza positiva todavía no nos dice cuál es el salario promedio de quienes tienen 12, 14 o 16 años de educación.

Para responder preguntas sobre cómo cambia el promedio de $Y$ entre distintos valores de $X$, necesitamos conservar más información.

Eso nos lleva a condicionar.

---

### 9. Probabilidad condicional

Supongamos que seleccionamos aleatoriamente un trabajador.

Antes de conocer su educación podemos preguntar por la probabilidad de que su salario tome determinado valor.

Después descubrimos que ese trabajador tiene 16 años de educación.

La información disponible ha cambiado. Puede cambiar también la probabilidad que asignamos a los distintos valores de su salario.

La **probabilidad condicional** formaliza esta idea.

Para dos eventos $A$ y $B$, con $P(B)>0$,

$$
P(A\mid B)
=
\frac{P(A\cap B)}{P(B)}.
$$

En términos de nuestras variables,

$$
P(Y=y\mid X=x)
=
\frac{P(Y=y,\;X=x)}{P(X=x)},
$$

siempre que $P(X=x)>0$.

Al calcular $P(Y=y\mid X=x)$, dejamos de utilizar toda la población como referencia. Restringimos nuestra atención a las unidades para las cuales $X=x$.

#### 9.1 Ejemplo

Supongamos que en una población hay 100 trabajadores.

De ellos, 25 tienen 16 años de educación. Entre esos 25, 10 ganan 70 unidades salariales.

Entonces,

$$
P(Y=70\mid X=16)
=
\frac{10}{25}
=
0.40.
$$

El denominador no es 100.

Una vez condicionamos en $X=16$, la población relevante para esa probabilidad está formada por los 25 trabajadores con 16 años de educación.

Esta idea de cambiar la población de referencia es la que necesitamos para construir una media condicionada en $X$.

---

### 10. Esperanza condicional

La esperanza $E[Y]$ responde a la pregunta de cuál es el promedio de $Y$ en toda la población.

La esperanza condicional pregunta cuál es el promedio de $Y$ entre las unidades para las cuales $X=x$.

Para una variable discreta,

$$
\boxed{
E[Y\mid X=x]
=
\sum_y yP(Y=y\mid X=x)
}
$$

La estructura es la misma que utilizamos para $E[Y]$. Lo que cambia son las probabilidades.

En la esperanza no condicional utilizamos $P(Y=y)$. En la esperanza condicional utilizamos $P(Y=y\mid X=x)$.

Estamos promediando $Y$ dentro de una subpoblación definida por $X=x$.

#### 10.1 Volver a educación y salarios

Supongamos que obtenemos:

| Educación $x$ | $E[Y\mid X=x]$ |
|---:|---:|
| 12 | 40 |
| 14 | 50 |
| 16 | 70 |

La primera fila significa

$$
E[Y\mid X=12]=40.
$$

No significa que todos los trabajadores con 12 años de educación ganen 40.

Puede haber trabajadores con salarios de 30, 35, 45 o 55. El número 40 resume el promedio de esa subpoblación.

La esperanza condicional no elimina la heterogeneidad individual. Resume el centro de $Y$ dentro de cada grupo definido por $X$.

---

### 11. Algo que los estudiantes ya habían utilizado

En la Semana 1 apareció la expresión

$$
E[Y_i\mid X_i=1]-E[Y_i\mid X_i=0].
$$

Ahora podemos entender mejor cada término.

$$
E[Y_i\mid X_i=1]
$$

es el resultado promedio entre las unidades para las cuales $X_i=1$, mientras

$$
E[Y_i\mid X_i=0]
$$

es el resultado promedio entre las unidades para las cuales $X_i=0$.

La diferencia compara dos esperanzas condicionales.

La Semana 1 mostró además algo que debemos conservar. Esa diferencia no es causal simplemente por estar escrita de esta manera.

En un experimento aleatorio, una condición adicional sobre cómo se asigna $X_i$ permite relacionar esa diferencia con un efecto causal promedio.

Sin esa condición, sigue siendo una diferencia entre resultados promedio observados de dos grupos.

La esperanza condicional es un objeto probabilístico. La interpretación causal necesita algo más.

---

### 12. De varias medias condicionales a una función

Cuando $X$ solamente toma los valores 0 y 1, existen dos medias condicionales.

Pero educación puede tomar valores como

$$
12,\;13,\;14,\;15,\;16,\;17,\;18.
$$

Podemos calcular una esperanza condicional para cada uno.

En lugar de considerar esos números como objetos desconectados podemos entenderlos como una función de $x$:

$$
\boxed{
x\longmapsto E[Y\mid X=x]
}
$$

Esta es la **función de esperanza condicional**, o CEF por sus siglas en inglés.

La CEF asigna a cada valor posible de $X$ el promedio de $Y$ entre las unidades que tienen ese valor de $X$.

---

### 13. Qué representa la CEF

Para cada nivel educativo $x$, la CEF responde cuál es el salario promedio entre quienes tienen $X=x$.

Supongamos:

| $x$ | $E[Y\mid X=x]$ |
|---:|---:|
| 12 | 40 |
| 13 | 43 |
| 14 | 48 |
| 15 | 55 |
| 16 | 63 |
| 17 | 68 |
| 18 | 72 |

Podemos representar los pares

$$
(x,E[Y\mid X=x])
$$

en un gráfico.

El eje horizontal contiene educación. El vertical contiene el salario promedio condicionado al nivel educativo.

La CEF no coloca necesariamente a cada trabajador sobre esos puntos. Los salarios individuales pueden estar por encima o por debajo del promedio correspondiente a su nivel educativo.

La función describe cómo cambia **el promedio condicional** de $Y$ con $X$.

#### 13.1 La CEF puede tener distintas formas

No existe ninguna razón probabilística para que los valores

$$
E[Y\mid X=x]
$$

formen exactamente una línea recta.

La relación promedio podría aumentar rápidamente en algunos rangos y lentamente en otros. Incluso podría cambiar de dirección.

En esta semana no necesitamos imponerle una forma.

Nuestro objetivo es entender el objeto poblacional que queremos describir:

$$
E[Y\mid X=x].
$$

La pregunta de cómo representar esa relación mediante una forma más sencilla queda abierta para el siguiente paso del curso.

---

### 14. CEF y resultados individuales

Una confusión frecuente consiste en interpretar

$$
E[Y\mid X=x]
$$

como si fuera el resultado de cada persona con $X=x$.

No lo es.

Supongamos que

$$
E[Y\mid X=16]=63.
$$

Un trabajador con 16 años de educación puede ganar 45, otro 60, otro 75 y otro 90.

El 63 representa el promedio de esa población condicionada.

Hay entonces dos niveles distintos de información. $Y$ describe el resultado que puede tomar una unidad particular, mientras $E[Y\mid X=x]$ describe el centro de los resultados dentro de un grupo caracterizado por $X=x$.

Esta diferencia será muy útil cuando comencemos a estudiar regresión. Una relación promedio no pretende explicar perfectamente cada observación individual.

---

### 15. CEF, asociación y causalidad

Regresemos a la pregunta de la Semana 1:

**¿Estudiar un año adicional aumenta el salario?**

Supongamos que encontramos

$$
E[Y\mid X=16]>E[Y\mid X=12].
$$

Podemos concluir que, en la población observada, los trabajadores con 16 años de educación tienen un salario promedio mayor que los trabajadores con 12 años.

Eso es una afirmación sobre una relación observada.

No podemos concluir únicamente de esta desigualdad que cuatro años adicionales de educación causaron toda la diferencia.

Los trabajadores con diferentes niveles educativos podrían diferir también en experiencia, oportunidades, antecedentes familiares, habilidad u otras características.

La CEF describe correctamente la relación entre educación observada y salario promedio observado. El problema causal es diferente.

La econometría debe mantener separadas dos preguntas:

$$
\text{¿Cómo se relacionan }X\text{ y }Y\text{ en la población?}
$$

y

$$
\text{¿Qué ocurriría con }Y\text{ si modificáramos }X\text{?}
$$

La CEF responde la primera.

Para atribuir una interpretación causal hacen falta condiciones adicionales.

---

### 16. Reconstruir el promedio general a partir de los grupos

Supongamos nuevamente que la población se divide según años de educación.

Conocemos

$$
E[Y\mid X=12],
\qquad
E[Y\mid X=14],
\qquad
E[Y\mid X=16],
$$

y también las proporciones

$$
P(X=12),
\qquad
P(X=14),
\qquad
P(X=16).
$$

El promedio general puede construirse como un promedio ponderado de los promedios de cada grupo.

Esta idea es la **ley de expectativas iteradas**.

Para una variable discreta $X$,

$$
\boxed{
E[Y]
=
\sum_x E[Y\mid X=x]P(X=x)
}
$$

Promediar dentro de cada grupo y después promediar entre grupos, utilizando sus tamaños relativos, devuelve el promedio de toda la población.

---

### 17. Un ejemplo numérico de la ley de expectativas iteradas

Supongamos:

| Educación $x$ | Proporción $P(X=x)$ | Salario promedio $E[Y\mid X=x]$ |
|---:|---:|---:|
| 12 | 0.40 | 40 |
| 14 | 0.35 | 50 |
| 16 | 0.25 | 70 |

Entonces,

$$
E[Y]
=
40(0.40)+50(0.35)+70(0.25)
=
16+17.5+17.5.
$$

Por tanto,

$$
\boxed{
E[Y]=51
}
$$

No basta con calcular

$$
\frac{40+50+70}{3}.
$$

Eso daría el mismo peso a cada grupo aunque los grupos tengan tamaños diferentes.

La ley utiliza las proporciones reales de la población.

---

### 18. Derivación de la ley de expectativas iteradas

Partimos de la definición:

$$
E[Y]
=
\sum_y yP(Y=y).
$$

Cada probabilidad $P(Y=y)$ puede reconstruirse considerando las distintas posibilidades de $X$:

$$
P(Y=y)
=
\sum_x P(Y=y,\;X=x).
$$

Sustituyendo,

$$
E[Y]
=
\sum_y y
\sum_x P(Y=y,\;X=x).
$$

Reordenando las sumas,

$$
E[Y]
=
\sum_x
\sum_y
yP(Y=y,\;X=x).
$$

Por la definición de probabilidad condicional,

$$
P(Y=y,\;X=x)
=
P(Y=y\mid X=x)P(X=x).
$$

Entonces,

$$
E[Y]
=
\sum_x
P(X=x)
\sum_y yP(Y=y\mid X=x).
$$

La suma interior es precisamente

$$
E[Y\mid X=x].
$$

Por tanto,

$$
\boxed{
E[Y]
=
\sum_x E[Y\mid X=x]P(X=x)
}
$$

#### Qué utilizamos en la derivación

El paso de una expresión a otra utiliza definiciones y reglas de probabilidad.

- No hemos supuesto que $X$ cause $Y$.
- No hemos supuesto que la relación sea lineal.
- No hemos supuesto asignación aleatoria.
- No hemos introducido ningún estimador.

La ley de expectativas iteradas es una propiedad probabilística.

---

### 19. El caso binario conecta directamente con la Semana 1

Si $X$ solamente puede tomar los valores 0 y 1, la expresión anterior se convierte en

$$
E[Y]
=
E[Y\mid X=1]P(X=1)
+
E[Y\mid X=0]P(X=0).
$$

El promedio de toda la población es un promedio ponderado de los resultados promedio de los dos grupos.

Si $X=1$ identifica tratamiento y $X=0$ no tratamiento, entonces $E[Y\mid X=1]$ es el resultado observado promedio de los tratados y $E[Y\mid X=0]$ el resultado observado promedio de los no tratados.

Nada en la ley de expectativas iteradas afirma que esos grupos sean comparables causalmente.

La probabilidad organiza la información que observamos. La causalidad requiere justificar la comparación que hacemos con esa información.

---

### 20. Una segunda aplicación: tamaño de clase y calificación

Sea $X=1$ si un estudiante está en una clase reducida y $X=0$ si está en una clase regular. Sea $Y$ su calificación.

Podemos preguntar:

$$
E[Y\mid X=1]
$$

y

$$
E[Y\mid X=0].
$$

Estas cantidades describen el rendimiento promedio en cada grupo.

También podemos calcular $E[Y]$.

La ley de expectativas iteradas establece

$$
E[Y]
=
E[Y\mid X=1]P(X=1)
+
E[Y\mid X=0]P(X=0).
$$

La relación es puramente aritmética y probabilística.

Si posteriormente quisiéramos interpretar

$$
E[Y\mid X=1]-E[Y\mid X=0]
$$

como efecto de reducir el tamaño de la clase, tendríamos que regresar al problema de asignación discutido en la Semana 1.

Una misma cantidad puede ser perfectamente válida como descripción y, al mismo tiempo, requerir condiciones adicionales para recibir una interpretación causal.

---

### 21. Qué aporta cada herramienta

La esperanza resume el nivel promedio de una variable:

$$
E[Y].
$$

La varianza resume su dispersión alrededor de ese promedio:

$$
\operatorname{Var}(Y).
$$

La covarianza resume una forma de movimiento conjunto entre dos variables:

$$
\operatorname{Cov}(X,Y).
$$

La probabilidad condicional cambia la población de referencia:

$$
P(Y=y\mid X=x).
$$

La esperanza condicional calcula el promedio de $Y$ dentro de esa población:

$$
E[Y\mid X=x].
$$

La función de esperanza condicional reúne esos promedios para todos los valores relevantes de $X$:

$$
x\longmapsto E[Y\mid X=x].
$$

Finalmente, la ley de expectativas iteradas conecta los promedios de las subpoblaciones con el promedio de toda la población:

$$
E[Y]
=
\sum_xE[Y\mid X=x]P(X=x).
$$

No son definiciones independientes que deban memorizarse como una lista. Cada objeto resuelve una dificultad creada por el anterior.

---

### 22. Por qué la CEF será central para lo que sigue

Hasta ahora hemos permitido que

$$
E[Y\mid X=x]
$$

cambie libremente con $x$.

Para educación y salarios podríamos obtener muchos promedios distintos.

Esto describe con bastante flexibilidad la relación promedio entre las variables.

Pero aparecen nuevas preguntas.

¿Qué ocurre si $X$ puede tomar muchos valores?

¿Necesitamos almacenar un promedio diferente para cada uno?

¿Podemos resumir la relación de una manera más parsimoniosa?

La siguiente etapa del curso buscará una representación sencilla de la relación entre $X$ y el promedio de $Y$.

Todavía no necesitamos construirla.

Lo que sí necesitamos llevarnos de esta semana es el objeto que queremos describir:

$$
\boxed{
E[Y\mid X=x]
}
$$

La regresión que aparece después no debe comenzar como una fórmula aislada. Debe aparecer como una respuesta a esta necesidad.

---

### 23. Límites que deben quedar claros

- Una esperanza no describe toda la distribución de una variable. Dos poblaciones pueden tener la misma media y distinta dispersión.
- Una covarianza resume una dimensión del movimiento conjunto, pero no conserva toda la forma de la relación entre dos variables.
- Una esperanza condicional es un promedio de una subpoblación. No describe necesariamente a cada individuo que pertenece a ella.
- La CEF puede adoptar formas distintas. Esta semana no existe ninguna razón para imponer que sea una recta.

Sobre todo,

$$
E[Y\mid X=x]
$$

no es automáticamente una relación causal.

La semana anterior aprendimos que una comparación causal exige preguntar qué habría ocurrido bajo una alternativa contrafactual y justificar por qué los grupos comparados permiten aprender sobre ella.

Condicionar en $X$ organiza la población observada.

No crea por sí mismo el contrafactual que falta.

---

### 24. Punto de llegada de la Semana 2

Al terminar esta semana, el estudiante debe poder pasar de una pregunta informal como

**“¿cómo se relacionan educación y salario?”**

a una expresión precisa:

$$
E[Y\mid X=x].
$$

Debe entender que esa expresión toma a quienes tienen $X=x$, calcula el promedio de $Y$ dentro de ese grupo y, cuando se considera para diferentes valores de $x$, construye la función de esperanza condicional.

Debe poder distinguir ese objeto de $E[Y]$, interpretar el papel de la varianza y la covarianza, y utilizar la ley de expectativas iteradas para conectar medias condicionadas con la media de toda la población.

También debe conservar una distinción aprendida desde el inicio del curso:

$$
\text{relación observada}
\neq
\text{efecto causal}.
$$

Con esta base probabilística ya estamos en condiciones de formular la siguiente pregunta:

**¿cómo podemos resumir mediante una relación sencilla el cambio de $E[Y\mid X=x]$ cuando cambia $x$?**

Esa pregunta abre la regresión simple.

