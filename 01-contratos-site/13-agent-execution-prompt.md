# Prompt de ejecución para Claude Code o Codex

Trabaja directamente en el repositorio abierto. Debes implementar el portal profesional completo a partir de la carpeta 01-contratos-site.

Lee primero, en orden, los catorce contratos. 00-README.md define la precedencia. 09-repository-manifest.yml define el inventario final. 11-acceptance-qa.yml define las pruebas. 12-implementation-order.md define la secuencia obligatoria.

Antes de cambiar un archivo, inspecciona el repositorio completo. Revisa git status, configuración, contenido, assets, rutas, scripts, workflows y dependencias. Conserva contenido verificable, URLs útiles y recursos válidos. Migra lo que sea necesario hacia los modelos nuevos. Sustituye la presentación visual deficiente. El portal actual sirve como fuente de contenido y estructura, no como modelo visual.

No construyas un portafolio convencional ni una plantilla académica. La Home debe sentirse como una publicación de inteligencia económica y financiera con jerarquía, densidad y energía editorial. Research debe tener profundidad institucional y rigor gráfico. Las referencias Bloomberg Economics y BlackRock Investment Institute orientan la composición. No copies sus marcas, paletas, navegación, texto ni componentes.

Implementa Astro como framework público. Usa MDX para piezas editoriales. Usa React y TypeScript solamente en islas que necesiten estado o interacción. Mantén Quarto y R Markdown como pipeline separado para slides, notas, laboratorios y materiales. Publica una salida estática en GitHub Pages. No introduzcas un servidor Node permanente.

No instales dependencias por conveniencia. Usa las declaradas y los recursos ya disponibles. Si una dependencia nueva es indispensable, documenta el requisito que la exige, explica por qué HTML, CSS, Astro o una dependencia existente no lo resuelven, y espera autorización antes de agregarla. Durante CI usa npm ci. No ejecutes instalaciones de paquetes de R.

Respeta los identificadores y estados. Implementa todo required_v1. No implementes future. Para content_pending crea el esquema y una demostración marcada como placeholder. Para approval_required centraliza la opción en configuración y bloquea el release cuando la decisión sea crítica. Nunca inventes experiencia, cargos, proyectos, publicaciones, resultados, indicadores, testimonios, credenciales, licencias, enlaces ni datos de contacto.

Evita una sucesión de tarjetas iguales. No uses repetidamente cajas con una línea de color a la izquierda. Cada módulo debe tener una función editorial clara. Si falta contenido real, reduce o elimina el módulo. No muestres “próximamente”, tarjetas vacías ni relleno artificial.

Sigue exactamente 12-implementation-order.md. Al terminar cada fase, ejecuta sus validaciones y corrige los fallos antes de continuar. No te detengas después de generar archivos. Continúa hasta que el sitio compile, la búsqueda se indexe, los enlaces pasen, el manifiesto esté completo, los materiales académicos de muestra funcionen y las pruebas required_v1 queden aprobadas.

Conserva los cambios preexistentes del usuario que no estén relacionados. No uses comandos destructivos para limpiar el repositorio. Si un archivo antiguo debe retirarse, verifica primero sus referencias y registra la migración.

Al finalizar, entrega un informe breve con:

1. Archivos creados, migrados y conservados.
2. Comandos ejecutados y resultados.
3. Evidencia de cada grupo QA.
4. Decisiones approval_required que todavía bloquean producción.
5. Diferencias justificadas respecto del manifiesto.

No declares terminado el trabajo si npm run ci falla, si falta una ruta required_v1, si existen placeholders en producción o si una prueba de aceptación obligatoria no tiene evidencia.
