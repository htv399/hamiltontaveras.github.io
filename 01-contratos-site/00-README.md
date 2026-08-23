# Paquete de contratos de implementación

Versión del paquete: 1.0.0

Fecha de cierre: 2026-08-23

Estado: aprobado para implementación, sujeto únicamente a las decisiones marcadas como approval_required.

## Propósito

Este directorio traduce el contrato conceptual del portal profesional en especificaciones ejecutables. Claude Code o Codex debe construir el repositorio a partir de estos archivos sin reinterpretar la dirección editorial, el sistema visual, la arquitectura pública ni el modelo técnico.

Este paquete especifica el sitio. No contiene la implementación.

## Orden de prevalencia

Si dos instrucciones parecen incompatibles, se aplica este orden:

1. Una instrucción explícita y posterior de Hamilton Taveras.
2. Este paquete, comenzando por 00-README.md.
3. Portal_profesional_contrato_diseno_conceptual_v2.docx.
4. Instrucciones del proyecto ampliadas(1).txt.
5. El contenido y la estructura útil del repositorio existente.
6. Las preferencias o convenciones del agente implementador.

La mención de Quarto como framework público en instrucciones anteriores queda sustituida. Astro es el framework público. Quarto y R Markdown forman un pipeline separado de producción académica.

El agente no debe resolver silenciosamente una contradicción que permanezca después de aplicar este orden. Debe registrar el conflicto, identificar los requisitos afectados y solicitar aprobación.

## Estados normativos

Cada requisito usa uno de estos estados:

| Estado | Significado |
| --- | --- |
| required_v1 | Debe existir y aprobarse en la primera versión. |
| future | Mejora posterior. No se implementa salvo autorización. |
| content_pending | La estructura se construye, pero el dato o contenido final debe quedar marcado como pendiente. |
| approval_required | La implementación afectada se mantiene configurable o bloqueada hasta recibir una elección. |

Los placeholders demostrativos deben llevar demo: true o placeholder: true. No pueden publicarse en producción.

## Identificadores

Los identificadores son contratos públicos entre archivos.

| Prefijo | Área |
| --- | --- |
| PRD | Producto y posicionamiento |
| VIS | Sistema visual |
| IA | Arquitectura de información |
| CM | Modelo de contenido |
| PG | Página o plantilla |
| CMP | Componente |
| TCH | Teaching |
| TECH | Arquitectura técnica |
| REP | Repositorio |
| SEED | Contenido mínimo |
| QA | Prueba de aceptación |

No se renombra un identificador. Si un requisito queda obsoleto, se conserva con status: deprecated y se crea uno nuevo.

## Lectura obligatoria antes de implementar

El agente debe leer, en este orden, los catorce archivos del paquete. Luego debe inspeccionar el repositorio completo, su historial visible, contenido, configuración, assets y scripts. La inspección no autoriza a conservar decisiones visuales deficientes.

El manifiesto 09-repository-manifest.yml define el inventario esperado. El orden de trabajo vive en 12-implementation-order.md. Las pruebas de cierre viven en 11-acceptance-qa.yml.

## Decisiones que el agente no puede modificar

- Astro es la capa editorial pública.
- React y TypeScript se usan solamente cuando existe interacción con estado.
- MDX y las colecciones tipadas administran el contenido editorial.
- Quarto y R Markdown permanecen como pipeline separado.
- GitHub Pages aloja una salida estática.
- La navegación principal es Home, Work, Research, Notes, Teaching y About.
- CV y Search son utilidades.
- Data, Economics, Finance, Valuation y Quantitative Finance son taxonomías o recorridos, no entradas raíz.
- Work es el centro principal de evidencia.
- La Home sigue una jerarquía editorial inspirada en Bloomberg Economics.
- Research sigue un tratamiento editorial inspirado en BlackRock Investment Institute.
- No se copian marcas, paletas, textos, navegación ni componentes de esas referencias.
- El sistema Blue Finance, sus tokens base y su disciplina gráfica no se sustituyen por una plantilla genérica.
- No se implementa dark mode en v1.
- No se llenan vacíos con tarjetas artificiales, mensajes de próximamente o experiencia inventada.
- No se publican datos privados, casos institucionales sensibles ni materiales restringidos.

## Decisiones pendientes

El agente debe mantener estas decisiones en src/config/site.ts o su equivalente tipado, sin inventarlas:

- APP-001, idioma por defecto.
- APP-002, descriptor profesional definitivo.
- APP-003, dominio y canonical base.
- APP-004, firma pública y correo profesional.
- APP-005, canal de contacto.
- APP-006, licencias por familia de contenido.
- APP-007, fotografía de About.
- APP-008, inventario real inicial y jerarquía editorial.
- APP-009, analítica de privacidad.

La implementación puede usar valores demostrativos claramente marcados. El build de producción debe fallar si un placeholder crítico sigue activo.

## Regla de trazabilidad

Todo archivo creado debe aparecer en 09-repository-manifest.yml o corresponder a un patrón expresamente permitido. Todo componente debe responder a uno o más requisitos CMP. Toda página debe responder a un PG y usar modelos CM. Toda prueba QA debe citar los requisitos que verifica.

## Definición de terminado

El trabajo termina cuando:

1. El manifiesto no reporta ausencias obligatorias.
2. Los esquemas validan todo contenido.
3. Astro y los materiales académicos de muestra compilan sin instalar dependencias durante CI.
4. Pagefind indexa la salida.
5. Todas las rutas y enlaces internos pasan.
6. Las pruebas responsive y WCAG AA pasan en las vistas definidas.
7. Ningún placeholder aparece en el build de producción.
8. La revisión visual confirma una publicación editorial, no un portafolio convencional, una plantilla académica ni una sucesión de cajitas.

