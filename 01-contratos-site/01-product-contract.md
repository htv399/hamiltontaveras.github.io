# Contrato de producto

ID: PRD-001

Versión: 1.0.0

## Propósito

El portal debe funcionar como la plataforma profesional CaribbeanQuant. Su propósito principal es presentar una plataforma de datos, productos cuantitativos de impacto y recursos docentes con evidencia accesible, verificable y bien presentada. Hamilton Taveras es la persona detrás del portal, pero su biografía no organiza la experiencia.

## Posicionamiento

PRD-001, required_v1. Usar CaribbeanQuant, con esa capitalización exacta, como identidad pública principal y presentar a Hamilton Taveras como la persona detrás del portal.

PRD-002, required_v1. Organizar la propuesta pública alrededor de DaaS Platform, Impact Products y Teaching, sin convertir esas áreas en claims promocionales no verificables.

PRD-003, required_v1. Evitar que el cargo actual o una institución específica definan la marca.

PRD-004, required_v1. Demostrar capacidades mediante preguntas resueltas, productos construidos, métodos, visuales, recursos y resultados verificables.

PRD-005, required_v1. Hacer que un visitante identifique en diez segundos CaribbeanQuant, a Hamilton Taveras como responsable y las tres áreas que puede explorar.

PRD-006, required_v1, APP-002. Usar exactamente “Professional website of Hamilton Taveras, M.A. in Economics, CDO.” como descripción profesional. CDO no se vincula con una institución específica.

PRD-007, content_pending, APP-008. Los catálogos, dashboards y productos se publican progresivamente solo cuando exista contenido real y aprobado. Los módulos vacíos se omiten.

## Audiencias y tareas

| Audiencia | Necesidad | Recorrido obligatorio |
| --- | --- | --- |
| Ejecutivo o cliente | Comprender la plataforma, los productos disponibles y el contacto | Home, DaaS Platform, Impact Products, About Me |
| CFO o profesional financiero | Revisar productos cuantitativos, metodología, acceso y límites | Impact Products, detalle y recursos |
| Data leader | Evaluar cobertura, procedencia, acceso y recursos de datos | DaaS Platform, catálogo y recursos |
| Economista o investigador | Revisar fuentes, métodos y productos documentados | DaaS Platform, Impact Products y detalle |
| Profesional cuantitativo | Revisar supuestos, modelos, riesgo y código público | Impact Products y recursos permitidos |
| Estudiante | Encontrar curso, semana y materiales | Teaching, Course, Week y Material |
| Visitante por enlace directo | Entender la pieza sin visitar la Home | Detalle, contexto, autoría, fecha y related work |

PRD-008, required_v1. Cada recorrido de la tabla debe completarse con un máximo de tres decisiones de navegación después de entrar a la sección.

## Principios editoriales

PRD-009, required_v1. Comenzar cada pieza por la pregunta, el fenómeno o la decisión. El método aparece después.

PRD-010, required_v1. Mostrar el hallazgo y una evidencia antes de cualquier descarga.

PRD-011, required_v1. Declarar fuentes, unidades, fecha de corte, supuestos, limitaciones y disclosure cuando corresponda.

PRD-012, required_v1. Distinguir published_at, updated_at, data_vintage y version.

PRD-013, required_v1. Las páginas de detalle deben admitir lectura rápida, revisión de evidencia y acceso posterior a método o recursos.

PRD-014, required_v1. No prometer una frecuencia editorial.

PRD-015, required_v1. No usar claims promocionales sin evidencia, testimonios inventados, cifras no verificadas ni listas de tecnologías como sustituto de resultados.

PRD-016, required_v1. No usar el patrón visual de una caja repetida con una línea de color lateral como recurso dominante. Los callouts deben ser escasos, semánticos y variar por función.

PRD-017, required_v1. Evitar cuadrículas uniformes de tarjetas cuando las piezas tengan distinta importancia.

## Alcance de v1

- Home compacta en inglés.
- DaaS Platform con overview y módulos de catálogo, dashboards, monitores, recursos y acceso condicional.
- Impact Products con landing y detalles de productos cuantitativos reales cuando existan.
- Teaching con cursos, semanas y materiales.
- About Me, CV, Contact, Resources, Search y 404.
- Taxonomías transversales.
- Sistema Blue Finance.
- Búsqueda estática.
- SEO, sitemap, feeds, Open Graph y JSON-LD.
- Despliegue estático mediante GitHub Actions y GitHub Pages.
- Fixtures demostrativos conservados únicamente para QA y excluidos de toda ruta pública.

## Fuera de alcance

PRD-018, future. Dark mode.

PRD-019, future. Newsletter, comentarios, cuentas, personalización, PWA y recomendaciones algorítmicas.

PRD-020, future. Traducción automática.

PRD-021, future. Backend propio, base transaccional, autenticación y servidor Node permanente.

PRD-022, future. LMS, calificaciones, anuncios y registros de estudiantes.

PRD-023, future. Animación compleja o decorativa.

## Contenido restringido

PRD-024, required_v1. Exámenes activos, soluciones restringidas, datos privados, información institucional sensible y casos confidenciales no se publican.

PRD-025, required_v1. Un caso reconstruido debe usar datos sintéticos y declararlo de forma visible.

PRD-026, required_v1. Los recursos descargables solo aparecen cuando access y license lo permiten.

## Criterios de producto

- QA-PRD-001 confirma la comprensión en diez segundos.
- QA-PRD-002 confirma un clic desde el hero a evidencia real.
- QA-PRD-003 confirma que no existe una Home tipo CV.
- QA-PRD-004 confirma que la tecnología no domina títulos, hero ni listings.
- QA-PRD-005 confirma ausencia de contenido inventado y placeholders en producción.

