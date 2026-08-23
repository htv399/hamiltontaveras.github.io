# Orden de implementación

El agente debe completar las fases en el orden indicado. No puede iniciar una fase si la anterior no cumple su condición de cierre.

## Fase 0. Inspección y resguardo

1. Leer los catorce contratos.
2. Inventariar archivos existentes con git status y búsqueda de rutas.
3. Identificar contenido, assets, URLs y scripts útiles.
4. Comparar el inventario con 09-repository-manifest.yml.
5. Registrar archivos que se conservan, migran, sustituyen o eliminan.

Comandos mínimos:

    git status --short
    find . -maxdepth 4 -type f
    rg -n "title:|slug:|permalink:|canonical:" .

Cierre: existe un informe local de migración y ninguna modificación se ha hecho todavía.

## Fase 1. Base técnica

Implementar REP-ROOT, REP-CONFIG y REP-SCRIPTS. Fijar versiones en package-lock.json. Configurar Astro para GitHub Pages, TypeScript estricto, MDX, React, sitemap y Pagefind.

Comandos:

    npm ci
    npm run check
    npm run contracts:validate

Cierre: Astro inicia y genera una página mínima sin warnings de configuración. No se instala ninguna dependencia fuera de package.json.

## Fase 2. Configuración y modelos

Implementar src/config, src/content.config.ts, vocabularios y colecciones CM-ANALYSIS a CM-RESOURCE. Incorporar validación de slugs, traducciones, fechas, access, license y placeholders.

Comandos:

    npm run content:validate
    npm run placeholders:check

Cierre: seed content válido pasa en modo preview y falla intencionalmente en production si contiene placeholder: true.

## Fase 3. Sistema visual

Implementar tokens VIS, fuentes, reset, base, tipografía, layouts, tablas, figuras y estados. Construir la página interna de revisión visual.

Comandos:

    npm run check
    npm run test:a11y
    npm run test:visual

Cierre: todos los tokens provienen de CSS variables, los contrastes AA pasan y no existe una galería uniforme de cajas con líneas laterales.

## Fase 4. Componentes

Implementar CMP-SHELL, CMP-EDITORIAL, CMP-DATA, CMP-RESOURCE, CMP-FILTER y CMP-TEACHING. Probar props obligatorias, estados y responsive.

Comandos:

    npm run test:unit
    npm run test:components

Cierre: cada componente tiene caso normal, caso móvil y estado vacío. Ningún componente opcional renderiza un encabezado vacío.

## Fase 5. Páginas de primer nivel

Construir Home, Work, Research, Notes, Teaching, About, CV, Contact, Resources, Search y 404. Mantener módulos de Home condicionales según PG-HOME.

Comandos:

    npm run build
    npm run search:index
    npm run links:check

Cierre: todas las rutas de primer nivel existen, la búsqueda devuelve resultados y la Home funciona con el seed mínimo.

## Fase 6. Páginas de detalle

Construir Analysis, Project, Research, Note, Course, Week, Material, Valuation y Data Product. Implementar related work por metadata.

Comandos:

    npm run content:validate
    npm run build
    npm run routes:check

Cierre: cada fixture requerido por QA-DETAIL compila y solo muestra bloques con contenido.

## Fase 7. Pipeline académico

Conectar Quarto y R Markdown sin incorporarlos a la navegación pública. Compilar un material de muestra con el entorno existente. Copiar salidas a rutas públicas estables.

Comandos:

    npm run academic:validate
    npm run academic:build
    npm run materials:check

Cierre: Course, Week y Material enlazan PDF, HTML y archivos fuente permitidos. No se instala un paquete durante el build.

## Fase 8. Bilingüismo y distribución

Implementar idioma según APP-001. Mientras esté pendiente, usar una opción temporal en configuración y bloquear release. Generar canonical, hreflang solo para equivalentes, feeds, sitemap, robots y Open Graph.

Comandos:

    npm run seo:check
    npm run routes:check
    npm run placeholders:check

Cierre: no hay hreflang huérfano, contenido duplicado sin canonical ni slug traducido sin redirect.

## Fase 9. Rendimiento, accesibilidad y seguridad

Ejecutar QA automatizado y manual en 1440, 1200, 768, 480 y 360 px. Verificar teclado, focus, reduced motion, gráficos, tablas y descargas.

Comandos:

    npm run test:a11y
    npm run test:responsive
    npm run test:visual
    npm run perf:budget
    npm audit --omit=dev

Cierre: se cumplen todos los umbrales TECH-PERF y no existen fallos críticos o altos.

## Fase 10. Despliegue reproducible

Ejecutar el workflow localmente cuando sea posible, revisar base path y desplegar un artefacto de Pages.

Comandos:

    npm ci
    npm run ci
    npm run build

Cierre: el artefacto estático contiene búsqueda, sitemap, feeds, 404, assets y materiales. La acción usa npm ci y no modifica lockfiles.

## Fase 11. Revisión final

1. Ejecutar todas las pruebas de 11-acceptance-qa.yml.
2. Comparar el repositorio con 09-repository-manifest.yml.
3. Buscar placeholders, próximamente, TODO, enlaces vacíos y encabezados sin cuerpo.
4. Revisar Home y Research contra su dirección editorial.
5. Preparar un informe de cumplimiento con requisito, evidencia y resultado.

Comandos:

    npm run ci
    npm run manifest:check
    rg -n "próximamente|coming soon|TODO|placeholder" src public

Cierre final: cero fallos required_v1, cero placeholders de producción y todas las decisiones approval_required resueltas o documentadas como bloqueantes de release.

