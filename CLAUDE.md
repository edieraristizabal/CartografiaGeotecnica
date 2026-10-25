# CLAUDE.md

Guía para Claude Code (y para cualquier persona) que actualice este repositorio. **Cada cambio que se haga en el repositorio debe registrarse al final, en "Registro de cambios".**

## Qué es

Curso **Cartografía Geotécnica** (Universidad Nacional de Colombia, sede Medellín, Prof. Edier Aristizábal). Un solo repositorio contiene tres productos:

| Producto | Fuente en `master` | URL publicada |
|---|---|---|
| Página del curso (Jekyll, tema minimal) | `index.md`, `_config.yml`, programa en PDF | https://edieraristizabal.github.io/CartografiaGeotecnica/ |
| Diapositivas (Reveal.js) | `html/CartografiaGeotecnica.html` y sus recursos en `html/` | https://edieraristizabal.github.io/CartografiaGeotecnica/html/CartografiaGeotecnica.html |
| Libro (Jupyter Book 1.x) | `libro/` | https://edieraristizabal.github.io/CartografiaGeotecnica/libro/ |

Remoto: https://github.com/edieraristizabal/CartografiaGeotecnica

## Resumen del contenido

El curso de Cartografía Geotécnica está orientado para estudiantes de ingeniería que deseen adquirir conocimientos sobre métodos para la evaluación y zonificación de la susceptibilidad y amenaza por movimientos en masa en zonas de montaña, con enfoque en ordenamiento territorial. 
El curso es teórico - práctico, en el cual se hace una revisión detallada del estado del arte y las diferentes metodologías de zonificación utilizadas alrededor del mundo, incluyendo métodos de aprendizaje automático (*machine learning*), minería de datos (*data mining*), y análisis espacial de datos.
El alcance de este curso es aprender a construir modelos para cartografiar y construir mapas de susceptibilidad y amenaza como herramienta para el ordenamiento del territorio.

El libro combina capítulos en Markdown y notebooks de Python con salidas guardadas (susceptibilidad, amenaza, modelos físicos SHALSTAB/TRIGRS, flujos y roca). Los datos de los notebooks están en `libro/data/` y se leen desde `https://raw.githubusercontent.com/edieraristizabal/CartografiaGeotecnica/master/libro/data/...`. Cada capítulo tiene su propia bibliografía (`{bibliography}` filtrada por documento) y además existe la general `100_references.md`; por eso se silencian en `libro/_config.yml` las advertencias de citas duplicadas.

Contenido del libro (orden de `libro/_toc.yml`; el título es el primer encabezado de cada archivo):

- `intro`: portada
- `00_medialuna`: Media Luna (1954)
- `00_introduccion`: Introducción
- `01_normatividad`: Amenaza & ordenamiento territorial
- `02_estadoarte`: Estado del arte
- `03_metodos`: Caracterización de movimientos en masa
- `04_susceptibilidad`: Susceptibilidad y amenaza
- `05_inventario`: Inventario de movimientos en masa
- `06_variables`: Variables condicionantes
- `07_detonante`: Factores detonantes
- `08_heuristico`: Métodos basados en el conocimiento
- `09_estadistico`: Métodos basados en datos
- `10_fisico`: Métodos con base física
- `11_evaluacion`: Evaluación del modelo
- `12_amenaza`: Análisis del factor detonante
- `13_flujos`: Movimientos en masa tipo flujos
- `14_roca`: Movimientos en masa en roca
- `15_python`: Python
- `100_references`: Referencias

## Ramas: dónde hacer pull y push

**Siempre se trabaja en `master`.** Ese es el único lugar donde se edita, se hace `git pull` y `git push`, tanto para cambios de la página como de las diapositivas o del libro.

**Nunca editar ni hacer push a `gh-pages` a mano.** Esa rama la genera el workflow `.github/workflows/deploy.yml` en cada push a `master`: construye Jekyll en `_site/`, construye el libro y lo copia en `_site/libro/`, y publica todo en `gh-pages` con `force_orphan` (cualquier cambio manual en `gh-pages` se pierde en el siguiente despliegue).

GitHub Pages sirve un solo sitio por repositorio; por eso la página y el libro se combinan en el Action. En **Settings → Pages** la fuente debe ser *Deploy from a branch*, rama `gh-pages`, carpeta `/ (root)`.

Flujo normal:

```bash
git pull origin master
# ... editar ...
# ... agregar una línea al Registro de cambios de este archivo ...
git add -A
git commit -m "Descripción del cambio"
git push origin master      # el Action publica en 2 a 5 minutos (pestaña Actions)
```

## Estructura

```
.github/
    workflows/
.gitignore
CLAUDE.md
Programa_CartoGeotecnica.pdf
Programa_CartoGeotecnica.tex
README.md
_config.yml
_layouts/
    default.html
build/
docs/
html/
    .github/
    CartografiaGeotecnica.html
index.md
libro/
```

Material fuente que **no** se sube (libros guía, presentaciones originales, documentos con derechos): está en `C:\Users\edier\Documents\CATEDRA\CARTOGRAFIA GEOTECNICA`, fuera de este repositorio.

## Cómo actualizar

**Página del curso.** Editar `index.md`. Los enlaces internos deben ser relativos (`Programa.pdf`, `html/...`), nunca empezar por `/`, porque el sitio vive en `/CartografiaGeotecnica/`. El libro se enlaza como `https://edieraristizabal.github.io/CartografiaGeotecnica/libro/<capítulo>.html`. Si cambia la evaluación, actualizarla también en el programa (`.tex` + PDF) y en las diapositivas.

**Diapositivas.** Editar `html/CartografiaGeotecnica.html` directamente (Reveal.js; ecuaciones con MathJax). Previsualizar con `python -m http.server` desde la raíz y abrir `http://localhost:8000/html/CartografiaGeotecnica.html`.

**Libro.** Capítulos en MyST Markdown (`.md`) o notebooks (`.ipynb`) dentro de `libro/`; un capítulo nuevo debe agregarse a `libro/_toc.yml`; las citas `{cite:t}` / `{cite:p}` deben existir en el archivo `.bib` del libro.

- **El Action no ejecuta los notebooks** (`execute_notebooks: "off"`): publica las salidas guardadas en cada `.ipynb`. Si se cambia código, ejecutar el notebook localmente y guardarlo con sus salidas antes del commit.
- `libro/requirements.txt` contiene solo lo necesario para construir el libro; `libro/requirements-notebooks.txt` (si existe) lista las librerías para ejecutar los notebooks.
- Si git falla con `bad object refs/desktop.ini` o `failed to perform geometric repack`, es porque Windows/Google Drive creó archivos `desktop.ini` dentro de `.git/`: borrarlos con `find .git -name desktop.ini -delete` y repetir el comando.
- Compilar localmente: `C:\Users\edier\.jbenv\Scripts\jupyter-book.exe build libro` y abrir `libro/_build/html/index.html`.



## Preferencias del autor

- Proponer los cambios de contenido del curso y esperar aprobación antes de editarlos.
- Prosa en párrafos, español, sin rayas (em dash): usar comas o paréntesis.
- Las figuras nuevas se buscan primero en los tableros de Pinterest del autor.

## Pendientes conocidos

- Cambiar en Settings → Pages la fuente a la rama `gh-pages`.
- Borrar el repositorio viejo `Libro_cartoGeotecnia` cuando se verifique el libro nuevo. Ojo: en ese repositorio quedaron las ramas de respaldo `respaldo-local-feb2026` (versión local) y la historia de marzo; al borrarlo se pierden. Si se quieren conservar, descargarlas antes.
- Advertencias menores del libro (contenido, no se corrigieron sin aprobación): niveles de encabezado en `15_python.ipynb` y `09_estadistico.ipynb`; campos faltantes en `sidle2006` (publisher) y `malamud2004` (journal) de `100_references.bib`.

## Registro de cambios

Cada cambio al repositorio se anota aquí, el más reciente primero, con el formato `- **AAAA-MM-DD**: qué se cambió y por qué (archivos principales).` Esta sección es la memoria del curso entre sesiones: leerla antes de empezar a trabajar.

- **2026-10-25**: Revisión de figuras de la presentación en el navegador; se reemplazaron 20 imágenes rotas de html/CartografiaGeotecnica.html: 9 recuperadas del Internet Archive, 3 fotografías de Wikimedia Commons (reptación, tipos de movimientos en masa del USGS, lluvia en Medellín), 7 figuras nuevas generadas con matplotlib (combinación lineal ponderada, clasificación de Varnes y Cruden y Varnes, matriz de correlación, volcamiento, métodos continuos, propagación tipo Flow-R, escala de Saaty) y una miniatura de Wikimedia con ancho no estándar corregida; todas fijadas en el tablero cartogeotecnia de Pinterest. Las figuras generadas quedan también en `html/img/figuras/`; la presentación las enlaza desde `i.pinimg.com`.
- **2026-10-25**: `README.md` con la sección "Enlaces del curso" (página, presentación y libro publicados) y el enlace a la página personal https://edieraristizabal.github.io/.
- **2026-10-24**: Interfaz en español: `_layouts/default.html` traducido ("Ver el proyecto en GitHub", "Este proyecto es mantenido por", `lang` = es), reemplaza el texto en inglés del tema minimal de Jekyll; `language: es` en `libro/_config.yml` para la interfaz del libro.
- **2026-10-24**: Revisión de idioma: se tradujeron al español 17 bloques en inglés de `html/CartografiaGeotecnica.html` (características de Python, métodos de inventario, distribución de Poisson, citas de Laplace y Poincaré, SINMAP, análisis pseudoestático y de Newmark, curvas magnitud-frecuencia) y se corrigió una etiqueta `</span>` mal cerrada. El libro ya estaba en español. Se mantienen en inglés títulos de libros, enlaces y tweets incrustados.
- **2026-10-24**: CLAUDE.md creado con resumen del contenido, estructura, flujo de ramas, pendientes y este registro.
- **2026-10-24**: Unificación en un solo repositorio: el libro (antes `Libro_cartoGeotecnia`) se copió a `libro/` con `execute_notebooks: off`; URLs de datos en 7 notebooks cambiadas de `Libro_cartoGeotecnia/master/data` a `CartografiaGeotecnica/master/libro/data`; workflow `deploy.yml`; `_config.yml` con `baseurl: /CartografiaGeotecnica`; enlaces de `index.md` relativos y apuntando a `/CartografiaGeotecnica/libro/`; requisitos para ejecutar notebooks movidos a `libro/requirements-notebooks.txt`.
- **2026-10-24**: Fusión de dos versiones divergentes del libro (historias no relacionadas): la local (febrero, mejoras didácticas, correcciones ortográficas, párrafos de contexto colombiano, etiquetas de figuras sin espacios, tres .tif) y la de GitHub (marzo, citas `{cite:t}`, clasificación de modelos físicos de Sachinthaka 2025 en `10_fisico`, URLs `raw.githubusercontent`, bibliografía por capítulo, salidas de notebooks). Celdas de código de GitHub; texto fusionado línea por línea; bibliografía como unión (198 entradas); claves de cita normalizadas a las del `.bib` (en minúscula).
- **2026-10-24**: Commit de respaldo de los cambios locales pendientes del sitio (36 archivos) antes de unificar; limpieza de `desktop.ini` dentro de `.git/` y de un `index.lock` huérfano.
