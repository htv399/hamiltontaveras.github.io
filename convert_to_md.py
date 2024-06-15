import nbformat
from nbconvert import MarkdownExporter
import os
import shutil

# Cargar el notebook con codificación UTF-8
with open('random_forest_estimator.ipynb', encoding='utf-8') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convertir el notebook a Markdown
markdown_exporter = MarkdownExporter()
(body, resources) = markdown_exporter.from_notebook_node(notebook_content)

# Crear carpeta para guardar las imágenes si no existe
img_folder = 'content/posts/random_forest_estimator_images'
os.makedirs(img_folder, exist_ok=True)

# Guardar imágenes en la carpeta específica
for img_name, img_data in resources['outputs'].items():
    img_path = os.path.join(img_folder, img_name)
    with open(img_path, 'wb') as img_file:
        img_file.write(img_data)

# Reemplazar las rutas de las imágenes en el contenido del Markdown
body = body.replace('](output_', f']({img_folder}/output_')

# Guardar el contenido convertido a un archivo Markdown
with open('content/posts/random_forest_estimator.md', 'w', encoding='utf-8') as f:
    f.write(body)

print("Notebook convertido a Markdown con éxito.")