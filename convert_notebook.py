from nbconvert import MarkdownExporter
from nbconvert.preprocessors import Preprocessor
from pathlib import Path
from traitlets.config import Config
import nbformat
import re
import os

class CustomPreprocessor(Preprocessor):
    """Remove blank code cells and unnecessary whitespace."""
    def preprocess(self, nb, resources):
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and not cell.source:
                nb.cells.pop(index)
            else:
                nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
        return nb, resources

    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == 'code':
            cell.source = cell.source.strip()
        return cell, resources

def doctor(string: str) -> str:
    post_code_newlines_patt = re.compile(r'(```)(\n+)')
    inter_output_newlines_patt = re.compile(r'(\s{4}\S+)(\n+)(\s{4})')

    post_code_filtered = re.sub(post_code_newlines_patt, r'\1\n\n', string)
    inter_output_filtered = re.sub(inter_output_newlines_patt, r'\1\n\3', post_code_filtered)

    return inter_output_filtered

def make_yaml_header(**kwargs):
    header = '---\n'
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float)):
            header += '{}: {}\n'.format(key, value)
        else:
            header += '{}:\n'.format(key)
            for item in value:
                header += '  - {}\n'.format(item)
    header += '---\n'
    return header

def notebook_to_markdown(path, date, slug, **kwargs):
    path_nb = Path(path)
    path_out = Path('content/posts') / date.split('-')[0] / date.split('-')[1] / slug
    path_post = Path('content/posts') / (date + '-' + slug + '.md')

    assert path_nb.exists(), f"Notebook file {path} does not exist."
    assert path_post.parent.exists(), f"Post directory {path_post.parent} does not exist."
    assert re.match(r'[0-9]{4}-[0-1][0-9]-[0-3][0-9]', date), 'Incorrect date format, need YYYY-MM-DD'

    with path_nb.open(encoding='utf-8') as fp:
        notebook = nbformat.read(fp, as_version=4)

    c = Config()
    c.MarkdownExporter.preprocessors = [CustomPreprocessor]
    markdown_exporter = MarkdownExporter(config=c)

    markdown, resources = markdown_exporter.from_notebook_node(notebook)
    md = doctor(markdown)

    yaml = make_yaml_header(date=date, slug=slug, **kwargs)
    md = yaml + md

    if 'outputs' in resources:
        if not path_out.exists():
            path_out.mkdir(parents=True)
        for key, data in resources['outputs'].items():
            img_path = path_out / key
            with img_path.open('wb') as img_file:
                img_file.write(data)
            print(f"Saved image {img_path}")  # Debug print statement
        
        # Actualizar las rutas de las imágenes en el contenido del Markdown
        img_folder = f'/{date.split("-")[0]}/{date.split("-")[1]}/{slug}'
        md = md.replace('](output_', f']({img_folder}/output_')

    with path_post.open('w', encoding='utf-8') as f:
        f.write(md)
    print(f"Markdown file created at {path_post}")  # Debug print statement

if __name__ == "__main__":
    path = r'C:/Users/Asus/quickstart/random_forest_estimator.ipynb'
    notebook_to_markdown(path=path,
                         date='2024-06-15',
                         slug='post_with_jupyter',
                         title='Random Forest Estimator',
                         author='Hamilton Taveras',
                         categories=['Data Science', 'Blogging'],
                         tags=['Jupyter', 'Hugo', 'Python', 'Data Science'],
                         summary='Blogging with Jupyter notebooks and Hugo.',
                         thumbnailImagePosition='left',
                         thumbnailImage='https://example.com/path/to/image.jpg')