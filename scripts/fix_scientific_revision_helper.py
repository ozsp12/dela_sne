"""Patch transient LaTeX replacement details before running the revision helper."""
from pathlib import Path

path = Path(__file__).with_name("apply_scientific_revision.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    'text = re.sub(r"\\\\title\\{\\\\textbf\\{.*?\\}\\}", rf"\\\\title{{\\\\textbf{{{title}}}}}", text, count=1)',
    'text = re.sub(r"\\\\title\\{\\\\textbf\\{.*?\\}\\}", lambda _m: rf"\\title{{\\textbf{{{title}}}}}", text, count=1)',
)
text = text.replace(
    'text = re.sub(r"\\\\begin\\{abstract\\}.*?\\\\end\\{abstract\\}", abstract, text, count=1, flags=re.S)',
    'text = re.sub(r"\\\\begin\\{abstract\\}.*?\\\\end\\{abstract\\}", lambda _m: abstract, text, count=1, flags=re.S)',
)
text = text.replace(
    'response_fig_end = r"\\\\label{fig:sne_neighbor_response}\\n\\\\end{figure}"',
    'response_fig_end = r"\\label{fig:sne_neighbor_response}" + "\\n" + r"\\end{figure}"',
)
path.write_text(text, encoding="utf-8")
