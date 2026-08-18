from pathlib import Path

path = Path("paper/main.tex")
text = path.read_text(encoding="utf-8")

keywords = r"\noindent\textbf{Keywords:} statistical mechanics; Gibbs distribution; Shannon entropy; clustering; locally adaptive clustering; stochastic neighbor embedding; t-SNE; perplexity; dimensionality reduction; local metric learning."
old_glossary = keywords + "\n\n" + r"\section*{Glossary}"
new_glossary = keywords + "\n\n" + r"\clearpage" + "\n" + r"\section*{Glossary}"
if old_glossary not in text:
    raise RuntimeError("Glossary insertion point not found or already modified.")
text = text.replace(old_glossary, new_glossary, 1)

old_section = r"\section{State-space comparison and interpretation}" + "\n" + r"\label{sec:unified}"
new_section = r"\section{Feature and neighbor ensembles and the dual construction}" + "\n" + r"\label{sec:unified}"
if old_section not in text:
    raise RuntimeError("Section 7 heading not found.")
text = text.replace(old_section, new_section, 1)

old_dual = r"\section{Prospective dual feature--neighbor construction}" + "\n" + r"\label{sec:dual}" + "\n\n"
new_dual = r"\label{sec:dual}" + "\n\n"
if old_dual not in text:
    raise RuntimeError("Section 8 heading not found.")
text = text.replace(old_dual, new_dual, 1)

path.write_text(text, encoding="utf-8")
Path(__file__).unlink()
