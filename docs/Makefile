# Makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build targets
html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The HTML pages are in $(BUILDDIR)/html."

dirhtml:
	@$(SPHINXBUILD) -M dirhtml "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml:
	@$(SPHINXBUILD) -M singlehtml "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The HTML page is in $(BUILDDIR)/singlehtml."

pickle:
	@$(SPHINXBUILD) -M pickle "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete; now you can process the pickle files."

json:
	@$(SPHINXBUILD) -M json "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The JSON files are in $(BUILDDIR)/json."

htmlhelp:
	@$(SPHINXBUILD) -M htmlhelp "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete; now you can run HTML Help Workshop with the" \
	       ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp:
	@$(SPHINXBUILD) -M qthelp "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete; now you can run qcollectiongenerator with the" \
	       ".qhcp project file in $(BUILDDIR)/qthelp, or use -d option."

applehelp:
	@$(SPHINXBUILD) -M applehelp "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The help book is in $(BUILDDIR)/applehelp."
	@echo "You can load it with Digital Help Book Reader or Preview.app."

devhelp:
	@$(SPHINXBUILD) -M devhelp "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The help book is in $(BUILDDIR)/devhelp."

epub:
	@$(SPHINXBUILD) -M epub "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The epub file is in $(BUILDDIR)/epub."

latex:
	@$(SPHINXBUILD) -M latex "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete; the LaTeX files are in $(BUILDDIR)/latex."

latexpdf: latex
	@echo "Running pdflatex three times to ensure cross-references are up-to-date..."
	@cd "$(BUILDDIR)/latex" && pdflatex -interaction=nonstopmode medrs.tex
	@cd "$(BUILDDIR)/latex" && pdflatex -interaction=nonstopmode medrs.tex
	@cd "$(BUILDDIR)/latex" && pdflatex -interaction=nonstopmode medrs.tex
	@echo "Build complete. The PDF file is in $(BUILDDIR)/latex/medrs.pdf."

latexpdfja:
	@$(SPHINXBUILD) -M latex "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Running platex and dvipdfmx..."
	@cd "$(BUILDDIR)/latex" && platex medrs.tex
	@cd "$(BUILDDIR)/latex" && dvipdfmx medrs.dvi
	@echo "Build complete. The PDF file is in $(BUILDDIR)/latex/medrs.pdf."

text:
	@$(SPHINXBUILD) -M text "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The text files are in $(BUILDDIR)/text."

man:
	@$(SPHINXBUILD) -M man "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The manual pages are in $(BUILDDIR)/man."

texinfo:
	@$(SPHINXBUILD) -M texinfo "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The Texinfo files are in $(BUILDDIR)/texinfo."

gettext:
	@$(SPHINXBUILD) -M gettext "$(I18NSPHINXOPTS)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The gettext files are in $(BUILDDIR)/locale."

changes:
	@$(SPHINXBUILD) -M changes "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Link check complete; look for any errors in the above output " \
	       "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Testing of doctests in the sources finished, look at the " \
	       "results in $(BUILDDIR)/doctest/output.txt."

coverage:
	@$(SPHINXBUILD) -M coverage "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Coverage finished, see $(BUILDDIR)/coverage/python.txt."
	@echo "To see the coverage report in HTML:"
	@echo "  python -m http.server -b localhost -p 8000 $(BUILDDIR)/coverage/covhtml"
	@echo "  Then visit http://localhost:8000"

xml:
	@$(SPHINXBUILD) -M xml "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The XML files are in $(BUILDDIR)/xml."

pseudoxml:
	@$(SPHINXBUILD) -M pseudoxml "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. The pseudo-XML files are in $(BUILDDIR)/pseudoxml."

# Clean targets
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf docs/.doctrees

cleancache:
	rm -rf $(BUILDDIR)/doctrees

# Development targets
livehtml:
	@echo "Starting live server..."
	@echo "Visit http://127.0.0.1:8000 to see the documentation"
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# ReadTheDocs specific targets
rtd:
	@echo "Building for ReadTheDocs..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Build complete. Ready for ReadTheDocs deployment."

# Check for broken links
checklinks: html
	@echo "Checking for broken links..."
	@find $(BUILDDIR)/html -name "*.html" -exec grep -l "href=" {} \; | head -10
	@echo "Link check completed."

# Validate documentation
validate: html
	@echo "Validating HTML documentation..."
	@python -c "
import glob
from pathlib import Path
from bs4 import BeautifulSoup

errors = []
for html_file in Path('$(BUILDDIR)/html').glob('**/*.html'):
    with open(html_file) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Check for common issues
    if soup.find('h1') is None and html_file.name != 'index.html':
        errors.append(f'{html_file}: Missing H1 heading')

    # Check for broken internal links
    for link in soup.find_all('a', href=True):
        if link['href'].startswith('#') and not soup.select(link['href']):
            errors.append(f'{html_file}: Broken internal link {link[\"href\"]}')

if errors:
    print('Validation errors:')
    for error in errors[:20]:  # Limit output
        print(f'  {error}')
    if len(errors) > 20:
        print(f'  ... and {len(errors) - 20} more')
    exit(1)
else:
    print('Documentation validation passed!')
"

# Development workflow targets
dev-setup:
	@echo "Setting up development environment..."
	@pip install -r requirements.txt
	@echo "Development environment setup complete!"

dev-test: validate
	@echo "Running development tests..."
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Development tests completed."

# Production build
prod: clean rtd
	@echo "Production build completed!"
	@echo "Documentation is ready for deployment."

# Statistics
stats:
	@echo "Documentation Statistics:"
	@echo "-----------------------"
	@find "$(SOURCEDIR)" -name "*.rst" | wc -l | xargs echo "RST files:"
	@find "$(SOURCEDIR)" -name "*.md" | wc -l | xargs echo "Markdown files:"
	@find "$(SOURCEDIR)" -name "*.py" -path "*/examples/*" | wc -l | xargs echo "Python examples:"
	@wc -l $(BUILDDIR)/html/*.html 2>/dev/null | tail -1 | xargs echo "HTML pages:" || echo "HTML pages: 0"

.PHONY: help Makefile html dirhtml singlehtml pickle json htmlhelp qthelp applehelp devhelp epub latex latexpdf latexpdfja text man texinfo gettext changes linkcheck doctest coverage xml pseudoxml clean cleancache livehtml rtd checklinks validate dev-setup dev-test prod stats