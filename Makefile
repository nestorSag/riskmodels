.PHONY: setup docs tests format build help

define verify_install
	if ! [ "$$(pip list | grep $(1) -c)" = "1" ]; then\
	  if [ $(1) = "riskmodels" ]; then\
	    pip install -e . ;\
	  else\
	    pip install $(1);\
	  fi;\
	fi
endef

html-docs: ## Updates html documentation
	@$(call verify_install, pdoc3);\
	rm -rf docs/* && pdoc3 --html -c latex_math=True -o docs riskmodels && mv docs/riskmodels/* docs/ && rm -rf docs/riskmodels;\

pdf-docs: ## Generates pdf documentation
	@$(call verify_install, pdoc3);\
  echo "for this to work, pandoc needs to be installed."
	@pdoc3 --pdf --config='docformat="google"' -c latex_math=True -c show_source_code=False riskmodels > docs.md
	@pandoc --metadata=title:"riskmodels package" \
	--from=markdown+abbreviations+tex_math_single_backslash \
	--pdf-engine=xelatex --variable=mainfont:"DejaVu Sans" \
	--toc \
	--toc-depth=4 \
	--output=docs.pdf  docs.md

tests: ## Tests package
	@$(call verify_install, pytest);
	@$(call verify_install, riskmodels);
	@pytest tests/

format: ## Formats modules code using Black
	@$(call verify_install, black);\
	black riskmodels/*

# lint: ## Formats modules code using Black
# 	@$(call verify_install, flake8);\
# 	flake8 --ignore=E501 riskmodels

build: ## Bundles objects for release to PyPI
	@$(call verify_install, build);\
	python -m build --sdist --outdir dist/
	#twine check dist/* && twine upload dist/*

help: ## Shows Makefile's help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)
