.PHONY: setup docs tests format bundle help

define verify_install
	if ! [ "$$(pip list | grep $(1) -c)" = "1" ]; then\
	  if [ $(1) = "riskmodels" ]; then\
	    pip install -e . ;\
	  else\
	    pip install $(1);\
	  fi;\
	fi
endef

docs: ## Updates documentation
	$(call verify_install,pdoc3);\
	rm -rf docs/* && pdoc --html -c latex_math=True -o docs riskmodels && mv docs/riskmodels/* docs/ && rm -rf docs/riskmodels;\

tests: ## Tests package
	@$(call verify_install, pytest);
	@$(call verify_install, riskmodels);
	@pytest tests/

format: ## Formats modules code using Black
	$(call verify_install, black);\
	black riskmodels/*

bundle: ## Bundles objects for release to PyPI
	$(call verify_install, twine);\
	python setup.py sdist 
	twine check dist/* && twine upload dist/*

help: ## Shows Makefile's help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)