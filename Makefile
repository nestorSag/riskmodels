

docs: ##Update documentation
	pdoc --html -c latex_math=True -o docs riskmodels && mv docs/riskmodels/* docs/ && rm -rf docs/riskmodels

