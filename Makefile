.PHONY: clean clean-test clean-pyc clean-build docs help docker
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean:  ## clean all build, python, and testing files
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .tox/
	rm -fr .coverage
	rm -fr coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache

build: ## run tox / run tests and lint
	tox

gen-docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/morflowgenesis*.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ morflowgenesis **/tests/
	$(MAKE) -C docs html

docs: ## generate Sphinx HTML documentation, including API docs, and serve to browser
	make gen-docs
	$(BROWSER) docs/_build/html/index.html

local-registry = docker-modeling-local.artifactory.corp.alleninstitute.org
version = v0.0.0
docker:
	DOCKER_BUILDKIT=1 \
	docker build \
	    --ssh github=$(HOME)/.ssh/id_rsa \
	    --build-arg version=$(version) \
	    -t $(local-registry)/mlops/morflowgenesis:$(version) .
	docker push $(local-registry)/mlops/morflowgenesis:$(version)
	docker tag $(local-registry)/mlops/morflowgenesis:$(version) $(local-registry)/mlops/morflowgenesis:latest
	docker push $(local-registry)/mlops/morflowgenesis:latest
