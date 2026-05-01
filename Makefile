PHONY: install

install:
	npm install
	mkdir -p ~/.pi/agent/extensions
	ln -sf ${PWD} ~/.pi/agent/extensions/purroxy
