PHONY: install

install:
	npm install
	mkdir -p ~/.pi/agent/extensions
	ln -s $(pwd) ~/.pi/agent/extensions/purroxy
