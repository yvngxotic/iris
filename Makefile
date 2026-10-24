.PHONY: dev run sim check calib test key tune demo lint help

dev run:
	./iris.sh dev

sim:
	./iris.sh sim

check:
	./iris.sh check

calib:
	./iris.sh calib

test:
	./iris.sh test

key:
	./iris.sh key

tune:
	./iris.sh tune

demo:
	./iris.sh demo

lint:
	./iris.sh lint

help:
	@./iris.sh help
