THIS = basix-paper

.PHONY: all pdf

all: pdf

pdf:
	docker run --rm --volume $(shell pwd):/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/paperdraft
