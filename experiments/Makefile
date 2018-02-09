.PHONY: clean env test 
all : clean-env env test 

NAME := winapi_deobf_experiments
PWD := $(shell pwd)
VERSION := $(shell cat VERSION)
UNAME := $(shell sh -c 'uname -s')

SITE_PACKAGES=$(PWD)/env/lib/python2.7/site-packages

env:
	virtualenv env --no-site-packages
	env/bin/pip install pip==9.0.1
	env/bin/pip install --no-cache-dir -r requirements.txt
	ln -sfn $(PWD)/src/$(NAME) $(SITE_PACKAGES)/$(NAME)

test:
	env/bin/nosetests -s tests/unit/

experiments:
	env/bin/python $(PWD)/src/$(NAME)/api.py 

build: env 
	env/bin/python setup.py sdist --formats=gztar

clean-env:
	rm -rf env/
