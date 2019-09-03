export SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance,inherited-members
sphinx-apidoc --separate --force -d 2 -o source/ ../../acgan
sphinx-build -E -a -b html source ../../../acgan-doc/
