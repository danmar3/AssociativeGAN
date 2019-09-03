sphinx-apidoc --separate --force -d 2 -o source/ ../../acgan
sphinx-build -E -a -b html source ../../../acgan-doc/
