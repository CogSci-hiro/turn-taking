@ECHO OFF
set SPHINXBUILD=sphinx-build
set SOURCEDIR=source
set BUILDDIR=_build

if "%1" == "" goto help
if "%1" == "help" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end

