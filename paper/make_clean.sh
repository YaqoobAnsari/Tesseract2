#!/usr/bin/env bash
# Regenerate the non-coloured copy from the coloured working file.
# Run after every editing session; never edit the clean copy by hand.
set -e
cd "$(dirname "$0")"
sed 's/\\revcolortrue/\\revcolorfalse/' revision_macros.tex > .revision_macros_clean.tex
sed 's/\\input{revision_macros}/\\input{.revision_macros_clean}/' main_revision_1.tex > main_revision_1_clean.tex
echo "main_revision_1_clean.tex regenerated."
