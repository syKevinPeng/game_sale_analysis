#!/bin/bash
jupyter nbconvert game_sale_analysis.ipynb --to markdown --output "README.md"
git add README.md game_sale_analysis.ipynb
git commit -m "generat publish file"
git push