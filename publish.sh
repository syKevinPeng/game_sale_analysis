#!/bin/bash
jupyter nbconvert game_sale.ipynb --to markdown --output "README.md"
git add README.md game_sale.ipynb
git commit -m "generate publish file"
git push