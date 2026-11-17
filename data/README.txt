The 10 sample cases as plain files. The pipeline reads this folder.
documents/*.txt  the 20 policy documents, exactly as the pipeline ingests them
gold.json        answer key per case (labels, route, rule, final status). Edit it to change expected answers.
gold.csv         the same answer key as a flat table for Excel (written by the generator, not read)
Regenerate everything: python -m pcd.data.generate_cases --out data --seed 7
