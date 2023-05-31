# Zdo semestrální práce
Repozitář k semestrální práci z předmětu KKY/ZDO. 

## Instalace
1. naklonování repozitáře
2. `conda env create -f environment.yml`
3. `conda activate zdo`
4. Ve root adresáři repozitáře vytvořit složku `checkpoints`
5. Do nově vytvořeného adresáře stáhnout checkpoint file `incision_only.pth` modelu z odkazu https://drive.google.com/drive/folders/179K6GUQ3xrSZvodGm9f7N_nB-ZeUBKXw?usp=sharing
6. python run.py output.json incision001.jpg
