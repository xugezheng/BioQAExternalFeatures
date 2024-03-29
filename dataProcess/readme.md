This repository is for data process scripts, with which we can extract the required external features (e.g., POS, NER, BioNER) in original text.

The repo structure is: 
```
dataProcess
├─ bioner.py # Extract BioNER features from https://bern.korea.ac.kr, provided by DMIS-Lab
├─ posTreat.py # Extract the POS features from SQuAD-like file, utilizing NLTK
├─ pos_ner_treat.py # Extract the NER features from SQuAD-like file, utilizing spacy
└─ readme.md
```

For the NER feature, we leveraged [spaCy tool](https://spacy.io/) to extract, and we extracted POS feature 
by [NLTK library](https://www.nltk.org/). As for compared BioNER feature, we utilized directly [the BERN model](https://github.com/dmis-lab/bern) and 
its web service.
