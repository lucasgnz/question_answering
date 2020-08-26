# Question answering sur des documents juridiques
La solution envisagée  est d'adapter le modèle R-Net de Microsoft (https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) au domaine juridique avec une méthode de Transfer Learning utilisant un corpus de documents spécifiques au domaine ; elle s’inspire du travail réalisé dans "Two-stage synthesis networks for transfer learning in machine comprehension" (https://arxiv.org/abs/1706.09789).

L’idée est de synthétiser des triplets (contexte-question-réponse) sur un corpus constitué de documents de la SEC (Securities and Exchange Commission, https://www.sec.gov/). La deuxième étape consiste à fournir les examples de ce nouveau corpus à un modèle R-net pré-entraîné sur le
dataset SQuAD (fine-tuning).

Ce repertoire contient:
- L'implémentation pour la constitution du corpus de documents issus de SEC (https://www.sec.gov/)
- L'implémentation de R-Net et Syn-Net (Microsoft Research)
- Le code pour les expériences réalisées avec DrQA, YodaQA, AllenNLP


![Approche envisagée](https://github.com/lucasgnz/question_answering/blob/master/addventa_approche.png)
