- parameters.py : Contient les hyper-paramètres du modèle. Pour chaque configuration un numéro de session est créé qui permet de charger un modèle pour poursuivre l'entraînement ou décoder les réponses du chatbot. Lancer ce script pour consulter les paramètres d'une session existante.

	Les hyper-paramètres ajustables sont les suivants:
		config['VOCAB_SIZE'] : taille du vocabulaire (en comptant les tokens spéciaux BOS, EOS, UNK)
		config['MAX_UNK'] : nombre de mots inconnus maximum par réplique, les conversations contenant des répliques avec trop de mots inconnus sont ignorées


		config['INPUT_EMBEDDING_SIZE'] : dimension des word-embeddings
		config['ENCODER_HIDDEN_UNITS'] : dimension des cellules LSTM

		config['ATTENTION_MECHANISM'] : booléen pour activer / desactiver l'attention mechanism

		config['HIST_LENGTH'] : nombre de répliques prises en compte dans le contexte (historique)
		config['DECODER_LENGTH'] : nombre de mot maximal par réplique générée

	
		config['PRETRAINED_EMBEDDINGS'] : booléen pour utiliser / ne pas utiliser les word-embeddings de GoogleNews pour l'initialisation

		config['INITIAL_LR'] : Learning rate initial
		config['N_EPOCH_HALF_LR'] : Nombre d'epoch pour que le learning rate soit divisé par deux

		config['INITIAL_TF'] : Taux de teacher forcing initial
		config['N_EPOCH_HALF_TF'] : Nombre d'epoch pour que le taux de teacher forcing soit divisé par deux

		config['CORPUS_PATH'] : chemin vers le corpus

		config['REG_OUT'] : coefficient de régularisation
		config['REG_AM'] : coefficient de régularisation sur les poids de l'attention mechanism
		config['DROPOUT_KEEP_PROB'] : probabilité de conserver une connexion pour le dropout

		config['LOSS_TYPE'] : loss utilisée
			1: loss calculée sur les tokens précédant à la fois le premier token EOS de la réponse générée et le token EOS de la réponse cible
			2: loss calculée sur les tokens pour lesquels la réponse générée ET la réponse cible ont une valeur differente du token PAD
			3: loss calculée sur l'ensemble des tokens, y compris les PAD
			4: loss non testée (simule des prédictions du token vide dès qu'un premier EOS est généré) (implémentation à vérifier, dans generator_hred_am_bs.py)

	Au lancement de main.py ou de chat.py, l'utilisateur doit rentrer un numéro de session. Si ce numéro correspond à une session déjà enregistrée, la configuration correspondante est chargée, sinon, une nouvelle configuration est créée.
	Pour forcer l'écrasement d'une configuration déjà existante, mettre load_conf = False.
	RESUME_TRAINING permet de spécifier, dans le cas d'une configuration déjà existante, si l'entraînement doit être repris où il avait été arrêté ou s'il doit repartir de zéro.

- main.py : permet d'entrainer un modèle. Dans l'ordre, rentrer la configuration dans parameters.py puis lancer main.py et entrer un numéro de session.

- preprocessing.py : contient mes fonctions nécéssaires au traitemement du corpus

- plt.py : permet de tracer les courbes de loss sur les set d'entrainement et de validation, pour la session demandée

- chat.py : simule une conversation avec le chatbot. Entrer un numéro de session et un numéro de checkpoint pour charger le modèle souhaité

- visualize_attention.py : permet de visualiser le comportement de l'attention mechanism. Lancer ce script juste après avoir généré une réplique par le chatbot permet de visualiser le comportement de l'attention lors de la génération de cette dernière.



FORMAT DU CORPUS:  
Les conversations sont des successions de répliques séparées par des retours à la ligne. Les conversations sont séparées par un double retour à la ligne.


réplique 1.1
réplique 1.2
réplique 1.3
réplique 1.4

réplique 2.1
réplique 2.2
réplique 2.3

réplique 3.1
réplique 3.2

réplique 4.1
réplique 4.2
réplique 4.3
