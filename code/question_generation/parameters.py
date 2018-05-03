SESSION = ""
while(SESSION.replace(" ","") == ""):
	SESSION = raw_input("SESSION number:")
SESSION=int(SESSION)
import tensorflow as tf
import os
import pickle

RESUME_TRAINING = False
RESUME_CKPT = 0
RESUME_LR = 0
load_conf = False

#CONFIG_TF = tf.ConfigProto(allow_soft_placement = False, log_device_placement = False)
#CONFIG_TF.gpu_options.visible_device_list = "0"

if os.path.exists("sessions/"+str(SESSION)+"/config.p") == True and load_conf==True:
	config = pickle.load(open( "sessions/"+str(SESSION)+"/config.p", "rb" ) )
	print("Loaded configuration:", config)

else:
	config={}
	
	config['PAD'] = 0
	config['BOS'] = 1
	config['EOS'] = 0
	config['UNK'] = 2

	config['N_SPECIAL_TOKENS'] = 3

	config['GEN_VOCAB_SIZE'] = 10 + config['N_SPECIAL_TOKENS']

	config['EMBED_SIZE'] = 300
	config['HIDDEN_UNITS'] = 100
	config['ATTENTION_UNITS'] = 30

	config['PRETRAINED_EMBEDDINGS'] = False

	config['INITIAL_LR'] = 1e-2
	config['N_EPOCH_HALF_LR'] = 5


	if os.path.exists("sessions/"+str(SESSION)) == False:
		os.system("mkdir sessions/"+str(SESSION))
	pickle.dump( config, open( "sessions/"+str(SESSION)+"/config.p", "wb" ) )

	print("New configuration (saved):", config)

if SESSION <= 0:
	config['PRETRAINED_EMBEDDINGS']=False

