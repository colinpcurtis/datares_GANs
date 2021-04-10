MODEL=cycleGAN
EPOCHS=10
LOGS=logs
SAVE_PATH=paintingGenerator.pt
DATASET=/monet2photo
WEIGHTS=/TrainedModels
# only change the arguments above
# invole the "make" command to run the model

train:
	python3 run.py -m $(MODEL) -e $(EPOCHS) -l $(LOGS) -s $(SAVE_PATH) -d $(DATASET) -t $(WEIGHTS)
