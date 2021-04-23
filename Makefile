MODEL=CycleGAN
EPOCHS=10
LOGS=logs
SAVE_PATH=TrainingModels
DATASET=/monet2photo
# only change the arguments above
# invoke the "make" command to run the model

train:
	python3 run.py -m $(MODEL) -e $(EPOCHS) -l $(LOGS) -s $(SAVE_PATH) -d $(DATASET)
