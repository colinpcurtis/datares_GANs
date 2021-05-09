MODEL=CycleGAN
EPOCHS=40
LOGS=logs
SAVE_PATH=trainedModels
DATASET=/monet2photo
# only change the arguments above
# invoke the "make" command to run the model

train:
	python3 run.py -m $(MODEL) -e $(EPOCHS) -l $(LOGS) -s $(SAVE_PATH) -d $(DATASET)

dataset: 
	python3 fetch_dataset.py
