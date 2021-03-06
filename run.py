import argparse
from Models.DCGAN.DCGAN import DCGAN
from Models.ConditionalGAN.conditionalGAN import conditionalGAN
from Models.cycleGAN.CycleGAN import CycleGAN

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", action="store", help="model type",
                        type=str, required=False)

    parser.add_argument("-e", "--epochs", action="store", help="num of epochs",
                        type=int, required=False)

    parser.add_argument("-v", "--verbose",  action="store", help="verbose: 0 for no batch stats output",
                        type=int, required=False)

    parser.add_argument("-l", "--saveLogs", action="store", help="save logs location",
                        type=str, required=False)

    parser.add_argument("-s", "--saveModel", action="store", 
                        help="save state dict path",
                        type=str, required=False)

    parser.add_argument("-d", "--datasetDirectory", action="store", 
                        help="directory of images for preprocessing",
                        type=str, required=False)

    parser.add_argument("-t", "--trainedWeights", action="store", 
                        help="load model state dicts from local directory",
                        type=str, required=False)

    args = parser.parse_args()

    if args.model == "DCGAN":
        model = DCGAN(args.epochs, args.verbose, args.saveLogs)
        model.train()

    if args.model == "conditionalGAN":
        model = conditionalGAN(args.epochs, args.saveLogs, args.saveModel)
        model.train()

    if args.model == "CycleGAN":
        model = CycleGAN(args.epochs, args.saveLogs, args.saveModel, args.datasetDirectory)
        model.train()
