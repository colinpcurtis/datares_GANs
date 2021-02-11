import argparse
from Models.DCGAN.DCGAN import DCGAN
from Models.ConditionalGAN.conditionalGAN import conditionalGAN
from Models.cycleGAN.cycleGAN import cycleGAN

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", action="store", help="model type: DCGAN or conditionalGAN",
                        type=str, required=False)

    parser.add_argument("-e", "--epochs", action="store", help="num of epochs",
                        type=int, required=False)

    parser.add_argument("-v", "--verbose",  action="store", help="verbose: 0 for no batch stats output",
                        type=int, required=False)

    parser.add_argument("-l", "--saveLogs", action="store", help="save logs location",
                        type=str, required=False)

    parser.add_argument("-s", "--saveModel", action="store", help="save state dict path",
                        type=str, required=False)

    parser.add_argument("-p", "--runPreprocess", action="store", help="run the preprocessing script",
                        type=bool, required=False)

    parser.add_argument("-d", "--datasetDirectory", action="store", help="directory of images for preprocessing",
                        type=str, required=False)

    args = parser.parse_args()

    if args.model == "DCGAN":
        model = DCGAN(args.epochs, args.verbose, args.saveLogs)
        model.train()

    if args.model == "conditionalGAN":
        model = conditionalGAN(args.epochs, args.saveLogs, args.saveModel)
        model.train()
        model.save_model(model.state_dict(), args.saveModel)

    if args.model == "cycleGAN":
        model = cycleGAN(args.epochs, args.saveLogs, args.saveModel, args.datasetDirectory)
        model.train()
        model.save_model(model.state_dict(), args.saveModel)

