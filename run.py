import argparse
from Models.DCGAN import DCGAN

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", action="store", help="model type: DCGAN",
                        type=str, required=True)

    parser.add_argument("-e", "--epochs", action="store", help="num of epochs",
                        type=int, required=True)

    parser.add_argument("-v", "--verbose",  action="store", help="verbose: 0 for no batch stats output",
                        type=int, required=True)

    parser.add_argument("-s", "--save", action="store", help="save logs location",
                        type=str, required=True)

    # parser.add_argument(name='-s', action="--action", help="save model?", default=False, type=bool)

    args = parser.parse_args()
    print(args)

    if args.model == "DCGAN":
        model = DCGAN(args.epochs, args.verbose, args.save)
        model.train()
