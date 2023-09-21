import ratio
import Preprocess
import tone
import sys


def main(file):
    ratio_ = ratio.returnRatio(file)
    tone_ = tone.tone(file)
    obesity_ = obesity.obesity(file)
    
    print(ratio_)
    print(tone_)
    print(obesity_)

if __name__ == "__main__":
    main(sys.argv[1])
