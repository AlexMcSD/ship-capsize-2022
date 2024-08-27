from PIL import Image
import sys
import os

if __name__=='__main__':
    dir=sys.argv[1]
    print(dir)
    if os.path.isdir(dir):
        files = os.listdir(dir)
        for file in files:
            file = os.path.join(dir,file)
            if file.endswith('.png'):
                output=file.replace('.png','.pdf')
                im= Image.open(file)
                im.save(output)