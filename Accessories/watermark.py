from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from random import randrange

@staticmethod
def addWatermark(path_to_image, watermark_text = "kotki sa\n super"):
    main = Image.open(path_to_image)

    watermark = Image.new("RGBA", main.size)
    waterdraw = ImageDraw.ImageDraw(watermark, "RGBA")
    
    width, height = main.size

    start_point = tuple((randrange(1, int(width/2)), randrange(1, int(height/2))))
    font = ImageFont.truetype("calibri.ttf", int(width/randrange(5, 10)))
    waterdraw.text((start_point), watermark_text, font=font)
    watermask = watermark.convert("L").point(lambda x: min(x, randrange(100, 200)))
    watermark.putalpha(watermask)
    main.paste(watermark, None, watermark)
    main = main.save(path_to_image) # to do zmiany!!!

    # plt.imshow(main)
    # plt.show()

if __name__ == '__main__':
    for i in range(10):
        addWatermark(path_to_image = 'E:/Desktop/PGa/Semestr_9/PUG/Datasets/Cats/test/Calico/44065654_53030.jpg')