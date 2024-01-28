import random
import os
from PIL import Image, ImageDraw, ImageFont
from random import randrange


# algorithm allowing to automatically add watermark to image with
# given chance, text and transparency
#
# in default transparency is random
def add_watermark(path_to_image, watermark_text="kotki sa\n super"):
    main = Image.open(path_to_image)

    watermark = Image.new("RGBA", main.size)
    waterdraw = ImageDraw.ImageDraw(watermark, "RGBA")
    
    width, height = main.size

    # random position, depends on image resolution
    start_point = tuple((randrange(1, int(width/2)), randrange(1, int(height/2))))
    font = ImageFont.truetype("calibri.ttf", int(width/randrange(5, 10)))
    waterdraw.text(start_point, watermark_text, font=font)

    # transparency, specified by last argument from range 0 to 255 (hidden, 100% visible)
    watermask = watermark.convert("L").point(lambda x: min(x, 255))  # randrange(100, 200)
    watermark.putalpha(watermask)
    main.paste(watermark, None, watermark)
    main.save(path_to_image)

    # add label (-4 == .jpg)
    os.rename(path_to_image, path_to_image[:-4] + '_watermark.jpg')


if __name__ == '__main__':
    path = 'E:/Desktop/PGa/Semestr_9/PUG/Datasets/Cats/test_watermark_100%_no_transparent/'
    breeds = os.listdir(path)
    percentage_of_watermarks = 0.5

    for breed in breeds:
        filenames = os.listdir(path + '/' + breed)
        i = 0
        for filename in filenames:
            if random.random() < percentage_of_watermarks:
                add_watermark(path_to_image=path + breed + '/' + filename)
                i += 1

        # summary
        print(f'{breed}: watermarked = {i}, all = {len(filenames)}, % = {i/len(filenames)}')
