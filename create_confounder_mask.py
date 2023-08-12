# Importing the PIL library
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import os


def create_tag(result_dir):
    white_board = Image.fromarray(np.ones((320, 320))*255)
    I1 = ImageDraw.Draw(white_board)
    myFont = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'data','font', 'FreeMonoBold.ttf'), 18)
    I1.text((20, 280), "CXR-ROOM1", font=myFont, fill=(0))
    mask = np.array(white_board)
    mask = 1 - mask/255
    path = os.path.join(result_dir, 'tag.txt')
    np.savetxt(path, mask)


def create_stripe(result_dir):
    mask = np.zeros((320, 320))
    mask[:, 20:25] = 1
    mask[:, -25:-20] = 1
    path = os.path.join(result_dir, 'hyperintensities.txt')
    np.savetxt(path, mask)

def create_dark(result_dir):
    white_board = Image.fromarray(np.ones((320, 320)) * 255)
    draw = ImageDraw.Draw(white_board)
    draw.polygon(((0, 320), (320, 320), (320, 280), (0, 310)), fill=(0))
    # white_board.show()
    mask = np.array(white_board)
    mask = 1 - mask / 255
    path = os.path.join(result_dir, 'obstruction.txt')
    np.savetxt(path, mask)


if __name__ == "__main__":
    result_dir = os.path.join(os.path.dirname(__file__), 'confounder_masks')
    os.makedirs(result_dir,exist_ok=True)
    create_tag(result_dir)
    create_stripe(result_dir)
    create_dark(result_dir)

    # mask = np.loadtxt(path)
    # white_board = Image.fromarray(mask*255)
    # white_board.show()