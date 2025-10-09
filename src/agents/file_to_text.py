import pytesseract
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def file_to_text():

    file_path = input("Enter File Path: ")
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image,lang='eng')

    return text