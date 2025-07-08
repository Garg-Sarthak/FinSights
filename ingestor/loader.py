import pymupdf
import logging

logger = logging.Logger(__name__)

def extract_text_from_pdf(file_path:str) -> str:
    print(f"loading file {file_path} : begin")
    # print(f"loading file {file_path} : begin")
    doc = pymupdf.open(filename=file_path)
    text = ""
    for page in doc:
        text += str(page.get_text().encode("utf-8"))

    # logger.log(1,msg=f"loading file {file_path} : end")
    print(f"loading file {file_path} : end")
    return text


if __name__ == "__main__":
    doc=pymupdf.open("../aapl_2024.pdf")
    for page in doc:
        print(str(page.get_text().encode("utf-8")))
        print(type(str(page.get_text().encode("utf-8"))))
        break
    # extract_text_from_pdf("../aapl_2024.pdf")