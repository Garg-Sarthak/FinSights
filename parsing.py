import pymupdf

doc = pymupdf.open("aapl.pdf")

out = open("output.txt","wb")

for page in doc :
    text = page.get_text().encode('utf8')
    print(text)
    break
