import pymupdf
from tabula.io import read_pdf
doc = pymupdf.open("aapl.pdf")

out = open("output.txt","wb")

# tables = doc[31].find_tables()

for page in doc:
    tables = page.find_tables()
    if tables.tables:
        print("yes")
        for table in tables.tables:
            print(table.extract())
    else:
        print("no")