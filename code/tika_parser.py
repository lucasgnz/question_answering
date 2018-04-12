from tika import parser

parsedPDF = parser.from_file("Example.pdf", xmlContent=True)

print(parsedPDF["content"])