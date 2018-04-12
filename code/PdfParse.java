InputStream input = new FileInputStream("sample.pdf");
ContentHandler handler = new BodyContentHandler(Integer.MAX_VALUE);
Metadata metadata = new Metadata();
new PDFParser().parse(input, handler, metadata, new ParseContext());
String plainText = handler.toString();
System.out.println(plainText);