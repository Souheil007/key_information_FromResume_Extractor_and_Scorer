import utils
text = ""
pdf_path = './arabicCV.pdf'  # Or a BytesIO object if PDF is loaded into memory

for extracted_text in utils.extract_text_from_pdf(pdf_path):
    text += extracted_text + "\n"

# Assuming you wanted to store the text under the key "content" in dictionary d
d = {"content": text}
annotations=[]
utils.extract_email(text,annotations)



#Combine all annotations
data = {
    "content": text,
    "annotations": annotations
}
# Print or write the annotations to a file
print(data)

