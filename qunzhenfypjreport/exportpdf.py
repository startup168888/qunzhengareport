# Python program to convert
# text file to pdf file

from fpdf import FPDF
class PDF(FPDF):
    def header(self):
        # Logo
        self.image('WClogo.jpg', 10, 8, 21)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'eCommerce Advisory Report', 0, 0,  'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

# save FPDF() class into
# a variable pdf
pdf = PDF()
pdf.alias_nb_pages()
# Add a page
pdf.add_page()

# set style and size of font
# that you want in the pdf
pdf.set_font("Arial", 'B', size = 14)

# open the text file in read mode
f = open("test2.txt", "r")

# insert the texts in pdf
pdf.cell( 200, 10, txt= "Analytics Report", ln = 1, align='C')
#pdf.set_font("Arial", 'B', size = 14)
#add website hyperlink as text in pdf
#pdf.cell( 200, 10, txt= "https://e-commerce-jay.herokuapp.com/index.html", link="https://e-commerce-jay.herokuapp.com/index.html", ln = 1, align='C')
for x in f:
    pdf.set_font("Arial", size = 12)
    pdf.cell(200, 7, txt = x, ln = 1, align = 'L')
pdf.add_page()
pdf.set_font("Arial", 'B', size = 14)
pdf.cell( 200, 10, txt= "Conclusion", ln = 1, align='C')
# save the pdf with name .pdf
pdf.output("report.pdf")


