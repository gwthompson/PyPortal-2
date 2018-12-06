import xlsxwriter
import numpy as np

class ExcelExporter():
    def __init__(self):
        #self.nSubjects = nSubjects
        #self.nFields = nFields
        self.curRow = 0
        self.curCol = 0
        self.workbook = xlsxwriter.Workbook('Results.xlsx')
        self.cell_format1 = self.workbook.add_format()
        self.cell_format1.set_bg_color('green')
        self.cell_format2 = self.workbook.add_format()
        self.cell_format2.set_bg_color('yellow')
        self.cell_format3 = self.workbook.add_format()
        self.cell_format3.set_bg_color('orange')
        self.cell_format4 = self.workbook.add_format()
        self.cell_format4.set_bg_color('red')
        self.worksheet = self.workbook.add_worksheet()

    def addHeader(self, header):
        self.worksheet.write(0, 0, header)

    def addNewSubject(self, subjName):
        self.curRow+=1
        self.curCol=1
        self.worksheet.write(self.curRow, 0, subjName)

    def addPairDataField(self, fieldName, dataVal1, dataVal2):

        self.worksheet.write(0, self.curCol, fieldName)
        if (dataVal1 > 75) and (dataVal2 > 75):
            self.worksheet.write(self.curRow, self.curCol, str(dataVal1) + '/' + str(dataVal2), self.cell_format1)
        elif (dataVal1 > 60) and (dataVal2 > 60):
            self.worksheet.write(self.curRow, self.curCol, str(dataVal1) + '/' + str(dataVal2), self.cell_format2)
        elif (dataVal1 > 50) and (dataVal2 > 50):
            self.worksheet.write(self.curRow, self.curCol, str(dataVal1) + '/' + str(dataVal2), self.cell_format3)
        else:
            self.worksheet.write(self.curRow, self.curCol, str(dataVal1) + '/' + str(dataVal2), self.cell_format4)
        self.curCol+=1

    def addFinisher(self, text, val):
        self.curRow+=1
        self.worksheet.write(self.curRow, 0, text + ' ' + str(val))

    def Finish(self):
        self.workbook.close()