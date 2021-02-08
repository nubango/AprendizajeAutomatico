from pandas.io.parsers import read_csv

def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    return valores.astype(float)