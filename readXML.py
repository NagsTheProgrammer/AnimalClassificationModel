import pandas as pd
import numpy as np
import xml.etree.cElementTree as et

tree = et.parse(r'C:\Users\Sumedha\Documents\GitHub\PedestrianDetectionModel\dataset\smallDataset\\annotations\\1.xml')
root = tree.getroot()

objects = []

for name in root.iter('name'):
    print(name.text)