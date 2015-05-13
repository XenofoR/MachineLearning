import numpy as np
import matplotlib.pyplot as plt
import math as math
from matplotlib.font_manager import FontProperties
import os
import itertools
allData = []
supData = []
Ydimension = 0
Xdimension = 0
mapeFilter = ['MAPE:', 'Trans:', '\n']
def fileReader(p_path):
    total_files = 0
    print("entered func")
    files = [f for f in os.listdir(p_path) if os.path.isfile(os.path.join(p_path,f))]
    for filename in files:
        inputfile = open(p_path + filename)
        total_files += 1
        print("found file")
        while(1):
            yData = []
            yDataSup = []
            trans = []
            line=inputfile.readline()
            print("reading lines")
            #So hardcoded it ain't even funny
            if line == "Supervised Results:\n":
                 line = inputfile.readline()
                 yData = [f for f in inputfile.readline().split(' ') if f not in mapeFilter]
                 yData.append(filename.split('.',1)[0] + "-Supervised")
                 supData.append(yData[:])
            if line == "Active results:\n":
                print("Found area")
                line = inputfile.readline()
                yData = [f for f in inputfile.readline().split(' ') if f not in mapeFilter]
                yData.append(filename.split('.',1)[0] + "-Active")
                trans = [f for f in inputfile.readline().split(' ') if f not in mapeFilter]
                trans.append(filename.split('.',1)[0] + "-Transduction")
                print("Built y-data")
                allData.append(yData[:])
                #allData.append(trans[:])
                print("Appended y-data")
                break
    return total_files

#Take directory
path = input("Path to target folder: ")
Xdimension = input("x dimension: ")

while not os.path.isdir(path):
    print("Path not found")
    path = input("path to target folder: ")
#Load all files in directory, note: all files means ALL files.
xPos = 1
yPos = 1
curr = 0
FUCKOFF = fileReader(path)
print("reading done")
for sublist in allData:
    myLabel = sublist.pop()
    if xPos == Xdimension:
        xPos = 1
        yPos+=1
    plt.subplot(FUCKOFF, xPos, yPos)
    plt.plot(range(0,len(allData[0])), sublist, marker=',', label=myLabel)
    plt.ylabel("SKOJ")
xPos = 1
yPos = 1
curr =0
for sublist in supData:
    myLabel = sublist.pop()
    if xPos == Xdimension:
        xPos = 1
        yPos+=1
    plt.subplot(FUCKOFF, xPos, yPos)
    plt.ylabel("WHADAYP")
    plt.plot(range(0,len(allData[0])), sublist, marker='+', label=myLabel)
    
print("showing")
fontP = FontProperties()
fontP.set_size('small')
plt.ylim(-1, 1)

plt.show()
