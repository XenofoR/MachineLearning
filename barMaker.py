import numpy as np
import matplotlib.pyplot as plt
import math as math
from matplotlib.font_manager import FontProperties
from operator import itemgetter
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
            if line == "Supervised results:\n":
                 print("super")
                 line = inputfile.readline()
                 yData = [f for f in inputfile.readline().split(' ') if f not in mapeFilter]
                 yData.append("Supervised")#filename.split('.',1)[0] + "-Supervised")
                 supData.append(yData[:])
            if line == "Active results:\n":
                print("Found area")
                line = inputfile.readline()
                yData = [f for f in inputfile.readline().split(' ') if f not in mapeFilter]
                yData.append(filename.split('.')[0])
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

while not os.path.isdir(path):
    print("Path not found")
    path = input("path to target folder: ")
#Load all files in directory, note: all files means ALL files.
xPos = 1
yPos = 1
FUCKOFF = fileReader(path)
print("reading done")

fig, ax = plt.subplots()
firstVal = []
midVal = []
lastVal = []
nameinfo = []
numfiles = 0
allData.sort( key=lambda x : float(x[-1]))
for sublist in allData:
    numfiles+=1
    fulkod = sublist.pop()
    if fulkod != '0':
        nameinfo.append(fulkod)
    else:
        nameinfo.append('0.1')
    firstVal.append(float(sublist[0]))
    midVal.append(float(sublist[math.floor(len(sublist)/2)]))
    lastVal.append(float(sublist[-1]))
print("hi")
N = numfiles
width = 0.25
ind = np.arange(N)
ax.set_xticks(ind+width*1.5)
ax.set_xticklabels( nameinfo )
ax.set_ylabel("SMAPE")
ax.set_xlabel("Alphavalue")
rect1 = ax.bar(ind, firstVal,width, color='r')
rect2 = ax.bar(ind + width, midVal,width, color='y')
rect3 = ax.bar(ind + width * 2, lastVal,width, color='g')

plt.show()