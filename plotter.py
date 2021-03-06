import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import itertools
allData = []
mapeFilter = ['MAPE:', 'Trans:', '\n']
def fileReader(p_path):
    print("entered func")
    files = [f for f in os.listdir(p_path) if os.path.isfile(os.path.join(p_path,f))]
    for filename in files:
        inputfile = open(p_path + filename)
        print("found file")
        while(1):
            yData = []
            trans = []
            line=inputfile.readline()
            print("reading lines")
            #So hardcoded it ain't even funny
            if line == "Active results:\n":
                print("Found area")
                line = inputfile.readline()
                yData = [f for f in inputfile.readline().split(' ') if f not in mapeFilter]
                yData.append(filename.split('.',1)[0] + "-Induction")
                trans = [f for f in inputfile.readline().split(' ') if f not in mapeFilter]
                trans.append(filename.split('.',1)[0] + "-Transduction")
                print("Built y-data")
                #allData.append(yData[:])
                allData.append(trans[:])
                print("Appended y-data")
                break;

#Take directory
path = input("Path to target folder: ")

while not os.path.isdir(path):
    print("Path not found")
    path = input("path to target folder: ")
#Load all files in directory, note: all files means ALL files.
fileReader(path)
print("reading done")
ax = plt.subplot(111)
for sublist, skit in zip(allData, itertools.cycle((',', '+', '.' , 'o', '*'))):
    print(sublist)
    myLabel = sublist.pop()
    ax.plot(range(0,len(allData[0])), sublist, marker=skit, label=myLabel)
print("showing")
fontP = FontProperties()
fontP.set_size('small')
plt.ylim(-1, 1)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5) ,prop=fontP)
plt.show()
