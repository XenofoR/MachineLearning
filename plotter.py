import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
allData = []
mapeFilter = ['MAPE:']
def fileReader(p_path):
    print("entered func")
    files = [f for f in os.listdir(p_path) if os.path.isfile(os.path.join(p_path,f))]
    for filename in files:
        inputfile = open(p_path + filename)
        print("found file")
        while(1):
            yData = []
            line=inputfile.readline()
            print("reading lines")
            #So hardcoded it ain't even funny
            if line == "Active results:\n":
                print("Found area")
                line = inputfile.readline()
                yData = [f for f in inputfile.readline().split(' ') if f not in mapeFilter]
                yData.append(filename.split('.',1)[0])
                print("Built y-data")
                allData.append(yData[:])
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
for sublist in allData:
    myLabel = sublist.pop()
    ax.plot(range(0,len(allData[0])), sublist, label=myLabel)
print("showing")
fontP = FontProperties()
fontP.set_size('small')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5) ,prop=fontP)
plt.show()
