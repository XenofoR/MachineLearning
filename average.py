import os
import fnmatch              
def CreateFile(p_path):
    numFiles = 0
    for root, dirnames, filenames in os.walk(p_path):
        outputFile = open(root + "\\" + "Average.ave", 'w+')
        numFiles = 0
        asMAE = []
        asMAPE = []
        aaMAE = []
        aaMAPE =[]
        aaTRANS = []
        filterzerozandtext = ['\tMAE:', '\tMAPE:', '0.0', '\n', '\tTrans:' , 'null\n']
        for filename in fnmatch.filter(filenames, '*.result'):
            numFiles+=1
            inputFile = open(root + "\\" + filename, 'r')
            while(1):
                line = inputFile.readline()
                if line == 'END':
                    break
                if line == 'Supervised Results: \n':
                    mae = [f for f in inputFile.readline().split(' ') if f not in filterzerozandtext]
                    mape = [f for f in inputFile.readline().split(' ') if f not in filterzerozandtext]
                    for i in range(len(mae)):
                        if i < len(asMAE):
                            asMAE[i] += float(mae[i])
                            asMAPE[i] += float(mape[i])
                        else:
                            asMAE.append(float(mae[i]))
                            asMAPE.append(float(mape[i]))
                if line == 'Active Results: \n':
                    mae = [f for f in inputFile.readline().split(' ') if f not in filterzerozandtext]
                    mape = [f for f in inputFile.readline().split(' ') if f not in filterzerozandtext]
                    trans = [f for f in inputFile.readline().split(' ') if f not in filterzerozandtext]
                    for i in range(len(mae)):
                        if i < len(aaMAE):
                            aaMAE[i] += float(mae[i])
                            aaMAPE[i] += float(mape[i])
                            aaTRANS[i] += float(trans[i])
                        else:
                            aaMAE.append(float(mae[i]))
                            aaMAPE.append(float(mape[i]))
                            aaTRANS.append(float(trans[i]))
        asMAE = [x / numFiles for x in asMAE]
        asMAPE = [x / numFiles for x in asMAPE]
        aaMAE = [x / numFiles for x in aaMAE]
        aaMAPE = [x / numFiles for x in aaMAPE]
        aaTRANS = [x / numFiles for x in aaTRANS]
        outputFile.write("Supervised results:\n")
        outputFile.write('MAE: ' +' '.join(map(str,asMAE)) +'\n')
        outputFile.write('MAPE: ' +' '.join(map(str,asMAPE))+'\n')
        outputFile.write("Active results:\n")
        outputFile.write('MAE: ' +' '.join(map(str,aaMAE))+'\n')
        outputFile.write('MAPE: ' +' '.join(map(str,aaMAPE))+'\n')
        outputFile.write('Trans: ' +' '.join(map(str,aaTRANS))+'\n')
        outputFile.close()                   
                             
        
path = input("Path to target folder: ")
while not os.path.isdir(path):
    print("Path not found")
    path = input("path to target folder: ")


CreateFile(path)