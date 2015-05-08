import shutil
import os
import fnmatch 
def CopyAndRename(p_srcDir, p_dstDir):
    for root, dirnames, filenames in os.walk(p_srcDir):
        for filename in fnmatch.filter(filenames, '*.ave'):
            shutil.copyfile(root+ "\\" + filename, p_dstDir + "\\" + os.path.basename(os.path.normpath(root)) + ".ave")

src = input("Path to source folder: ")
while not os.path.isdir(src):
    print("Path not found")
    src = input("path to source folder: ")

dst = input("Path to destination folder: ")
while not os.path.isdir(dst):
    print("Path not found")
    dst = input("path to destination folder: ")

CopyAndRename(src,dst)
