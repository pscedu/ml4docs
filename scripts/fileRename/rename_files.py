#!/usr/bin/env python3
import argparse
import os
import shutil
import xml.etree.ElementTree as ET

"""
Module Docstring
"""

__author__ ="Paola Buitrago"
__version__ = "0.1.0"
__license__ = "MIT"



def parseArgs():
    parser = argparse.ArgumentParser(description='Converts the filenames of images and corresponding labelme annotations.')
    parser.add_argument('--mapFile', type=str,
                    help='Full path of the mapping file.')
    parser.add_argument('--oldFolder', type=str,
                    help='Full path of the folder with the files named according to the old standard.')
    parser.add_argument('--newFolder', type=str,
                    help='Full path of the folder where the new files will be located according to the new naming standard. This folder has two subfolder: "images" and "annotations"')
    args = parser.parse_args()   
    return args 

def loadMappingFile(filename):
    dic = {}
    with open(filename) as f:
       for line in f:
          if len(line.split()) == 2:
              (key, val) = line.split()
              dic[key] = val
    return dic

def listFilesInFolder(folderPath):

    files = []
    folderpath = folderPath + "\Images"
    for r, d, f in os.walk(folderPath):
        #for file in f:
        files = f
        #print(f)
    return files

def copyAndRenameFile(oldDir, oldFileName, newDir, newFileName):
    
    newImagePath=newDir+"Images/"+newFileName
    oldImagePath=oldDir+"Images/"+oldFileName

    newFileRoot = newFileName.split(".")[0]
    oldFileRoot = oldFileName.split(".")[0]
    newAnnotationsFile=newDir+"Annotations/"+newFileRoot+".xml"
    oldAnnotationsFile=oldDir+"Annotations/"+oldFileRoot+".xml"

    # Copying the image
    if not os.path.exists(newImagePath):  # folder exists, file does not
        print("Copying file ", oldImagePath, " to ", newImagePath)    
        shutil.copy(oldImagePath, newImagePath) 
    #if not os.path.exists(newAnnotationsFile):  # folder exists, file does not
    #    print("Copying file ", oldAnnotationsFile, " to ", newAnnotationsFile)
    #    shutil.copy(oldAnnotationsFile, newAnnotationsFile)

    #Load new xml
    if not os.path.exists(newAnnotationsFile):  # folder exists, file does not
        tree = ET.parse(oldAnnotationsFile)
        root = tree.getroot()
    
        for elem in root.iter('filename'):
            elem.text = newFileName
    
        for elem in root.iter('folder'):
            elem.text = "campaign3"

        tree.write(newAnnotationsFile)

def renameFiles(dic, oldFilesList, oldDir, newDir):
    
    nFilesRenamed = 0
    for f in oldFilesList:
        #print (f, dic)
        if f in dic:
            if dic[f] != "-":
                newFileName=dic[f]
                oldFileName=f
                print("Moving and renaming file: ", oldFileName, newFileName)
                copyAndRenameFile(oldDir, oldFileName, newDir, newFileName)
                nFilesRenamed += 1

    print("------> Files renamed: ", nFilesRenamed)

def main():
    """ Main entry point of the app """
    print("Renaming files")
    args = parseArgs()
    mappingFile = args.mapFile
    oldDir = args.oldFolder
    newDir = args.newFolder


    # Loading the file name mappings into a dictionary
    dic = loadMappingFile(mappingFile)

    # Reading the list of Files
    oldFilesList = listFilesInFolder(oldDir)
    
    renameFiles(dic, oldFilesList, oldDir, newDir)

    print(oldFilesList)   
    print(dic)
    print(type(dic))

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
