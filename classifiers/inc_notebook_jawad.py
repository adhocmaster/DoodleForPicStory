import os, sys
currentFolder = os.path.abspath('')
try:
    sys.path.remove(str(currentFolder))
except ValueError: # Already removed
    pass

projectFolder = 'F:/myProjects/tfKeras/UCSC/Opensource/DoodleForPicStory/'
projectFolder = 'C:/TFModels/DoodleForPicStory/'
sys.path.append(str(projectFolder))
os.chdir(projectFolder)
print( f"current working dir{os.getcwd()}")