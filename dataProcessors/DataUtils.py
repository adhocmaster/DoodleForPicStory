class DataUtils:


    def convertFilenameToClass(self, fileNameNoExt):
        return fileNameNoExt.replace("full_numpy_bitmap_", "").replace(" ", "_")
    

    def convertClassToFilename(self, className):
        return "full_numpy_bitmap_" + className.replace("_", " ")
        
