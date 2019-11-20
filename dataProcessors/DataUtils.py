class DataUtils:


    def convertFilenameToClass(self, fileNameNoExt):
        return fileNameNoExt.replace("full_numpy_bitmap_", "").replace(" ", "_")
