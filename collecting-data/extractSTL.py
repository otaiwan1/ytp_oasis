import zipfile
import os
import shutil
baseDir = r"/tmp2/b14902031/ytp_oasis/collecting-data/scanfiles"
outputDir = r"/tmp2/b14902031/ytp_oasis/collecting-data/stlFiles"
zipNameBase = r"OrthoCAD_Export_"
stlNameSuffix = r"_shell_lockedocclusion_ul.stl"

for uuid in os.listdir(baseDir):
    print("UUID", uuid)
    for zipName in os.listdir(f"{baseDir}/{uuid}"):
        exportId = zipName[zipName.rfind('_') + 1:-4]
        exportFilePath = f"{baseDir}/{uuid}/{zipNameBase}{exportId}.zip"
        stlFileName = f"{exportId}{stlNameSuffix}"
        outputFilePath = f"{outputDir}/{uuid}_{exportId}.stl"
        try:
            with zipfile.ZipFile(exportFilePath, "r") as zipFile:
                zipFile.extract(stlFileName)
            shutil.move(stlFileName, outputFilePath)
            print(f"Extracted completed to {outputFilePath}")
        except Exception as e:
            with open("error.txt", "a", encoding = "utf-8") as file:
                file.write(f"{e}\n")