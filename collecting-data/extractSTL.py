import zipfile
import os
import shutil
baseDir = r"P:\ytp_oasis\collecting-data\downloaded"
outputDir = r"P:\ytp_oasis\collecting-data\outputTest"
zipNameBase = r"OrthoCAD_Export_"
stlNameSuffix = r"_shell_lockedocclusion_ul.stl"

for uuid in os.listdir(baseDir)[:3]:
    print("UUID", uuid)
    for zipName in os.listdir(f"{baseDir}\\{uuid}"):
        exportId = zipName[zipName.rfind('_') + 1:-4]
        exportFilePath = f"{baseDir}\\{uuid}\\{zipNameBase}{exportId}.zip"
        stlFileName = f"{exportId}{stlNameSuffix}"
        outputFilePath = f"{outputDir}\\{uuid}_{exportId}.stl"
        with zipfile.ZipFile(exportFilePath, "r") as zipFile:
            zipFile.extract(stlFileName)
        shutil.move(stlFileName, outputFilePath)
        print(f"Extracted completed to {outputFilePath}")