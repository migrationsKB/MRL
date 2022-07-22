from rdflib import Graph
import os

g = Graph()

for file in os.listdir("../output/mgkb2.0/"):
    filepath = os.path.join("../output/mgkb2.0/", file)
    outputfile = filepath.replace(".xml", ".nt")
    print(filepath, "--->", outputfile)
    g.parse(filepath)
    g.serialize(destination=outputfile, format="nt")

