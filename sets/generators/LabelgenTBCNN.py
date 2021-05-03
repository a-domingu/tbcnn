import os
import ast
import shutil

folder = os.getcwd()
folder2 = 'C:\\Users\\dminetca\\OneDrive - Capgemini\\Desktop\\generatorfind\\projects\\withoutgen'

i=0

for _root, _dirs, files in os.walk(folder2):
    for filepath in files:
            if filepath.endswith(".py") and not filepath.startswith("__init__"):# and not filepath.startswith("gen"):
                try:
                    i+=1
                    tree = ast.parse(open(_root+'\\'+filepath, encoding='utf-8').read())
                    #for node in ast.walk(tree):
                        #if node.__class__.__name__ == "Yield":
                    shutil.copy(_root+'\\'+filepath, "noprov\\"+filepath)
                    print(i, "archivo sin generator", filepath)
                            #print("Generator en ", filepath)
                            #break
                except FileNotFoundError:
                    #pass
                    print("No se pudo acceder a ", filepath)
                except SyntaxError:
                    print("Syntaxerror en:", filepath)