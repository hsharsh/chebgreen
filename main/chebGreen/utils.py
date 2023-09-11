from chebGreen.backend import MATLABPath, os

def runMatlabScript(example, script):
    examplematlab = "\'"+example+"\'"
    matlabcmd = f"{MATLABPath} -nodisplay -nosplash -nodesktop -r \"{script}({examplematlab}); exit;\" | tail -n +11"
    with open("temp.sh", 'w') as f:
        f.write(matlabcmd)
        f.close()
    os.system(f"bash temp.sh")
    os.remove("temp.sh")